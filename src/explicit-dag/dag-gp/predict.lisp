(uiop:define-package #:dag-gp/explicit-dag/dag-gp/predict
    (:use #:cl
          #:mgl-mat
          #:dag-gp/output
          #:dag-gp/lapack
          #:dag-gp/quadrature
          #:dag-gp/explicit-dag/networked-base/all
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/linear-parent/all
          #:dag-gp/explicit-dag/dag-gp/gp)
  (:shadowing-import-from #:lla
                          #:invert)
  (:import-from #:array-operations)
  (:export #:predict
           #:update-combined-distributions
           #:predict-combined))

(in-package #:dag-gp/explicit-dag/dag-gp/predict)


;; This looks ridiculously complicated, but this whole file is really only 3 or 4
;; lines of MATLAB. It's hard to parse and needs cleaning up.

(defmethod predict ((gp dag-gp) pred-locs)
  ;; Explicitly construct the matrix between the observation and prediction locations
  (let ((output-dim (output-dim gp))
        adjacency a-array a-mat
        (total-obs 0)
        (total-pred 0)
        y-obs
        y-pred
        Kyy-obs
        Kyy-pred
        Kyy-pred-obs
        Kyy-pred-obs-copy
        (n-pred (list-length pred-locs))
        (pred-loc-indices nil)
        (output-loc-indices nil)
        current-row current-col
        pred-mean pred-cov)

    ;; Make A matrix
    (setf adjacency (make-array (list output-dim output-dim)
                                :element-type 'double-float
                                :initial-element 0d0))
    (loop for i below output-dim
          for constituent-gp in (constituent-gps gp)
          for parents = (cdr (assoc i (factors gp)))
          do (setf (aref adjacency i i) 1d0)
             (loop for parent in parents
                   for param in (parent-params constituent-gp)
                   do (setf (aref adjacency i parent) (- (aref param 0)))))
    (setf a-array (invert adjacency)
          a-mat (array-to-mat a-array :ctype :double))

    ;; Remake the covariances for the constituents
    (loop for constituent-gp in (constituent-gps gp) do
      (incf total-obs (n-true-obs constituent-gp))
      (loop for loc in pred-locs do
        (unless (member loc (true-obs-locs constituent-gp) :test #'equalp)
          (incf total-pred)))
      (preprocess-prediction constituent-gp pred-locs n-pred))

    ;; Make the observed y mean and covariance
    (setf y-obs (make-array total-obs :element-type 'double-float)
          Kyy-obs (make-array (list total-obs total-obs)
                              :initial-element 0d0
                              :element-type 'double-float)
          Kyy-pred (make-array (list total-pred total-pred)
                               :initial-element 0d0
                               :element-type 'double-float)
          Kyy-pred-obs (make-array (list total-pred total-obs)
                                   :initial-element 0d0
                                   :element-type 'double-float))
    ;; To make life easier, make a list of (output loc-index) pairs of true observations
    ;; In all GPs, the observed locations are listed in the same order
    (setf current-row 0)
    (loop for loc in (constituent-locs gp)
          for outputs = (gethash loc (constituent-obs gp))
          ;; Constituent-loc-obs holds true observations
          for loc-outputs = (gethash loc (constituent-loc-obs gp))
          do (loop for output in outputs
                   for con-gp = (nth output (constituent-gps gp))
                   do (setf output-loc-indices
                            (nconc output-loc-indices
                                   (list (list output
                                               (position loc
                                                         (obs-locs con-gp)
                                                         :test #'equalp))))
                            (aref y-obs current-row)
                            (cdr (assoc output loc-outputs)))
                      (incf current-row)))
    ;;(format t "output-loc-indices = ~a~%" output-loc-indices)
    ;; Order in terms of all locs for each sequential output
    ;; Make a list of (output loc-index) pairs for each unobserved value
    (loop for output below (output-dim gp)
          for constituent-gp in (constituent-gps gp)
          do (loop for loc in pred-locs
                   for loc-index from 0
                   do (unless (member loc (true-obs-locs constituent-gp) :test #'equalp)
                        (setf pred-loc-indices
                              (nconc pred-loc-indices
                                     (list (list output loc-index)))))))
    ;;(format t "pred-loc-indices = ~a~%" pred-loc-indices)

    ;; Construct Kyy
    (loop for (row-output row-loc-index) in output-loc-indices
          for row from 0
          do (loop for (col-output col-loc-index) in output-loc-indices
                   for col from 0 upto row
                   do (loop for output below (output-dim gp)
                            for con-gp in (constituent-gps gp)
                            do (with-facets ((Kff ((Kff con-gp) 'array :direction :input)))
                                 (incf (aref Kyy-obs row col)
                                       (* (aref a-array row-output output)
                                          (aref a-array col-output output)
                                          (aref Kff row-loc-index col-loc-index))))))
             (loop for col below row do
               (setf (aref Kyy-obs col row) (aref Kyy-obs row col))))
    ;(format t "Kyy-obs = ~a~%" Kyy-obs)

    (loop for (row-output row-loc-index) in pred-loc-indices
          for row from 0
          do (loop for (col-output col-loc-index) in pred-loc-indices
                   for col from 0 upto row
                   do (loop for output below (output-dim gp)
                            for con-gp in (constituent-gps gp)
                            do (with-facets ((pred-Kff ((pred-Kff con-gp) 'array
                                                        :direction :input)))
                                 (incf (aref Kyy-pred row col)
                                       (* (aref a-array row-output output)
                                          (aref a-array col-output output)
                                          (aref pred-Kff row-loc-index col-loc-index))))))
             (loop for col below row do
               (setf (aref Kyy-pred col row) (aref Kyy-pred row col))))
    ;(format t "Kyy-pred = ~a~%" Kyy-pred)

    (loop for (row-output row-loc-index) in pred-loc-indices
          for row from 0
          do (loop for (col-output col-loc-index) in output-loc-indices
                   for col from 0
                   do (loop for output below (output-dim gp)
                            for con-gp in (constituent-gps gp)
                            do (with-facets ((pred-obs-Kff ((pred-obs-Kff con-gp) 'array
                                                            :direction :input)))
                                 (incf (aref Kyy-pred-obs row col)
                                       (* (aref a-array row-output output)
                                          (aref a-array col-output output)
                                          (aref pred-obs-Kff row-loc-index col-loc-index)))))))
    ;(format t "Kyy-pred-obs = ~a~%" Kyy-pred-obs)


    ;; Make everything as mats
    (setf y-pred (make-mat total-pred :ctype :double)
          y-obs (array-to-mat y-obs :ctype :double)
          Kyy-obs (array-to-mat Kyy-obs :ctype :double)
          Kyy-pred (array-to-mat Kyy-pred :ctype :double)
          Kyy-pred-obs (array-to-mat Kyy-pred-obs :ctype :double)
          Kyy-pred-obs-copy (copy-mat Kyy-pred-obs))

    ;; Make the prediction
    (potrf! Kyy-obs :uplo #\L :n total-obs :lda total-obs)
    (potrs! Kyy-obs Kyy-pred-obs-copy
            :uplo #\L :n total-obs :nrhs total-pred
            :lda total-obs :ldb total-obs :transpose-b? t)
    (gemm! -1d0 Kyy-pred-obs-copy Kyy-pred-obs 1d0 Kyy-pred
           :transpose-b? t :m total-pred :n total-pred :k total-obs
           :lda total-obs :ldb total-obs :ldc total-pred)
    (gemv! 1d0 Kyy-pred-obs-copy y-obs 0d0 y-pred
           :m total-pred :n total-obs :lda total-obs)
    
    ;; Fill in any values that are known
    (when (equal total-pred (* output-dim n-pred))
      (return-from predict (list y-pred Kyy-pred)))

    (setf pred-mean (make-array (* output-dim n-pred)
                                :element-type 'double-float)
          pred-cov (make-array (list (* output-dim n-pred)
                                     (* output-dim n-pred))
                               :initial-element 0d0
                               :element-type 'double-float))
    (setf current-row 0)
    (with-facets ((mu (y-pred 'backing-array :direction :input))
                  (K (Kyy-pred 'array :direction :input)))
      (loop for row-output below (output-dim gp)
            for row-gp in (constituent-gps gp)
            do (loop for row-loc in pred-locs
                     for row-output-index from 0
                     for row-index = (+ (* row-output n-pred) row-output-index)
                     do (if (member row-loc (true-obs-locs row-gp) :test #'equalp)
                            (progn
                              (setf (aref pred-mean row-index)
                                    (cdr (assoc row-output
                                                (gethash row-loc (constituent-loc-obs gp))))))
                            (progn
                              (setf (aref pred-mean row-index)
                                    (aref mu current-row))
                              (block col-loop
                                (setf current-col 0)
                                (loop for col-output from 0 upto row-output
                                      for col-gp in (constituent-gps gp)
                                      do (loop for col-loc in pred-locs
                                               for col-output-index from 0
                                               for col-index = (+ (* col-output n-pred)
                                                                  col-output-index)
                                               do (when (> col-index row-index)
                                                    (return-from col-loop nil))
                                                  (unless (member col-loc (true-obs-locs col-gp)
                                                                  :test #'equalp)
                                                    (setf (aref pred-cov row-index col-index)
                                                          (aref K current-row current-col))
                                                    (incf current-col)
                                                    (unless (equal row-index col-index)
                                                      (setf (aref pred-cov col-index row-index)
                                                            (aref pred-cov row-index col-index)))))))
                              (incf current-row))))))
    
    (list (array-to-mat pred-mean :ctype :double)
          (array-to-mat pred-cov :ctype :double))))

(defmethod predict ((gp variational-dag-gp) pred-locs)
  (let ((predictive-posteriors (call-next-method))
        (true-obs))
    
    (when (all-gaussian-p gp)
      (return-from predict
        (gaussian-variational-predict gp pred-locs predictive-posteriors)))
    
    (let ((quad-sets nil)
          mean-arrays cov-arrays
          mean variance)
      ;; Determine a factor ordering
      (unless (factor-ordering-up-to-date gp)
        (solve-factor-ordering gp))

      ;; Make all the quad sets
      (dolist (i pred-locs)
        (declare (ignore i))
        (setf quad-sets (nconc quad-sets (list (make-instance 'quad-set)))))
      
      ;; Expecting predictive-posteriors to be a list of (mean var) for each gp
      (loop for variable in (factor-ordering gp)
            for parents = (cdr (assoc variable (factors gp)))
            for constituent-gp = (nth variable (constituent-gps gp))
            for output = (nth variable (outputs gp))
            for parent-params = (parent-params constituent-gp)
            for parent-outputs = (parent-outputs constituent-gp)
            for predictive-posterior = (nth variable predictive-posteriors)
        do (destructuring-bind (pred-mu pred-cov) predictive-posterior

             ;; Extract mean and cov arrays
             (if (listp pred-mu)
                 (setf mean-arrays (loop for mu in pred-mu
                                         collect (mat-to-array mu))
                       cov-arrays (loop for cov in pred-cov
                                        collect (mat-to-array cov)))
                 (setf mean-arrays (mat-to-array pred-mu)
                       cov-arrays (mat-to-array pred-cov)))

             ;; Loop through locations for this output
               (loop for loc in pred-locs
                     for loc-ind from 0
                     for quad-set in quad-sets
                     for loc-pos = (position loc (true-obs-locs constituent-gp)
                                             :test #'equalp)
                 do (if loc-pos
                        ;; If there has been an observation at the location, then add
                        ;; that variable only
                        (progn
                          (setf true-obs (nth loc-pos (true-obs constituent-gp)))
                          (flet ((quadrature-basis (parent-vals)
                                   (declare (ignore parent-vals))
                                   (list (list true-obs 1d0))))
                            (add-quad-variable quad-set
                                               variable
                                               #'quadrature-basis
                                               parents)))
                        
                        (progn
                          (if (listp mean-arrays)
                              (setf mean (loop for mu in mean-arrays
                                               collect (aref mu loc-ind))
                                    variance (loop for cov in cov-arrays
                                                   collect (aref cov loc-ind
                                                                 loc-ind)))
                              (setf mean (aref mean-arrays loc-ind)
                                    variance (aref cov-arrays loc-ind loc-ind)))
                      
                          (flet ((parameterized-quadrature-basis (parent-vals)
                                   (make-quadrature-basis output mean
                                                          variance
                                                          parent-vals
                                                          (list-length parents)
                                                          parent-outputs
                                                          parent-params)))
                            (add-quad-variable quad-set
                                               variable
                                               #'parameterized-quadrature-basis
                                               parents)))))))
      quad-sets)))


(defun gaussian-variational-predict (gp pred-locs predictive-posteriors)
  ;; Make the predictive posteriors
  (let* ((output-dim (output-dim gp))
         (n-fe-rows (* 2 output-dim))
         adjacency
         a-array
         a-mat
         middle
         (Keey-temp (make-mat (list output-dim output-dim)
                              :ctype :double))
         (Kefy (make-mat (list output-dim output-dim)
                         :ctype :double))
         mean-temp
         row-1 row-2
         Kfefey mufey
         muy
         (Kyy-intermediate (make-mat (list output-dim n-fe-rows)
                                     :ctype :double))
         Kyy)
    
    ;; Make A matrix TODO: Make own function?
    (setf adjacency (make-array (list output-dim output-dim)
                                :element-type 'double-float
                                :initial-element 0d0))
    (loop for i below output-dim
          for constituent-gp in (constituent-gps gp)
          for parents = (cdr (assoc i (factors gp)))
          do (setf (aref adjacency i i) 1d0)
             (loop for parent in parents
                   for param in (parent-params constituent-gp)
                   do (setf (aref adjacency i parent) (- (aref param 0)))))
    (setf a-array (invert adjacency)
          a-array (aops:stack-cols a-array a-array)
          a-mat (array-to-mat a-array :ctype :double))
    
    ;; TODO: skip if everything is known
    (loop for loc in pred-locs
          for loc-ind from 0
          for obs = (gethash loc (constituent-obs gp))
          for loc-obs = (gethash loc (constituent-loc-obs gp))
          for n-obs = (list-length obs)
          do (destructuring-bind (mufyi mueyi Kffyi Kefyi Keeyi)
                 (make-local-conditional-fe gp a-array loc-ind obs loc-obs
                                            n-obs output-dim n-fe-rows)
               (destructuring-bind (mufy Kffy)
                   (make-conditional-f predictive-posteriors loc-ind output-dim)

                 ;; middle <- Kffy - Kffyi
                 (setf middle (m- Kffy Kffyi))
                 
                 ;; Kefyi <- Kefyi Kffyi^-1 
                 (potrf! Kffyi :uplo #\L :n output-dim :lda output-dim)
                 (potrs! Kffyi Kefyi
                         :uplo #\L :n output-dim :lda output-dim
                         :nrhs output-dim :ldb output-dim :transpose-b? t)

                 ;; Keeyi <- Keey
                 (gemm! 1d0 Kefyi middle 0d0 Keey-temp
                        :m output-dim :n output-dim :k output-dim
                        :lda output-dim :ldb output-dim :ldc output-dim)
                 (gemm! 1d0 Keey-temp Kefyi 1d0 Keeyi
                        :transpose-b? t :m output-dim :n output-dim :k output-dim
                        :lda output-dim :ldb output-dim :ldc output-dim)

                 ;; mueyi <- muey = mueyi + Kefyi Kffyi^-1 (mufy - mufyi)
                 (setf mean-temp (m- mufy mufyi))
                 (gemv! 1d0 Kefyi mean-temp 1d0 mueyi
                        :m output-dim :n output-dim :lda output-dim)
                 
                 ;; Kefy <- Kefyi Kffyi^-1 Kffy
                 (gemm! 1d0 Kefyi Kffy 0d0 Kefy
                        :m output-dim :n output-dim :k output-dim
                        :lda output-dim :ldb output-dim :ldc output-dim)

                 
                 ;; Stack the posteriors together
                 (setf mufey (stack 0 (list mufy mueyi))
                       row-1 (stack 1 (list Kffy (transpose Kefy)))
                       row-2 (stack 1 (list Kefy Keeyi))
                       Kfefey (stack 0 (list row-1 row-2)))

                 ;; Multiply to get the final results
                 (setf muy (make-mat output-dim :ctype :double))
                 (gemv! 1d0 a-mat mufey 0d0 muy
                        :m output-dim :n n-fe-rows :lda n-fe-rows)
                 (setf Kyy (make-mat (list output-dim output-dim) :ctype :double))
                 (gemm! 1d0 a-mat Kfefey 0d0 Kyy-intermediate
                        :m output-dim :n n-fe-rows :k n-fe-rows
                        :lda n-fe-rows :ldb n-fe-rows :ldc n-fe-rows)
                 (gemm! 1d0 Kyy-intermediate a-mat 0d0 Kyy
                        :transpose-b? t :m output-dim :n output-dim :k n-fe-rows
                        :lda n-fe-rows :lda n-fe-rows :ldc output-dim)))

          collect (list muy Kyy))))


(defun make-local-conditional-fe (gp a-mat loc-ind obs loc-obs n-obs
                                  output-dim n-fe-rows)
  (let ((output-dim (output-dim gp))
        (mufe (make-array n-fe-rows
                          :element-type 'double-float
                          :initial-element 0d0))
        (muf (make-array output-dim
                         :element-type 'double-float))
        (mue (make-array output-dim
                         :element-type 'double-float))
        (muy (make-mat n-obs
                       :ctype :double))
        (obs-y (make-array n-obs
                           :element-type 'double-float))
        (Kfefe (make-array (list n-fe-rows n-fe-rows)
                           :element-type 'double-float
                           :initial-element 0d0))
        (Kff (make-array (list output-dim output-dim)
                         :element-type 'double-float))
        (Kef (make-array (list output-dim output-dim)
                         :element-type 'double-float))
        (Kee (make-array (list output-dim output-dim)
                         :element-type 'double-float))
        (Kfey (make-mat (list n-fe-rows n-obs)
                        :ctype :double))
        Kfey-copy
        (Kyy (make-mat (list n-obs n-obs)
                       :ctype :double))
        ;; TODO: Cache slices
        (a-sub (array-to-mat
                (slice-a-matrix a-mat obs n-fe-rows)
                :ctype :double)))
    (loop for row below output-dim
          for output in (outputs gp)
          for constituent-gp in (constituent-gps gp)
          do (with-facets ((cov-array ((pred-Kff constituent-gp)
                                       'array
                                       :direction :input)))
               (setf (aref Kfefe row row)
                     (aref cov-array loc-ind loc-ind)
                     (aref Kfefe (+ row output-dim) (+ row output-dim))
                     (exp (aref (output-params output) 0)))))
    (loop for obs-ind in obs
          for i below n-obs
          do (setf (aref obs-y i) (cdr (assoc obs-ind loc-obs))))

    (setf mufe (array-to-mat mufe :ctype :double))
    (gemv! 1d0 a-sub mufe 0d0 muy
           :m n-obs :n n-fe-rows :lda n-fe-rows)
    (setf obs-y (array-to-mat obs-y :ctype :double))
    (setf Kfefe (array-to-mat Kfefe :ctype :double))
    (gemm! 1d0 Kfefe a-sub 0d0 Kfey
           :transpose-b? t
           :m n-fe-rows :n n-obs :k n-fe-rows
           :lda n-fe-rows :ldb n-fe-rows :ldc n-obs)
    (gemm! 1d0 a-sub Kfey 0d0 Kyy
           :m n-obs :n n-obs :k n-fe-rows
           :lda n-fe-rows :ldb n-obs :ldc n-obs)

    ;; Condition on the observations
    (potrf! Kyy :uplo #\L :n n-obs :lda n-obs)
    (setf Kfey-copy (copy-mat Kfey))
    ;; Kfy <- Kfy Kyy^-1
    (potrs! Kyy Kfey
            :uplo #\L :n n-obs :lda n-obs
            :nrhs n-fe-rows :ldb n-obs :transpose-b? t)
      ;;; Kff <- Kff - Kfy Kyy^-1 Kyf
    (gemm! -1d0 Kfey Kfey-copy 1d0 Kfefe
           :transpose-b? t
           :m n-fe-rows :n n-fe-rows :k n-obs
           :lda n-obs :ldb n-obs :ldc n-fe-rows)

    (setf obs-y (m- obs-y muy))
    (gemv! 1d0 Kfey obs-y 1d0 mufe
           :m n-fe-rows :n n-obs :lda n-obs)

    ;; Split up the mats
    ;; TODO: Could potentially solve for each part separately
    (with-facets ((mufe-array (mufe 'backing-array :direction :input))
                  (Kfefe-array (Kfefe 'array :direction :input)))
      (dotimes (i output-dim)
        (setf (aref muf i) (aref mufe-array i)
              (aref mue i) (aref mufe-array (+ i output-dim)))
        (dotimes (j output-dim)
          (setf (aref Kff i j) (aref Kfefe-array i j)
                (aref Kef i j) (aref Kfefe-array (+ i output-dim) j)
                (aref Kee i j) (aref Kfefe-array (+ i output-dim)
                                     (+ j output-dim))))))

    (list (array-to-mat muf :ctype :double)
          (array-to-mat mue :ctype :double)
          (array-to-mat Kff :ctype :double)
          (array-to-mat Kef :ctype :double)
          (array-to-mat Kee :ctype :double))))


(defun make-conditional-f (predictive-posteriors loc-ind output-dim)
  (let ((muf (make-array output-dim
                         :element-type 'double-float))
        (Kff (make-array (list output-dim output-dim)
                         :element-type 'double-float
                         :initial-element 0d0)))
    (loop for row below output-dim
          for posterior in predictive-posteriors
          do (destructuring-bind (pred-mean pred-cov) posterior
               (with-facets ((mean-array (pred-mean 'backing-array
                                                    :direction :input))
                             (cov-array (pred-cov 'array :direction :input)))
                 (setf (aref muf row)
                       (aref mean-array loc-ind)
                       (aref Kff row row)
                       (aref cov-array loc-ind loc-ind)))))

    (list (array-to-mat muf :ctype :double)
          (array-to-mat Kff :ctype :double))))


;; I can't believe there isn't an easy way to do this. Really, lisp?
(defun slice-a-matrix (a-mat rows n-cols)
  (let ((sub-a (make-array (list (list-length rows) n-cols)
                           :element-type 'double-float)))
    (loop for row in rows
          for i from 0
          do (dotimes (col n-cols)
               (setf (aref sub-a i col)
                     (aref a-mat row col))))
    sub-a))


(defgeneric update-combined-distributions (gp)
  (:documentation "Predicts the values of the outputs at each input location, and updates information for the constutuents.")
  (:method ((gp dag-gp))
    ;; If closed downwards, then copy the observations
    (when (dag-closed-downwards-p gp)
      (error "Should not enter update-combined-distributions.")
      ;; (unless (factor-ordering-up-to-date gp)
      ;;   (solve-factor-ordering gp))
      
      ;; (setf (combined-dists gp)
      ;;       (loop for constituent-gp in (constituent-gps gp)
      ;;             collect (obs constituent-gp)))

      ;; (loop for constituent-gp in (constituent-gps gp)
      ;;       for index from 0
      ;;       do (setf (var-parent-distributions constituent-gp)
      ;;                (loop for factor in (cdr (assoc index (factors gp)))
      ;;                      collect (nth factor (combined-dists gp)))))
      ;; (return-from update-combined-distributions)
      )
    
    (let ((predictions (predict gp (constituent-locs gp))))
      (setf (combined-dists gp) predictions)

      (loop for constituent-gp in (constituent-gps gp)
            for index from 0
            do (setf (var-parent-distributions constituent-gp)
                     (full-sub-distribution-for-factor
                      predictions
                      index
                      (cdr (assoc index (factors gp)))
                      (n-obs constituent-gp))))))

  (:method ((gp variational-dag-gp))
    (let ((predictions (predict gp (constituent-locs gp))))
      (loop for loc in (constituent-locs gp)
            for pred in predictions
            do (setf (gethash loc (combined-dists gp)) pred))

      (if (all-gaussian-p gp)
          ;; When all gaussian, run the sub distribution function
          (loop for constituent-gp in (constituent-gps gp)
                for index from 0
                do (setf (var-parent-distributions constituent-gp) nil)
                   (loop for loc in (obs-locs constituent-gp) do
                     (setf (var-parent-distributions constituent-gp)
                           (nconc (var-parent-distributions constituent-gp)
                                  (list (sub-distribution-for-factor
                                         (gethash loc (combined-dists gp))
                                         index
                                         (cdr (assoc index (factors gp)))))))))

          ;; When not all gaussian, copy and marginalize the quad sets
          (loop for constituent-gp in (constituent-gps gp)
                for index from 0
                for parents = (cdr (assoc index (factors gp)))
                do (setf (var-parent-distributions constituent-gp) nil)
                   (loop for loc in (obs-locs constituent-gp)
                         for pred = (gethash loc (combined-dists gp))
                         do (let ((new-quad (copy-quad-set pred)))
                              ;; Marginalize everything unneeded
                              (loop for var below (output-dim gp) do
                                (unless (or (equal var index)
                                            (member var parents))
                                  (marginalize new-quad var)))
                              (flatten new-quad (list* index parents))
                              (setf (var-parent-distributions constituent-gp)
                                    (nconc (var-parent-distributions constituent-gp)
                                           (list new-quad))))))))))


;; (defgeneric predict-combined (gp pred-locs)
;;   (:documentation "Makes a joint distributions prediction for all pred-locs.")
;;   (:method ((gp variational-dag-gp) pred-locs)
;;     ;; For now, only works when there is a single output-dimension
;;     (assert (equal (output-dim gp) 1))
;;     (let* ((constituent-gp (nth 0 (constituent-gps gp)))
;;            (predictive-posterior
;;              (make-predictive-posteriors constituent-gp
;;                                          pred-locs))
;;            (latent-quad-set (make-instance 'quad-set))
;;            (quad-set (make-instance 'quad-set))
;;            (n-pred-locs (list-length pred-locs))
;;            true-obs
;;            mean-arrays cov-arrays
;;            mean-lists var-lists
;;            (output (output constituent-gp))
;;            parent-bases)
      
;;       (destructuring-bind (pred-mu pred-cov) predictive-posterior
        
;;         ;; Cholesky decompose the covariances
;;         (loop for cov in pred-cov do
;;           (zero-potrf! cov :uplo #\L :n n-pred-locs :lda n-pred-locs))

;;         (setf mean-arrays (loop for mu in pred-mu
;;                                 collect (mat-to-array mu))
;;               cov-arrays (loop for cov in pred-cov
;;                                collect (mat-to-array cov)))
        
;;         ;; Loop through locations
;;         (loop for loc in pred-locs
;;               for loc-ind from 0
;;               for loc-pos = (position loc (true-obs-locs constituent-gp)
;;                                       :test #'equalp)
;;           do (setf mean-lists
;;                    (loop for mu in mean-arrays
;;                          collect (aref mu loc-ind))
;;                    var-lists
;;                    (loop for cov in cov-arrays
;;                          collect (make-array (1+ loc-ind)
;;                                              :element-type 'double-float
;;                                              :displaced-to cov
;;                                              :displaced-index-offset
;;                                              (* loc-ind n-pred-locs))))
;;              (format t "mean-lists = ~a~%" mean-lists)
;;              (format t "var-lists = ~a~%" var-lists)

;;              (flet ((quadrature-basis (parent-vals)
;;                       (let (f-list
;;                             den
;;                             bases)
;;                         (format t "parent-vals = ~a~%" parent-vals)
;;                         (if parent-vals
;;                             (setf bases (loop for cov-i in var-lists
;;                                               for parent-val in parent-vals
;;                                               collect (loop for par-val in parent-val
;;                                                             for cov-el across cov-i
;;                                                             sum (* par-val cov-el))))
;;                             (setf bases (make-list (n-gps output)
;;                                                    :initial-element 0d0)))
;;                         (format t "bases = ~a~%" bases)
;;                         (loop for point in (flattened (quadrature-set output))
;;                               for weight in (weights (quadrature-set output))
;;                           do (setf den 1d0
;;                                    f-list (loop for point-i in point
;;                                                 for mean-i in mean-lists
;;                                                 for cov-i in var-lists
;;                                                 for var-i = (aref cov-i loc-ind)
;;                                                 for base-i in bases
;;                                                 for f = (+ base-i mean-i
;;                                                            (* (sqrt 2) var-i
;;                                                               point-i))
;;                                                 collect f))
;;                              (format t "f-list = ~a~%" f-list)
;;                              (error "ffgw")
;;                               collect (list f-list weight)))))

;;                (add-quad-variable latent-quad-set loc-ind #'quadrature-basis))
             






;;                  ;; (if loc-pos
;;                  ;;     ;; If there has been an observation at the location, then add
;;                  ;;     ;; that variable only
;;                  ;;     (progn
;;                  ;;       (setf true-obs (nth loc-pos (true-obs constituent-gp)))
;;                  ;;       (flet ((quadrature-basis (parent-vals)
;;                  ;;                (declare (ignore parent-vals))
;;                  ;;                (list (list true-obs 1d0))))
;;                  ;;         (add-quad-variable quad-set
;;                  ;;                            0
;;                  ;;                            #'quadrature-basis)))

;;                  ;;     (progn
;;                  ;;       (setf mean-lists
;;                  ;;             (loop for mu in mean-arrays
;;                  ;;                   collect (aref mu loc-ind))
;;                  ;;             var-lists
;;                  ;;             (loop for cov in cov-arrays
;;                  ;;                   collect (make-array (1+ loc-ind)
;;                  ;;                                       :element-type 'double-float
;;                  ;;                                       :displaced-to cov
;;                  ;;                                       :displaced-index-offset
;;                  ;;                                       (* loc-ind n-pred-locs))))
;;                  ;;       ;; (format t "mean lists = ~a~%" mean-lists)
;;                  ;;       ;; (format t "var-lists = ~a~%" var-lists)
;;                  ;;       (flet ((augmented-quadrature-basis (parent-vals)
;;                  ;;                (let (f-list
;;                  ;;                      den)
;;                  ;;                (loop for point in (flattened (quadrature-set output))
;;                  ;;                      for weight in (weights (quadrature-set output))
;;                  ;;                      ;; mean-i is a list of means?
;;                  ;;                      do ;(format t "~a~%" mean-lists)
;;                  ;;                         ;(format t "~a~%" var-lists)
;;                  ;;                         (loop for point-i in point
;;                  ;;                               for mean-i in mean-lists
;;                  ;;                               for cov-i in var-lists
;;                  ;;                               for base-i = (loop for )
;;                  ;;                               for f = (safe-exp mean-i
;;                  ;;                                                 ))
;;                  ;;                        (return)))))
;;                  ;;         (add-quad-variable quad-set
;;                  ;;                            0
;;                  ;;                            #'augmented-quadrature-basis))))
;;               )
        
;;         ;predictive-posterior

;;         latent-quad-set))))
