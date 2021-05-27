(uiop:define-package #:dag-gp/explicit-dag/parent-dependent-base/latent
    (:use #:cl
          #:mgl-mat
          #:dag-gp/kernel
          #:dag-gp/output
          #:dag-gp/lapack
          #:dag-gp/explicit-dag/parent-dependent-base/base)
  (:export #:initialize-latent-locations
           #:initialize-q-means
           #:make-q-means
           #:initialize-q-cholesky
           #:compute-q-covariances
           #:dNLL/dqmean
           #:dvarexp/dqcov
           #:dNLL/dqchol))

(in-package #:dag-gp/explicit-dag/parent-dependent-base/latent)


;; Should be called only once, so need not be particularly efficient.
(defgeneric initialize-latent-locations (gp &key custom-obs-locs)
  (:documentation "If they have not already been set, generate an array of latent locations.")
  (:method ((gp variational-parent-dependent-gp) &key (custom-obs-locs nil))
    (let ((obs-locs (if custom-obs-locs
                        custom-obs-locs
                        (obs-locs gp)))
          (n-obs (if custom-obs-locs
                     (list-length custom-obs-locs)
                     (n-obs gp))))
      ;; Set up the u-locs vectors
      (setf (u-locs gp) (make-array (list (n-latent gp)
                                          (input-dim gp))
                                    :element-type 'double-float
                                    :initial-element 0d0))
    
      ;; Average based on the measurements.
      (let ((obs-per-latent (floor (/ n-obs (n-latent gp))))
            (count 0))
        (loop for u below (n-latent gp) do
          (loop for i below obs-per-latent do
            (loop for j below (input-dim gp) do
              (incf (aref (u-locs gp) u j)
                    (nth j (nth count obs-locs))))
            (incf count))
          (loop for j below (input-dim gp) do
            (setf (aref (u-locs gp) u j)
                  (/ (aref (u-locs gp) u j) obs-per-latent))))))))


(defgeneric initialize-q-means (gp)
  (:documentation "Constructs the initial q-means vector.")
  (:method ((gp variational-parent-dependent-gp))
    (setf (q-mean-params gp)
          (make-array (n-latent gp)
                      :element-type 'double-float
                      :displaced-to (param-vec gp)
                      :displaced-index-offset (+ (n-kern-params (kernel gp))
                                                 (n-output-params (output gp)))))
    (dotimes (i (n-latent gp))
      (setf (aref (q-mean-params gp) i) 0d0)))

  (:method ((gp variational-combined-output-parent-dependent-gp))
    (let ((offset (+ (loop for kern in (kernel gp)
                           sum (n-kern-params kern))
                     (n-output-params (output gp)))))
      (setf (q-mean-params gp) nil)
      (dotimes (i (n-combined gp))
        (setf (q-mean-params gp)
              (nconc (q-mean-params gp)
                     (list (make-array (n-latent gp)
                                       :element-type 'double-float
                                       :displaced-to (param-vec gp)
                                       :displaced-index-offset offset))))
        (incf offset (n-latent gp))))))


(defgeneric make-q-means (gp)
  (:documentation "Constructs mats for q-means.")
  (:method ((gp variational-parent-dependent-gp))
    (setf (q-mean gp) (array-to-mat (q-mean-params gp) :ctype :double)))
  (:method ((gp variational-combined-output-parent-dependent-gp))
    (setf (q-mean gp) nil)
    (loop for q-mean-params in (q-mean-params gp) do
      (setf (q-mean gp)
            (nconc (q-mean gp)
                   (list (array-to-mat q-mean-params :ctype :double)))))))


(defgeneric initialize-q-cholesky (gp)
  (:documentation "Initializes posterior Cholesky factorizations to be equal to the Cholesky factorization of the prior.")
  (:method ((gp variational-parent-dependent-gp))
    (setf (q-chol gp)
          (make-array (/ (* (n-latent gp)
                            (1+ (n-latent gp)))
                         2)
                      :element-type 'double-float
                      :displaced-to (param-vec gp)
                      :displaced-index-offset (+ (n-kern-params (kernel gp))
                                                 (n-output-params (output gp))
                                                 (n-latent gp))))
    
    ;; Initialize to cholesky factorization of Kuu
    (let ((Kuu (array-to-mat (evaluate-matrix (kernel gp)
                                              (uu-squared-dist gp)
                                              (uu-abs-dist gp)
                                              nil
                                              nil
                                              :add-jitter t)
                             :ctype :double))
          (count 0))

      (potrf! Kuu :uplo #\L :n (n-latent gp) :lda (n-latent gp))
      (with-facets ((Kuu-array (Kuu 'array :direction :input)))
        (dotimes (r (n-latent gp))
          (dotimes (c (1+ r))
            (setf (aref (q-chol gp) count) (aref Kuu-array r c))
            (incf count)))))

    ;; Initialize to array of ones
    ;; (let ((count 0))
    ;;   (dotimes (r (n-latent gp))
    ;;     (setf (aref (q-chol gp) count) 1d0)
    ;;     (incf count (+ r 2))))

    ;; (let ((count 0))
    ;;   (dotimes (r (/ (* (1+ (n-latent gp)) (n-latent gp)) 2))
    ;;     (setf (aref (q-chol gp) count) 1d0)
    ;;     (incf count 1)))
    )

  (:method ((gp variational-combined-output-parent-dependent-gp))
    (let ((offset (+ (loop for kern in (kernel gp)
                           sum (n-kern-params kern))
                     (n-output-params (output gp))
                     (* (n-combined gp) (n-latent gp))))
          (n-chol (/ (* (n-latent gp)
                        (1+ (n-latent gp)))
                     2))
          Kuu
          count
          q-chol)
      (setf (q-chol gp) nil)
      (loop for kern in (kernel gp) do
        (setf q-chol (make-array n-chol
                                 :element-type 'double-float
                                 :displaced-to (param-vec gp)
                                 :displaced-index-offset offset)
              Kuu (array-to-mat (evaluate-matrix kern
                                                 (uu-squared-dist gp)
                                                 (uu-abs-dist gp)
                                                 nil
                                                 nil
                                                 :add-jitter t)
                                :ctype :double)
              count 0)
        (potrf! Kuu :uplo #\L :n (n-latent gp) :lda (n-latent gp))
        (with-facets ((Kuu-array (Kuu 'array :direction :input)))
          (dotimes (r (n-latent gp))
            (dotimes (c (1+ r))
              (setf (aref q-chol count) (aref Kuu-array r c))
              (incf count))))
        (setf (q-chol gp) (nconc (q-chol gp) (list q-chol)))
        (incf offset n-chol)))))


(defgeneric compute-q-covariances (gp)
  (:documentation "Computes the covariances of the posterior from their cholesky factorizations.")
  (:method ((gp variational-parent-dependent-gp))
    (let ((q-chol-mat (make-array (list (n-latent gp) (n-latent gp))
                                  :element-type 'double-float
                                  :initial-element 0d0))
          q-cov
          (count 0))
      (dotimes (r (n-latent gp))
        (dotimes (c (1+ r))
          (setf (aref q-chol-mat r c) (aref (q-chol gp) count))
          (incf count)))
      (setf q-chol-mat (array-to-mat q-chol-mat :ctype :double)
            (q-chol-mat gp) q-chol-mat
            q-cov (m* q-chol-mat q-chol-mat :transpose-b? t)
            (q-cov gp) q-cov)))

  (:method ((gp variational-combined-output-parent-dependent-gp))
    (let (q-chol-mat
          count)
      (setf (q-chol-mat gp) nil
            (q-cov gp) nil)
      (loop for q-chol in (q-chol gp) do
        (setf q-chol-mat (make-array (list (n-latent gp) (n-latent gp))
                                     :element-type 'double-float
                                     :initial-element 0d0)
              count 0)
        (dotimes (r (n-latent gp))
          (dotimes (c (1+ r))
            (setf (aref q-chol-mat r c) (aref q-chol count))
            (incf count)))

        (setf q-chol-mat (array-to-mat q-chol-mat :ctype :double)
              (q-chol-mat gp) (nconc (q-chol-mat gp)
                                     (list q-chol-mat))
              (q-cov gp) (nconc (q-cov gp) (list (m* q-chol-mat q-chol-mat :transpose-b? t))))))))


(defgeneric dNLL/dqmean (gp dvarexp/dqmean deriv-array)
  (:documentation "Computes the derivative of the negative log likelihood with respect to the posterior mean elements.")
  (:method ((gp variational-parent-dependent-gp) dvarexp/dqmean deriv-array)
    ;; Elements for varexp have already been computed
    (let ((mean-index (+ (n-kern-params (kernel gp))
                         (n-output-params (output gp)))))
      (with-facets ((q-mean-array ((q-mean gp) 'backing-array :direction :input))
                    (dvarexp/dqmean-array (dvarexp/dqmean 'backing-array
                                                          :direction :input)))
        (dotimes (i (n-latent gp))
          (setf (aref deriv-array (+ mean-index i))
                (- (aref q-mean-array i)
                   (aref dvarexp/dqmean-array i)))))))

  (:method ((gp variational-combined-output-parent-dependent-gp) dvarexp/dqmean
            deriv-array)
    (let ((mean-index (+ (loop for kernel in (kernel gp)
                               sum (n-kern-params kernel))
                         (n-output-params (output gp))))
          (n-latent (n-latent gp))
          (offset 0))
      (loop for q-mean in (q-mean gp)
            for dvarexp/dqmean-i in dvarexp/dqmean
            do (with-facets ((q-mean-array (q-mean 'backing-array
                                            :direction :input))
                             (dvarexp/dqmean-array (dvarexp/dqmean-i
                                                    'backing-array
                                                    :direction :input)))
                 (dotimes (i n-latent)
                   (setf (aref deriv-array (+ mean-index i offset))
                         (- (aref q-mean-array i)
                            (aref dvarexp/dqmean-array i))))
                 (incf offset n-latent))))))


(defgeneric dvarexp/dqcov (gp)
  (:documentation "Computes the derivative of the variational part of the log likelihood with respect to the posterior covariance elements.")
  (:method ((gp variational-parent-dependent-gp))
    ;; Make two n-latent x n-latent arrays
    (let (dLL/dqcov-array)
      (setf dLL/dqcov-array (make-array (list (n-latent gp) (n-latent gp))
                                        :element-type 'double-float
                                        :initial-element 0d0))
      ;; Fill in the array
      ;; dLL/dqcov_r,c = \sum_out \sum_obs dLL/dvarout,obs foutuind_obs,r foutuind_obs,c )
      (with-facets ((Kfu ((Kfu gp) 'array :direction :input)))
        ;; Loop over observations
        (loop for dLL-out in (dLL/dqf-var gp)
              for obs from 0 do            
                (dotimes (r (n-latent gp))
                  ;; The derivative matrix is symmetric
                  (dotimes (c r)
                    (incf (aref dLL/dqcov-array r c)
                          (* dLL-out
                             (aref Kfu obs r)
                             (aref Kfu obs c))))
                          (incf (aref dLL/dqcov-array r r)
                                (* dLL-out
                                   (expt (aref Kfu obs r) 2))))))
      ;; Make the symmetric elements
      (dotimes (r (n-latent gp))
        (dotimes (c r)
          (setf (aref dLL/dqcov-array c r)
                (aref dLL/dqcov-array r c))))
      
      dLL/dqcov-array))

  ;; TODO: Separate out a function to be called on dLL/dqf-var and Kfu
  (:method ((gp variational-combined-output-parent-dependent-gp))
    (loop for dLL/dqf-var in (dLL/dqf-var gp)
          for Kfu-i in (Kfu gp)
          collect (let (dLL/dqcov-array)
                    (setf dLL/dqcov-array (make-array (list (n-latent gp)
                                                            (n-latent gp))
                                        :element-type 'double-float
                                        :initial-element 0d0))

                    (with-facets ((Kfu (Kfu-i 'array :direction :input)))
                      ;; Loop over observations
                      (loop for dLL-out in dLL/dqf-var
                            for obs from 0 do            
                              (dotimes (r (n-latent gp))
                                ;; The derivative matrix is symmetric
                                (dotimes (c r)
                                  (incf (aref dLL/dqcov-array r c)
                                        (* dLL-out
                                           (aref Kfu obs r)
                                           (aref Kfu obs c))))
                                (incf (aref dLL/dqcov-array r r)
                                      (* dLL-out
                                         (expt (aref Kfu obs r) 2))))))
                    ;; Make the symmetric elements
                    (dotimes (r (n-latent gp))
                      (dotimes (c r)
                        (setf (aref dLL/dqcov-array c r)
                              (aref dLL/dqcov-array r c))))
                    
                    dLL/dqcov-array))))


(defgeneric dLL/dqcov (gp dvarexp/dqcov)
  (:documentation "Computes the derivative of the log likelihood with respect to the posterior covariance elements.")
  (:method ((gp variational-parent-dependent-gp) dvarexp/dqcov)
    ;; d D(N0||N1)/S0 = 0.5 (S1^-T - S0^-T)
    (let ((q-inv (copy-mat (q-chol-mat gp)))
          (n-latent (n-latent gp)))
      (potri! q-inv :n n-latent :uplo #\L :lda n-latent)
      (with-facets ((Kuu-inv-array ((Kuu-inv gp) 'array :direction :input))
                    (q-inv-array (q-inv 'array :direction :input)))
        (dotimes (r n-latent)
          (dotimes (c (1+ r))
            (incf (aref dvarexp/dqcov r c)
                  ;; These inverses are symmetric, and we only have the
                  ;; lower half anyway
                  (* 0.5 (- (aref q-inv-array r c)
                            (aref Kuu-inv-array r c))))))
        ;; Fill out the opposite parts
        (dotimes (c n-latent)
          (dotimes (r c)
            (setf (aref dvarexp/dqcov r c)
                  (aref dvarexp/dqcov c r))))))
    dvarexp/dqcov)

  (:method ((gp variational-combined-output-parent-dependent-gp) dvarexp/dqcov)
    (let ((n-latent (n-latent gp)))
      (loop for q-chol-mat in (q-chol-mat gp)
            for dvarexp/dqcov-i in dvarexp/dqcov
            for Kuu-inv in (Kuu-inv gp)
            do (dLL/dqcov-single-gp n-latent dvarexp/dqcov-i q-chol-mat Kuu-inv)))))


(defun dLL/dqcov-single-gp (n-latent dvarexp/dqcov q-chol-mat Kuu-inv)
  (let ((q-inv (copy-mat q-chol-mat)))
    (potri! q-inv :n n-latent :uplo #\L :lda n-latent)
    (with-facets ((Kuu-inv-array (Kuu-inv 'array :direction :input))
                  (q-inv-array (q-inv 'array :direction :input)))
      (dotimes (r n-latent)
        (dotimes (c (1+ r))
          (incf (aref dvarexp/dqcov r c)
                ;; These inverses are symmetric, and we only have the
                ;; lower half anyway
                (* 0.5 (- (aref q-inv-array r c)
                          (aref Kuu-inv-array r c))))))
      ;; Fill out the opposite parts
      (dotimes (c n-latent)
        (dotimes (r c)
          (setf (aref dvarexp/dqcov r c)
                (aref dvarexp/dqcov c r)))))))


(defgeneric dNLL/dqchol (gp dvarexp/dqcov deriv-array)
  (:documentation "Computes the negative derivative of the log likelihood with respect to the posterior cholesky decomposition elements.")
  (:method ((gp variational-parent-dependent-gp) dvarexp/dqcov deriv-array)
    ;; Alter the derivative to include the KL part
    ;; dvarexp/dqcov <- dLL/dqcov
    (dLL/dqcov gp dvarexp/dqcov)
    (let ((chol-index (+ (n-kern-params (kernel gp))
                         (n-output-params (output gp))
                         (n-latent gp))))
      (with-facets ((q-chol ((q-chol-mat gp) 'array :direction :input)))
        ;; Loop through cholesky elements
        (dotimes (r (n-latent gp))
          (loop for c below (1+ r)
                for deriv-ind = (+ (/ (* r (1+ r)) 2) c chol-index)
                ;; Multiply by column c of q-chol along rows and columns r
                do (setf (aref deriv-array deriv-ind) 0d0)
                   (dotimes (i (n-latent gp))
                     (incf (aref deriv-array deriv-ind)
                           (- (* (aref q-chol i c)
                                 (+ (aref dvarexp/dqcov i r)
                                    (aref dvarexp/dqcov r i)))))))))))

  (:method ((gp variational-combined-output-parent-dependent-gp) dvarexp/dqcov
            deriv-array)
    (dLL/dqcov gp dvarexp/dqcov)
    (let* ((n-latent (n-latent gp))
           (chol-index (+ (loop for kernel in (kernel gp)
                                sum (n-kern-params kernel))
                          (n-output-params (output gp))
                          (* (n-combined gp) n-latent)))
           (offset 0)
           (offset-inc (/ (* n-latent (1+ n-latent)) 2)))
      (loop for q-chol-mat in (q-chol-mat gp)
            for dvarexp/dqcov-i in dvarexp/dqcov
            do (with-facets ((q-chol (q-chol-mat 'array :direction :input)))
                 ;; Loop through cholesky elements
                 (dotimes (r n-latent)
                   (loop for c below (1+ r)
                         for deriv-ind = (+ (/ (* r (1+ r)) 2) c
                                            chol-index offset)
                         ;; Multiply by column c of q-chol along rows and columns r
                         do (setf (aref deriv-array deriv-ind) 0d0)
                            (dotimes (i n-latent)
                              (incf (aref deriv-array deriv-ind)
                                    (- (* (aref q-chol i c)
                                          (+ (aref dvarexp/dqcov-i i r)
                                             (aref dvarexp/dqcov-i r i))))))))
                 (incf offset offset-inc))))))
