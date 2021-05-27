(uiop:define-package #:dag-gp/explicit-dag/parent-dependent-base/covariance
    (:use #:cl
          #:mgl-mat
          #:dag-gp/kernel
          #:dag-gp/output
          #:dag-gp/lapack
          #:dag-gp/explicit-dag/parent-dependent-base/base)
  (:import-from #:array-operations)
  (:export #:compute-covariances
           #:dLL/dKfu
           #:dLL/dKuu
           #:dNLL/dkernparam
           #:dNLL/dnoise
           #:compute-predictive-covariances))

(in-package #:dag-gp/explicit-dag/parent-dependent-base/covariance)


(defgeneric compute-covariances (gp &key use-all-locs)
  (:documentation "Computes the necessary covariance matrices.")
  
  (:method ((gp parent-dependent-gp) &key (use-all-locs t))
    ;; There is only one kernel and no scaling, so this is easy.
    (setf (Kff gp) (evaluate-matrix (kernel gp)
                                    (ff-squared-dist gp)
                                    (ff-abs-dist gp)
                                    nil
                                    nil))

    ;; Add noise, because we will never use Kff without
    (let ((noise-var (safe-exp (aref (output-params (output gp)) 0)))
          ;; If we are recomputing for prediction, use all observations
          (n-obs (if (and (closed-downwards-p gp)
                          (not use-all-locs))
                     (n-true-obs gp)
                     (n-obs gp))))
      (loop for i below n-obs do
        (incf (aref (Kff gp) i i) noise-var)))

    ;; Save as a mat
    (setf (Kff gp) (array-to-mat (Kff gp) :ctype :double)))

  (:method ((gp variational-parent-dependent-gp) &key (use-all-locs t))
    (declare (ignore use-all-locs))
    ;; Make Kff mats without jitter
    (setf (Kff gp) (evaluate-matrix (kernel gp)
                                    (ff-squared-dist gp)
                                    (ff-abs-dist gp)
                                    nil
                                    nil
                                    :add-jitter nil))

    ;; Make Kfu mats
    (setf (Kfu gp) (evaluate-matrix (kernel gp)
                                    (fu-squared-dist gp)
                                    (fu-abs-dist gp)
                                    nil
                                    nil
                                    :add-jitter nil)
          (Kfu gp) (array-to-mat (Kfu gp) :ctype :double))

    ;; Make Kuu mats
    (setf (Kuu gp) (evaluate-matrix (kernel gp)
                                    (uu-squared-dist gp)
                                    (uu-abs-dist gp)
                                    nil
                                    nil
                                    :add-jitter t)
          (Kuu gp) (array-to-mat (Kuu gp) :ctype :double))
    (let ((chol (copy-mat (Kuu gp))))
      (potrf! chol :uplo #\L :n (n-latent gp) :lda (n-latent gp))
      (setf (Kuu-chol gp) chol)))

  (:method ((gp variational-combined-output-parent-dependent-gp)
            &key (use-all-locs t))
    (declare (ignore use-all-locs))
    (let (Kfu
          Kuu
          chol)
      (setf (Kff gp) nil
            (Kfu gp) nil
            (Kuu gp) nil
            (Kuu-chol gp) nil)
      
      (loop for kern in (kernel gp) do
        ;; Make Kff mats without jitter
        (setf (Kff gp)
              (nconc (Kff gp) (list (evaluate-matrix kern
                                                     (ff-squared-dist gp)
                                                     (ff-abs-dist gp)
                                                     nil
                                                     nil
                                                     :add-jitter nil))))

        ;; Make Kfu mats
        (setf Kfu (evaluate-matrix kern
                                   (fu-squared-dist gp)
                                   (fu-abs-dist gp)
                                   nil
                                   nil
                                   :add-jitter nil)
              (Kfu gp)
              (nconc (Kfu gp) (list (array-to-mat Kfu :ctype :double))))
        
        ;; Make Kuu mats
        (setf Kuu (evaluate-matrix kern
                                   (uu-squared-dist gp)
                                   (uu-abs-dist gp)
                                   nil
                                   nil
                                   :add-jitter t)
              Kuu (array-to-mat Kuu :ctype :double)
              (Kuu gp) (nconc (Kuu gp) (list Kuu)))

        (setf chol (copy-mat Kuu))
        (potrf! chol :uplo #\L :n (n-latent gp) :lda (n-latent gp))
        (setf (Kuu-chol gp) (nconc (Kuu-chol gp) (list chol)))))))


(defun safe-exp (f)
  (cond
    ((> f 11.5d0)
     (exp 11.5d0))
    ((< f -11.5d0)
     (exp -11.5d0))
    (t
     (exp f))))


(defgeneric dLL/dKfu (gp)
  (:documentation "Computes the derivative of the log likelihood with respect to elements of Kfu.")
  (:method ((gp variational-parent-dependent-gp))
    (let ((n-latent (n-latent gp))
          (n-obs (n-obs gp))
          dLL/dKfu)

        ;; Make q-mean-copy
      (setf (q-mean-copy gp) (copy-mat (q-mean gp)))
      ;; q-mean <- Kuu^-1 q-mean
      (potrs! (Kuu-chol gp) (q-mean gp) :uplo #\L :n n-latent :lda n-latent
                                        :nrhs 1 :ldb n-latent :transpose-b? t)
        
      ;; KfuM <- Kfu Kuu^-1 M Kuu^-1
      (potrs! (Kuu-chol gp) (KfuM gp) :uplo #\L :n n-latent :lda n-latent
                                      :nrhs n-obs :ldb n-latent :transpose-b? t)
      ;; Set up array to hold derivatives - will be entirely set
      (setf dLL/dKfu (make-array (list n-obs n-latent)
                                 :element-type 'double-float))
      (with-facets ((KfuM-array ((KfuM gp) 'array :direction :input))
                    (q-mean-array ((q-mean gp) 'backing-array :direction :input)))
        ;; Derivative is 2 dv Kfu Kuu^-1 M Kuu^-1
        (loop for r below n-obs
              for dLL-var-obs in (dLL/dqf-var gp)
              for dLL-mu-obs in (dLL/dqf-mu gp) do 
                (dotimes (c n-latent)
                  (setf (aref dLL/dKfu r c)
                        (+ (* 2 dLL-var-obs (aref KfuM-array r c))
                           (* dLL-mu-obs (aref q-mean-array c)))))))
      dLL/dKfu))

  (:method ((gp variational-combined-output-parent-dependent-gp))
    (let ((n-latent (n-latent gp))
          (n-obs (n-obs gp)))
      (loop for q-mean in (q-mean gp)
            for Kuu-chol in (Kuu-chol gp)
            for KfuM in (KfuM gp)
            for dLL/dqf-mu in (dLL/dqf-mu gp)
            for dLL/dqf-var in (dLL/dqf-var gp)
            for dLL/dKfu-out = (dLL/dKfu-single-gp n-latent n-obs q-mean Kuu-chol
                                                   KfuM dLL/dqf-mu dLL/dqf-var)
            collect (first dLL/dKfu-out) into q-mean-copy
            collect (second dLL/dKfu-out) into dLL/dKfu
            finally (setf (q-mean-copy gp) q-mean-copy)
                    (return dLL/dKfu)))))


(defun dLL/dKfu-single-gp (n-latent n-obs q-mean Kuu-chol KfuM
                           dLL/dqf-mu dLL/dqf-var)
  (let (q-mean-copy
        dLL/dKfu)
    ;; Make q-mean-copy
    (setf q-mean-copy (copy-mat q-mean))
    ;; q-mean <- Kuu^-1 q-mean
    (potrs! Kuu-chol q-mean :uplo #\L :n n-latent :lda n-latent
                            :nrhs 1 :ldb n-latent :transpose-b? t)
    
    ;; KfuM <- Kfu Kuu^-1 M Kuu^-1
    (potrs! Kuu-chol KfuM :uplo #\L :n n-latent :lda n-latent
                          :nrhs n-obs :ldb n-latent :transpose-b? t)
    ;; Set up array to hold derivatives - will be entirely set
    (setf dLL/dKfu (make-array (list n-obs n-latent)
                               :element-type 'double-float))
    (with-facets ((KfuM-array (KfuM 'array :direction :input))
                  (q-mean-array (q-mean 'backing-array :direction :input)))
      ;; Derivative is 2 dv Kfu Kuu^-1 M Kuu^-1
      (loop for r below n-obs
            for dLL-var-obs in dLL/dqf-var
            for dLL-mu-obs in dLL/dqf-mu do 
              (dotimes (c n-latent)
                (setf (aref dLL/dKfu r c)
                      (+ (* 2 dLL-var-obs (aref KfuM-array r c))
                         (* dLL-mu-obs (aref q-mean-array c)))))))
    (list q-mean-copy dLL/dKfu)))


(defgeneric dLL/dKuu (gp dvarexp/dqcov)
  (:documentation "Computes the derivative of the variational part of the log likelihood with respect to the elements of Kuu.")
  (:method ((gp variational-parent-dependent-gp) dvarexp/dqcov)
    (let ((n-latent (n-latent gp))
          (n-obs (n-obs gp))
          (q-cholT (transpose (q-chol-mat gp)))
          dLL/dKuu
          mu-term
          var-term
          product-vec)
      ;; Make zero mats for the results
      (setf mu-term (make-mat (list n-latent n-latent)
                              :ctype :double
                              :initial-element 0d0)
            var-term (make-mat (list n-latent n-latent)
                                :ctype :double
                                :initial-element 0d0)
            product-vec (make-mat n-latent
                                  :ctype :double
                                  :initial-element 0d0))

      (let ((dLL/dqf-mu-mat (make-mat n-obs
                                      :ctype :double
                                      :initial-contents (dLL/dqf-mu gp)))
            (dLL/dqf-var-mat (make-mat n-obs
                                       :ctype :double
                                       :initial-contents (dLL/dqf-var gp))))
                
        ;; product-vec <- Kuu^-1 Kuf dmu
        ;; Build product vectors to use later      
        (gemv! 1d0 (Kfu gp) dLL/dqf-mu-mat 1d0 product-vec
               :transpose-a? t :m n-obs :n n-latent :lda n-latent)
             
        ;; KfuM <- dv Kfu Kuu^-1 M Kuu^-1
        (scale-rows! dLL/dqf-var-mat (KfuM gp))
        (gemm! -1d0 (Kfu gp) (KfuM gp) 1d0 var-term
               :transpose-a? t :m n-latent :n n-latent :k n-obs
               :lda n-latent :ldb n-latent :ldc n-latent)

        ;; - 0.5( -(Kuu^-1 qcov Kuu^-1)^T - Kuu^-1 q-mean q-mean^T Kuu^-1 + Kuu^-1)
        ;; Mean term
        (ger! -1d0 product-vec (q-mean gp) mu-term
              :m n-latent :n n-latent :lda n-latent)
        
        ;; q-cholT <- q-chol-mat Kuu^-1
        (potrs! (Kuu-chol gp) q-cholT
                :uplo #\L :n n-latent :lda n-latent
                :nrhs n-latent :ldb n-latent :transpose-b? t)
        (gemm! 0.5d0 q-cholT q-cholT 1d0 mu-term
               :transpose-a? t :m n-latent :n n-latent :k n-latent
               :lda n-latent :ldb n-latent :ldc n-latent)
        (ger! 0.5d0 (q-mean gp) (q-mean gp) mu-term
              :m n-latent :n n-latent :lda n-latent)

        ;; Combine results as needed
        ;; mu-term + var-term + var-term^T - dvarexp/dqcov - 0.5 Kuu-inv
        (setf dLL/dKuu (make-array (list n-latent n-latent)
                                   :element-type 'double-float
                                   :initial-element 0d0))
        (with-facets ((mu-term-array (mu-term 'array :direction :input))
                      (var-term-array (var-term 'array :direction :input))
                      (Kuu-inv-array ((Kuu-inv gp) 'array :direction :input)))
          (dotimes (r n-latent)
            (dotimes (c n-latent)
              (setf (aref dLL/dKuu r c)
                    (+ (aref mu-term-array r c)
                       (aref var-term-array r c)
                       (aref var-term-array c r)
                       (- (aref dvarexp/dqcov r c))
                       (* -0.5 (aref Kuu-inv-array r c)))))))
        
        (values dLL/dKuu product-vec))))

  (:method ((gp variational-combined-output-parent-dependent-gp) dvarexp/dqcov)
    (let ((n-latent (n-latent gp))
          (n-obs (n-obs gp)))
      (loop for q-chol-mat in (q-chol-mat gp)
            for dLL/dqf-mu in (dLL/dqf-mu gp)
            for dLL/dqf-var in (dLL/dqf-var gp)
            for Kfu in (Kfu gp)
            for KfuM in (KfuM gp)
            for q-mean in (q-mean gp)
            for Kuu-chol in (Kuu-chol gp)
            for Kuu-inv in (Kuu-inv gp)
            for dvarexp/dqcov-i in dvarexp/dqcov
            for dLL/dKuu-out = (dLL/dKuu-single-gp n-latent n-obs q-chol-mat
                                                   dLL/dqf-mu dLL/dqf-var Kfu KfuM
                                                   q-mean Kuu-chol Kuu-inv
                                                   dvarexp/dqcov-i)
            collect (first dLL/dKuu-out) into dLL/dKuu
            collect (second dLL/dKuu-out) into product-vec
            finally (return (values dLL/dKuu product-vec))))))


(defun dLL/dKuu-single-gp (n-latent n-obs q-chol-mat dLL/dqf-mu dLL/dqf-var Kfu KfuM
                           q-mean Kuu-chol Kuu-inv dvarexp/dqcov)
  (let ((q-cholT (transpose q-chol-mat))
        dLL/dKuu
        mu-term
        var-term
        product-vec)
    ;; Make zero mats for the results
    (setf mu-term (make-mat (list n-latent n-latent)
                            :ctype :double
                            :initial-element 0d0)
          var-term (make-mat (list n-latent n-latent)
                             :ctype :double
                             :initial-element 0d0)
          product-vec (make-mat n-latent
                                :ctype :double
                                :initial-element 0d0))

    (let ((dLL/dqf-mu-mat (make-mat n-obs
                                    :ctype :double
                                    :initial-contents dLL/dqf-mu))
          (dLL/dqf-var-mat (make-mat n-obs
                                     :ctype :double
                                     :initial-contents dLL/dqf-var)))
                
      ;; product-vec <- Kuu^-1 Kuf dmu
      ;; Build product vectors to use later      
      (gemv! 1d0 Kfu dLL/dqf-mu-mat 1d0 product-vec
             :transpose-a? t :m n-obs :n n-latent :lda n-latent)
             
        ;; KfuM <- dv Kfu Kuu^-1 M Kuu^-1
      (scale-rows! dLL/dqf-var-mat KfuM)
      (gemm! -1d0 Kfu KfuM 1d0 var-term
             :transpose-a? t :m n-latent :n n-latent :k n-obs
             :lda n-latent :ldb n-latent :ldc n-latent)

      ;; - 0.5( -(Kuu^-1 qcov Kuu^-1)^T - Kuu^-1 q-mean q-mean^T Kuu^-1 + Kuu^-1)
      ;; Mean term
      (ger! -1d0 product-vec q-mean mu-term
            :m n-latent :n n-latent :lda n-latent)
        
      ;; q-cholT <- q-chol-mat Kuu^-1
      (potrs! Kuu-chol q-cholT
              :uplo #\L :n n-latent :lda n-latent
              :nrhs n-latent :ldb n-latent :transpose-b? t)
      (gemm! 0.5d0 q-cholT q-cholT 1d0 mu-term
             :transpose-a? t :m n-latent :n n-latent :k n-latent
             :lda n-latent :ldb n-latent :ldc n-latent)
      (ger! 0.5d0 q-mean q-mean mu-term
            :m n-latent :n n-latent :lda n-latent)
      
      ;; Combine results as needed
      ;; mu-term + var-term + var-term^T - dvarexp/dqcov - 0.5 Kuu-inv
      (setf dLL/dKuu (make-array (list n-latent n-latent)
                                 :element-type 'double-float
                                 :initial-element 0d0))
      (with-facets ((mu-term-array (mu-term 'array :direction :input))
                    (var-term-array (var-term 'array :direction :input))
                    (Kuu-inv-array (Kuu-inv 'array :direction :input)))
        (dotimes (r n-latent)
          (dotimes (c n-latent)
            (setf (aref dLL/dKuu r c)
                  (+ (aref mu-term-array r c)
                     (aref var-term-array r c)
                     (aref var-term-array c r)
                     (- (aref dvarexp/dqcov r c))
                     (* -0.5 (aref Kuu-inv-array r c)))))))
        
      (list dLL/dKuu product-vec))))


(defgeneric dNLL/dkernparam (gp kernel du param dLL/dKfu dLL/dKuu dLL/dqf-var)
  (:documentation "Computes the derivative of log likelihood with respect to a kernel parameter.")

  (:method ((gp parent-dependent-gp) (kernel kernel) du
            param dLL/dKfu dLL/dKuu dLL/dqf-var)
    (declare (ignore du dLL/dKfu dLL/dKuu))
    (let ((dKff (dk/dparam-matrix kernel param (ff-squared-dist gp)
                                  (ff-abs-dist gp) nil nil)))
      (- (aops:sum-index (i j)
           (* (aref dLL/dqf-var i j)
              (aref dKff i j))))))

  (:method ((gp variational-parent-dependent-gp) (kernel kernel) du
            param dLL/dKfu dLL/dKuu dLL/dqf-var)
    (declare (ignore du))
    (let* ((dLL/dkernparam 0d0)
           (dKuu (dk/dparam-matrix kernel param (uu-squared-dist gp)
                                   (uu-abs-dist gp) nil nil))
           (dKff (dk/dparam-matrix kernel param (ff-squared-dist gp)
                                   (ff-abs-dist gp) nil nil))
           (dKfu (dk/dparam-matrix kernel param (fu-squared-dist gp)
                                   (fu-abs-dist gp) nil nil)))
      
        ;; Contribution to Kuu
      (incf dLL/dkernparam (aops:sum-index (i j)
                             (* (aref dLL/dKuu i j)
                                (aref dKuu i j))))
        
        ;; Contribution to Kff
      (loop for dLL/dqf-var-el in dLL/dqf-var
            for dKff-el across dKff
            do (incf dLL/dkernparam (* dLL/dqf-var-el
                                       dKff-el)))
          
      ;; Contribution to Kfu        
      (incf dLL/dkernparam (aops:sum-index (i j)
                             (* (aref dLL/dKfu i j)
                                (aref dKfu i j))))
        
      (- dLL/dkernparam))))


(defgeneric dNLL/dnoise (gp)
  (:documentation "Computes the derivative of negative log likelihood with respect to noise.")
  (:method ((gp parent-dependent-gp))
    (let ((dLL/dqf-var (dLL/dqf-var gp))
          (noise-deriv (safe-exp (aref (output-params (output gp)) 0))))
      (- (aops:sum-index (i)
           (* (aref dLL/dqf-var i i)
              noise-deriv))))))


(defgeneric compute-predictive-covariances (gp pred-locs &key use-all-locs)
  (:documentation "Computes the necessary covariance matrices.")
  
  (:method ((gp parent-dependent-gp) pred-locs &key (use-all-locs t))
    (declare (ignore pred-locs use-all-locs))
    (setf (pred-Kff gp) (evaluate-matrix (kernel gp)
                                         (pred-ff-squared-dist gp)
                                         (pred-ff-abs-dist gp)
                                         nil
                                         nil
                                         :add-jitter nil)
          (pred-Kff gp) (array-to-mat (pred-Kff gp) :ctype :double))

    (setf (pred-obs-Kff gp) (evaluate-matrix (kernel gp)
                                             (pred-obs-ff-squared-dist gp)
                                             (pred-obs-ff-abs-dist gp)
                                             nil
                                             nil
                                             :add-jitter nil)
          (pred-obs-Kff gp) (array-to-mat (pred-obs-Kff gp) :ctype :double)))

  (:method ((gp variational-parent-dependent-gp) pred-locs &key (use-all-locs t))
    (declare (ignore pred-locs use-all-locs))
    ;; Make Kff mats without jitter
    (setf (pred-Kff gp) (evaluate-matrix (kernel gp)
                                         (pred-ff-squared-dist gp)
                                         (pred-ff-abs-dist gp)
                                         nil
                                         nil
                                         :add-jitter nil)
          (pred-Kff gp) (array-to-mat (pred-Kff gp) :ctype :double))

    ;; Make Kfu mats
    (setf (pred-Kfu gp) (evaluate-matrix (kernel gp)
                                         (pred-fu-squared-dist gp)
                                         (pred-fu-abs-dist gp)
                                         nil
                                         nil
                                         :add-jitter nil)
          (pred-Kfu gp) (array-to-mat (pred-Kfu gp) :ctype :double)))

  (:method ((gp variational-combined-output-parent-dependent-gp) pred-locs
            &key (use-all-locs t))
    (declare (ignore pred-locs use-all-locs))
    (let (pred-Kff
          pred-Kfu)
    (setf (pred-Kff gp) nil
          (pred-Kfu gp) nil)

    (loop for kern in (kernel gp) do
      ;; Make Kff mats without jitter
      (setf pred-Kff (evaluate-matrix kern
                                      (pred-ff-squared-dist gp)
                                      (pred-ff-abs-dist gp)
                                      nil
                                      nil
                                      :add-jitter nil)
            (pred-Kff gp) (nconc (pred-Kff gp)
                                 (list (array-to-mat pred-Kff :ctype :double))))

      ;; Make Kfu mats
      (setf pred-Kfu (evaluate-matrix kern
                                      (pred-fu-squared-dist gp)
                                      (pred-fu-abs-dist gp)
                                      nil
                                      nil
                                      :add-jitter nil)
            (pred-Kfu gp) (nconc (pred-Kfu gp)
                                 (list (array-to-mat pred-Kfu :ctype :double))))))))
