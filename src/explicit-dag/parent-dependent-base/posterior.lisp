(uiop:define-package #:dag-gp/explicit-dag/parent-dependent-base/posterior
    (:use #:cl
          #:mgl-mat
          #:dag-gp/lapack
          #:dag-gp/output
          #:dag-gp/explicit-dag/parent-dependent-base/base)
  (:export #:make-posteriors
           #:dLL/dpost))

(in-package #:dag-gp/explicit-dag/parent-dependent-base/posterior)


(defgeneric make-posteriors (gp)
  (:documentation "Outputs the posterior means and diagonal covariances.")
  (:method ((gp variational-parent-dependent-gp))
    (let (f-mean)

      ;; Make the 'middle' matrix Su - Kuu
      (setf (middle gp) (m- (q-cov gp) (Kuu gp))
            f-mean (make-mat (n-obs gp) :ctype :double :initial-element 0d0))
      ;; This will be zeroed out in symm!, so ok to be garbage
      (setf (KfuM gp) (make-mat (list (n-obs gp) (n-latent gp))
                                :ctype :double))
      ;; Kfu <- Kfu Kuu^-1
      (potrs! (Kuu-chol gp) (Kfu gp)
              :uplo #\L :n (n-latent gp) :lda (n-latent gp)
              :nrhs (n-obs gp) :ldb (n-latent gp) :transpose-b? t)
      ;; Matrix vector multiplication
      (gemv! 1d0 (Kfu gp) (q-mean gp) 1d0 f-mean
             :m (n-obs gp) :n (n-latent gp) :lda (n-latent gp))

      ;; Compute KfuM
      (symm! 1d0 (middle gp) (Kfu gp) 0d0 (KfuM gp)
             :side #\R :uplo #\L :m (n-obs gp) :n (n-latent gp)
             :lda (n-latent gp) :ldb (n-latent gp) :ldc (n-latent gp))
      
      ;; We only need the diagonal elements, so compute with dot, and
      ;; store in Kff
      (loop for ob-ind below (n-obs gp)
            for displace = (* ob-ind (n-latent gp)) do
              ;; Extract the appropriate rows of KfuM and Kfu
              (reshape-and-displace! (KfuM gp) (n-latent gp) displace)
              (reshape-and-displace! (Kfu gp) (n-latent gp) displace)
              ;; Row of KfuM * row of Kfu
              (incf (aref (Kff gp) ob-ind)
                    (dot (KfuM gp) (Kfu gp) :n (n-latent gp))))
      ;; Reset Kfu dimension
      (reshape-and-displace! (KfuM gp) (list (n-obs gp) (n-latent gp)) 0)
      (reshape-and-displace! (Kfu gp) (list (n-obs gp) (n-latent gp)) 0)      
      (setf (qf-mean gp) f-mean)))

  (:method ((gp variational-combined-output-parent-dependent-gp))
    (let (f-mean
          middle
          KfuM
          qf-means)
      (setf (middle gp) nil
            (KfuM gp) nil
            (qf-mean gp) nil)
      
      (loop for q-cov in (q-cov gp)
            for Kuu in (Kuu gp)
            for Kuu-chol in (Kuu-chol gp)
            for Kfu in (Kfu gp)
            for q-mean in (q-mean gp)
            for Kff in (Kff gp)

            ;; Make the 'middle' matrix Su - Kuu
            do (setf middle (m- q-cov Kuu)
                     (middle gp) (nconc (middle gp) (list middle))
                     f-mean (make-mat (n-obs gp) :ctype :double :initial-element 0d0))
               ;; This will be zeroed out in symm!, so ok to be garbage
               (setf KfuM (make-mat (list (n-obs gp) (n-latent gp))
                                    :ctype :double)
                     (KfuM gp) (nconc (KfuM gp) (list KfuM)))
                ;; Kfu <- Kfu Kuu^-1
               (potrs! Kuu-chol Kfu
                       :uplo #\L :n (n-latent gp) :lda (n-latent gp)
                       :nrhs (n-obs gp) :ldb (n-latent gp) :transpose-b? t)
               ;; Matrix vector multiplication
               (gemv! 1d0 Kfu q-mean 1d0 f-mean
                      :m (n-obs gp) :n (n-latent gp) :lda (n-latent gp))
               
               ;; Compute KfuM
               (symm! 1d0 middle Kfu 0d0 KfuM
                      :side #\R :uplo #\L :m (n-obs gp) :n (n-latent gp)
                      :lda (n-latent gp) :ldb (n-latent gp) :ldc (n-latent gp))

               ;; We only need the diagonal elements, so compute with dot, and
               ;; store in Kff
               (loop for ob-ind below (n-obs gp)
                     for displace = (* ob-ind (n-latent gp)) do
                       ;; Extract the appropriate rows of KfuM and Kfu
                       (reshape-and-displace! KfuM (n-latent gp) displace)
                       (reshape-and-displace! Kfu (n-latent gp) displace)
                       ;; Row of KfuM * row of Kfu
                       (incf (aref Kff ob-ind)
                             (dot KfuM Kfu :n (n-latent gp))))
               ;; Reset Kfu dimension
               (reshape-and-displace! KfuM (list (n-obs gp) (n-latent gp)) 0)
               (reshape-and-displace! Kfu (list (n-obs gp) (n-latent gp)) 0)      
               (setf (qf-mean gp) (nconc (qf-mean gp) (list f-mean))))

      ;; Transform from ((ind0loc0 ind0loc1 ...) (ind1loc0 ind1loc1 ...)) to
      ;; ((loc0ind0 loc0ind1) (loc1ind0 loc1ind1) ...)
      (setf (reshaped-qf-mean gp) nil
            (reshaped-Kff gp) nil)
      (loop for qf-mean in (qf-mean gp) do
        (setf qf-means (nconc qf-means (list (mat-to-array qf-mean)))))
      (loop for loc-ind below (n-obs gp)
            ;; for reshaped-qf-mean-el = (make-array (n-combined gp)
            ;;                                       :element-type 'double-float)
            ;; for reshaped-Kff-el = (make-array (n-combined gp)
            ;;                                   :element-type 'double-float)
            ;; do (loop for qf-mean in qf-means
            ;;          )
            collect (loop for qf-mean in qf-means
                          collect (aref qf-mean loc-ind))
              into reshaped-qf-mean
            collect (loop for Kff in (Kff gp)
                          collect (aref Kff loc-ind))
              into reshaped-Kff
            finally (setf (reshaped-qf-mean gp) reshaped-qf-mean
                          (reshaped-Kff gp) reshaped-Kff)))))


;; TODO: Currently operates on lists instead of arrays
(defgeneric dLL/dpost (gp)
  (:documentation "Computes the derivative of the log likelihood with respect to the posterior.")

  (:method ((gp parent-dependent-gp))
    (let* ((n-obs (if (closed-downwards-p gp)
                      (n-true-obs gp)
                      (n-obs gp)))
           (dLL/dKff (make-mat (list n-obs n-obs)
                              :ctype :double
                              :initial-element 0d0))
          (Kff-inv (copy-mat (Kff gp))))
      ;; d yT Kff^-1 y/dKff = - (Kff^-1 y)^T (Kff^-1 y)
      (ger! 1d0 (obs-mat-copy gp) (obs-mat-copy gp) dLL/dKff
            :m n-obs :n n-obs :lda n-obs)
      
      ;; dlogdet(Kff)/dKff = Kff^-1 
      ;; Kff-inv <- Kff^-1
      (symmetric-potri! Kff-inv :uplo #\L :n n-obs :lda n-obs)
      (setf (Kff-inv gp) Kff-inv)
      
      (setf dLL/dKff (m- dLL/dKff Kff-inv))
      ;; Direction input because we no longer care about the mat
      (with-facets ((dLL/dKff-array (dLL/dKff 'array :direction :input)))
        (setf (dLL/dqf-var gp)
              (aops:vectorize* 'double-float (dLL/dKff-array)
                (* 0.5 dLL/dKff-array))))))
  
  (:method ((gp variational-parent-dependent-gp))
    (with-facets ((qf-mean-array ((qf-mean gp) 'backing-array :direction :input)))
      (let* ((qf-mean-list (coerce qf-mean-array 'list))
             (Kff-list (coerce (Kff gp) 'list)))
        (destructuring-bind (dLL/dqf-mu dLL/dqf-var)
            (dLL/dmu+dvar (output gp) qf-mean-list Kff-list
                          (obs gp) 0 nil)
          (setf (dLL/dqf-mu gp) dLL/dqf-mu
                (dLL/dqf-var gp) dLL/dqf-var))))))

