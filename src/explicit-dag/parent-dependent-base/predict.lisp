(uiop:define-package #:dag-gp/explicit-dag/parent-dependent-base/predict
    (:use #:cl
          #:mgl-mat
          #:dag-gp/lapack
          #:dag-gp/explicit-dag/parent-dependent-base/base
          #:dag-gp/explicit-dag/parent-dependent-base/distance-matrix
          #:dag-gp/explicit-dag/parent-dependent-base/covariance
          #:dag-gp/explicit-dag/parent-dependent-base/latent
          #:dag-gp/explicit-dag/parent-dependent-base/posterior
          #:dag-gp/explicit-dag/parent-dependent-base/likelihood)
  (:export #:preprocess-prediction
           #:make-predictive-posteriors))

(in-package #:dag-gp/explicit-dag/parent-dependent-base/predict)


(defgeneric preprocess-prediction (gp pred-locs n-pred &key use-all-locs)
  (:documentation "Prepares the GP for prediction at a list of prediction locations.")
  (:method ((gp parent-dependent-gp) pred-locs n-pred &key (use-all-locs t))
    (make-predictive-distance-matrices gp pred-locs n-pred :use-all-locs use-all-locs)
    (compute-predictive-covariances gp pred-locs :use-all-locs use-all-locs)))


(defgeneric make-predictive-posteriors (gp pred-locs &key closed-downwards-p use-all-locs)
  (:documentation "Outputs the predictive means and covariances.")

  (:method ((gp parent-dependent-gp) pred-locs
            &key (closed-downwards-p nil) (use-all-locs nil))
    (let* ((n-obs (if closed-downwards-p
                      (n-true-obs gp)
                      (n-obs gp)))
           (n-pred (list-length pred-locs))
           (pred-mean (make-mat n-pred :ctype :double
                                       :initial-element 0d0))
           pred-obs-Kff-copy)
      (preprocess-prediction gp pred-locs n-pred :use-all-locs use-all-locs)
      
      (setf pred-obs-Kff-copy (copy-mat (pred-obs-Kff gp)))
      
      ;; pred-mean <- pred-obs-Kff Kff^-1 y
      (gemv! 1d0 (pred-obs-Kff gp) (obs-mat-copy gp) 0d0 pred-mean
             :m n-pred :n n-obs :lda n-obs)

      ;; pred-obs-Kff-copy <- pred-obs-Kff-copy Kff^-1
      (potrs! (Kff gp) pred-obs-Kff-copy
              :uplo #\L :n n-obs :nrhs n-pred
              :lda n-obs :ldb n-obs :transpose-b? t)
      ;; pred-Kff <- pred-Kff - pred-obs-Kff-copy Kff^-1 pred-obs-Kff^T
      (gemm! -1d0 pred-obs-Kff-copy (pred-obs-Kff gp) 1d0 (pred-Kff gp)
             :transpose-b? t :m n-pred :n n-pred :k n-obs
             :lda n-obs :ldb n-obs :ldc n-pred)

      (list pred-mean (pred-Kff gp))))

  (:method ((gp variational-parent-dependent-gp) pred-locs
            &key (closed-downwards-p nil) (use-all-locs nil))
    (declare (ignore closed-downwards-p use-all-locs))
    (let ((n-pred (list-length pred-locs))
          (n-latent (n-latent gp))
          KfuM
          pred-mean
          pred-cov)
      (preprocess-prediction gp pred-locs n-pred)

      ;; These may not exist
      ;; TODO: Track when they need to be recreated
      ;(unless (middle gp)
      (compute-covariances gp)
      (make-q-means gp)
      (compute-q-covariances gp)
      (make-posteriors gp)
      (setf (q-mean-copy gp) (copy-mat (q-mean gp)))
      ;)

      ;; Kfu <- Kfu Kuu^-1
      (potrs! (Kuu-chol gp) (pred-Kfu gp)
              :uplo #\L :n n-latent :lda n-latent
              :nrhs n-pred :ldb n-latent :transpose-b? t)

      ;; Construct the needed KfuM
      ;; This will be zeroed out in symm!, so ok to be garbage
      (setf KfuM (make-mat (list n-pred n-latent)
                           :ctype :double))
      ;; Compute KfuM
      (symm! 1d0 (middle gp) (pred-Kfu gp)
             0d0 KfuM
             :side #\R :uplo #\L :m n-pred
             :n n-latent :lda n-latent :ldb n-latent :ldc n-latent)
            
      ;; Predictive mean
      (setf pred-mean (make-mat n-pred :ctype :double))
      (gemv! 1d0 (pred-Kfu gp) (q-mean-copy gp) 0d0 pred-mean
             :m n-pred :n n-latent :lda n-latent)
               
      ;; Predictive covariance
      ;; Loop over u indices to build element (df-1,df-2)
      (setf pred-cov (make-mat (list n-pred n-pred)
                               :ctype :double))
      (gemm! 1d0 KfuM (pred-Kfu gp) 1d0 pred-cov
             :transpose-a? nil :transpose-b? t
             :m n-pred :n n-pred :k n-latent
             :lda n-latent :ldb n-latent :ldc n-pred)

      ;; Add pred-cov-el to Kff
      (setf pred-cov (m+ (pred-Kff gp) pred-cov))

      (list pred-mean pred-cov)))

  (:method ((gp variational-combined-output-parent-dependent-gp) pred-locs
            &key (closed-downwards-p nil) (use-all-locs nil))
    (declare (ignore closed-downwards-p use-all-locs))
    (let ((n-pred (list-length pred-locs))
          (n-latent (n-latent gp))
          KfuM
          pred-mean (pred-mean-out nil)
          pred-cov (pred-cov-out nil)
          q-mean-copy)
      (preprocess-prediction gp pred-locs n-pred)

      ;; These may not exist
      ;; TODO: Track when they need to be recreated
      (compute-covariances gp)
      (make-q-means gp)
      (compute-q-covariances gp)
      (make-posteriors gp)

      (setf (q-mean-copy gp) nil)

      (loop for q-mean in (q-mean gp)
            for Kuu-chol in (Kuu-chol gp)
            for pred-Kfu in (pred-Kfu gp)
            for middle in (middle gp)
            for pred-Kff in (pred-Kff gp)

            do (setf q-mean-copy (copy-mat q-mean)
                     (q-mean-copy gp)
                     (nconc (q-mean-copy gp) (list q-mean-copy)))

               ;; Kfu <- Kfu Kuu^-1
               (potrs! Kuu-chol pred-Kfu
                       :uplo #\L :n n-latent :lda n-latent
                       :nrhs n-pred :ldb n-latent :transpose-b? t)

               ;; Construct the needed KfuM
               ;; This will be zeroed out in symm!, so ok to be garbage
               (setf KfuM (make-mat (list n-pred n-latent)
                                    :ctype :double))
               ;; Compute KfuM
               (symm! 1d0 middle pred-Kfu
                      0d0 KfuM
                      :side #\R :uplo #\L :m n-pred
                      :n n-latent :lda n-latent :ldb n-latent :ldc n-latent)
               
               ;; Predictive mean
               (setf pred-mean (make-mat n-pred :ctype :double))
               (gemv! 1d0 pred-Kfu q-mean-copy 0d0 pred-mean
                      :m n-pred :n n-latent :lda n-latent)
               
               ;; Predictive covariance
               ;; Loop over u indices to build element (df-1,df-2)
               (setf pred-cov (make-mat (list n-pred n-pred)
                                        :ctype :double))
               (gemm! 1d0 KfuM pred-Kfu 1d0 pred-cov
                      :transpose-a? nil :transpose-b? t
                      :m n-pred :n n-pred :k n-latent
                      :lda n-latent :ldb n-latent :ldc n-pred)

               ;; Add pred-cov-el to Kff
               (setf pred-cov (m+ pred-Kff pred-cov)
                     pred-mean-out (nconc pred-mean-out (list pred-mean))
                     pred-cov-out (nconc pred-cov-out (list pred-cov))))

      (list pred-mean-out pred-cov-out))))
