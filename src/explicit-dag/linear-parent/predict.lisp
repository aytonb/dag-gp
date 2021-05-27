(uiop:define-package #:dag-gp/explicit-dag/linear-parent/predict
    (:use #:cl
          #:mgl-mat
          #:dag-gp/lapack
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/linear-parent/gp)
  (:export #:make-predictive-posteriors))

(in-package #:dag-gp/explicit-dag/linear-parent/predict)


;; Prediction isn't possible individually in a non-variational linear-parent-gp
;; due to not knowing the values of the parents.

;; (defmethod make-predictive-posteriors ((gp linear-parent-gp) pred-locs)
;;   (let* ((n-pred (list-length pred-locs))
;;          (pred-mean (make-mat n-pred :ctype :double))
;;          pred-obs-Kff-copy
;;          (true-obs-mat (make-mat (n-true-obs gp)
;;                                  :initial-contents (true-obs gp)
;;                                  :ctype :double)))
;;     (preprocess-prediction gp pred-locs n-pred)

;;     (setf pred-obs-Kff-copy (copy-mat (pred-obs-Kff gp)))

;;     ;; Need to also compute the Kff for the true observations. By design, this is
;;     ;; the first n-true-obs rows of Kff, and its Cholesky factorization is the first
;;     ;; n-true-obs rows of the Cholesky factorization of Kff.
;;     (reshape! (Kff gp) (list (n-true-obs gp) (n-true-obs gp)))

;;     (potrs! (Kff gp) true-obs-mat
;;             :uplo #\L :n (n-true-obs gp) :nrhs 1
;;             :lda (n-true-obs gp) :ldb (n-true-obs gp) :transpose-b? t)
;;     ;; pred-mean <- pred-obs-Kff Kff^-1 y
;;     (gemv! 1d0 (pred-obs-Kff gp) true-obs-mat 0d0 pred-mean
;;            :m n-pred :n (n-true-obs gp) :lda (n-true-obs gp))

;;     ;; pred-obs-Kff-copy <- pred-obs-Kff-copy Kff^-1
;;     (potrs! (Kff gp) pred-obs-Kff-copy
;;             :uplo #\L :n (n-true-obs gp) :nrhs n-pred
;;             :lda (n-true-obs gp) :ldb (n-true-obs gp) :transpose-b? t)
;;     ;; pred-Kff <- pred-Kff - pred-obs-Kff-copy Kff^-1 pred-obs-Kff^T
;;     (gemm! -1d0 pred-obs-Kff-copy (pred-obs-Kff gp) 1d0 (pred-Kff gp)
;;            :transpose-b? t :m n-pred :n n-pred :k (n-true-obs gp)
;;            :lda (n-true-obs gp) :ldb (n-true-obs gp) :ldc n-pred)

;;     (list pred-mean (pred-Kff gp))))
