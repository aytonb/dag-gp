(uiop:define-package #:dag-gp/explicit-dag/parent-dependent-base/divergence
    (:use #:cl
          #:mgl-mat
          #:dag-gp/lapack
          #:dag-gp/explicit-dag/parent-dependent-base/base)
  (:export #:post-prior-kl))

(in-package #:dag-gp/explicit-dag/parent-dependent-base/divergence)


(defgeneric post-prior-kl (gp)
  (:documentation "Computes the KL divergence KL(q(u)||p(u)).")
  (:method ((gp variational-parent-dependent-gp))
    ;; D(N0||N1) = 1/2 (tr(S1^-1 S0) + m0 S1^-1 m0 + logdet(S1) - logdet(S0))
    (let ((KL 0)
          (count 0)
          (n-latent (n-latent gp))
          (Kuu-inv (copy-mat (Kuu-chol gp)))
          (q-cov (q-cov gp))
          (q-mean-copy (copy-mat (q-mean gp)))
          (q-chol (q-chol gp)))
      
      ;; Kuu-chol copy <- Kuu^-1
      (symmetric-potri! Kuu-inv :n n-latent :uplo #\L :lda n-latent)
      (setf (Kuu-inv gp) Kuu-inv)
      
      ;; Compute trace as sum of dot products of rows
      ;; (because they are symmetric)
      (loop for lat-ind below n-latent
            for displace = (* lat-ind n-latent)
            do (reshape-and-displace! Kuu-inv n-latent displace)
               (reshape-and-displace! q-cov n-latent displace)
               (incf KL (dot Kuu-inv q-cov :n n-latent)))

      ;; Fix the shaping of Kuu-inv
      (reshape-and-displace! Kuu-inv (list n-latent n-latent) 0)
      
      ;; q-mean-copy <- Kuu^-1 q-mean-copy
      (potrs! (Kuu-chol gp) q-mean-copy :n n-latent :uplo #\L :nrhs 1
                                        :lda n-latent :ldb n-latent
                                        :transpose-b? t)
      (incf KL (dot (q-mean gp) q-mean-copy :n n-latent))

      ;; logdet(S1) = 2*sum(log(diag(chol(S1))))
      ;; Flipping the value of diagonal cholesky elements doesn't
      ;; influence determinant. Need to refind proof.
      (dotimes (i n-latent)
        (incf KL (* -2 (log (abs (aref q-chol count)))))
        (incf count (+ 2 i)))

      ;; Elements should be positive. Add jitter.
      (with-facets ((Kuu-chol-facet ((Kuu-chol gp) 'array :direction :input)))
        (dotimes (i n-latent)
          (incf KL (* 2 (log (+ (aref Kuu-chol-facet i i)
                                1d-10))))))

      (* 0.5 KL)))

  (:method ((gp variational-combined-output-parent-dependent-gp))
    (let ((KL 0)
          count
          (n-latent (n-latent gp))
          Kuu-inv
          q-mean-copy)
      (setf (Kuu-inv gp) nil)
      
      (loop for Kuu-chol in (Kuu-chol gp)
            for q-cov in (q-cov gp)
            for q-mean in (q-mean gp)
            for q-chol in (q-chol gp)

            do (setf Kuu-inv (copy-mat Kuu-chol)
                     q-mean-copy (copy-mat q-mean))
               
               ;; Kuu-chol copy <- Kuu^-1
               (symmetric-potri! Kuu-inv :n n-latent :uplo #\L :lda n-latent)
               (setf (Kuu-inv gp) (nconc (Kuu-inv gp) (list Kuu-inv)))
             
               ;; Compute trace as sum of dot products of rows
               ;; (because they are symmetric)
               (loop for lat-ind below n-latent
                     for displace = (* lat-ind n-latent)
                     do (reshape-and-displace! Kuu-inv n-latent displace)
                        (reshape-and-displace! q-cov n-latent displace)
                        (incf KL (dot Kuu-inv q-cov :n n-latent)))

               ;; Fix the shaping of Kuu-inv
               (reshape-and-displace! Kuu-inv (list n-latent n-latent) 0)
      
               ;; q-mean-copy <- Kuu^-1 q-mean-copy
               (potrs! Kuu-chol q-mean-copy :n n-latent :uplo #\L :nrhs 1
                                            :lda n-latent :ldb n-latent
                                            :transpose-b? t)
               (incf KL (dot q-mean q-mean-copy :n n-latent))

               ;; logdet(S1) = 2*sum(log(diag(chol(S1))))
               ;; Flipping the value of diagonal cholesky elements doesn't
               ;; influence determinant. Need to refind proof.
               (setf count 0)
               (dotimes (i n-latent)
                 (incf KL (* -2 (log (abs (aref q-chol count)))))
                 (incf count (+ 2 i)))
               
               ;; Elements should be positive. Add jitter.
               (with-facets ((Kuu-chol-facet (Kuu-chol 'array :direction :input)))
                 (dotimes (i n-latent)
                   (incf KL (* 2 (log (+ (aref Kuu-chol-facet i i)
                                         1d-10)))))))

      (* 0.5 KL))))
