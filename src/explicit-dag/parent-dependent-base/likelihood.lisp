(uiop:define-package #:dag-gp/explicit-dag/parent-dependent-base/likelihood
    (:use #:cl
          #:mgl-mat
          #:lbfgs-wrapper
          #:dag-gp/kernel
          #:dag-gp/output
          #:dag-gp/explicit-dag/parent-dependent-base/base
          #:dag-gp/explicit-dag/parent-dependent-base/covariance
          #:dag-gp/explicit-dag/parent-dependent-base/latent
          #:dag-gp/explicit-dag/parent-dependent-base/posterior)
  (:export #:NLL-and-derivs
           #:negative-log-likelihood
           #:train))

(in-package #:dag-gp/explicit-dag/parent-dependent-base/likelihood)


(defgeneric NLL-and-derivs (gp deriv-array &key latent)
  (:documentation "Computes the log likelihood and its derivatives for kernel parameters, posterior mean, posterior cholesky, and A-matrix parameters.")

  (:method ((gp parent-dependent-gp) deriv-array &key (latent nil))
    (declare (ignore latent))
    ;; Compute the covariance. Closed downwards GPs only use local obs for training
    (compute-covariances gp
                         :use-all-locs (not (closed-downwards-p gp)))

    (let* ((NLL (negative-log-likelihood gp)))

      (dLL/dpost gp)

      ;; Kernel derivatives
      (loop for param below (n-kern-params (kernel gp))
            do (setf (aref deriv-array param)
                     (dNLL/dkernparam gp (kernel gp) 0 param nil nil
                                      (dLL/dqf-var gp))))

      ;; Noise deriv
      (setf (aref deriv-array (n-kern-params (kernel gp)))
            (dNLL/dnoise gp))

      NLL))

  (:method ((gp variational-parent-dependent-gp) deriv-array &key (latent nil))
    (compute-covariances gp)
    (make-q-means gp)
    (compute-q-covariances gp)
    (make-posteriors gp)

    (let ((NLL (negative-log-likelihood gp)))
      
      (dLL/dpost gp)
      (let* ((dvarexp/dqcov (dvarexp/dqcov gp))
             (dLL/dKfu (dLL/dKfu gp)))
        (multiple-value-bind (dLL/dKuu dvarexp/dqmean) (dLL/dKuu gp dvarexp/dqcov)

          ;; If latent, do the latent derivatives only
          (if latent

              (progn
                (dNLL/dqmean gp dvarexp/dqmean deriv-array)
                (dNLL/dqchol gp dvarexp/dqcov deriv-array)
                (loop for param below (n-kern-params (kernel gp)) do
                  (setf (aref deriv-array param) 0d0)))

              (progn
                (let* ((mean-index (+ (n-kern-params (kernel gp))
                                      (n-output-params (output gp))))
                       (chol-index (+ mean-index (n-latent gp))))
                  (dotimes (i (n-latent gp))
                    (setf (aref deriv-array (+ mean-index i)) 0d0))
                  (dotimes (r (n-latent gp))
                    (loop for c below (1+ r)
                          for deriv-ind = (+ (/ (* r (1+ r)) 2) c chol-index)
                          ;; Multiply by column c of q-chol along rows and columns r
                          do (setf (aref deriv-array deriv-ind) 0d0)))
                  (loop for param below (n-kern-params (kernel gp)) do
                    (setf (aref deriv-array param)
                          (dNLL/dkernparam gp (kernel gp) 0 param dLL/dKfu dLL/dKuu
                                           (dLL/dqf-var gp)))))))
          
          
          ;; ;; Mean elements derivative
          ;; (dNLL/dqmean gp dvarexp/dqmean deriv-array)

          ;; ;; Cholesky elements derivative
          ;; (dNLL/dqchol gp dvarexp/dqcov deriv-array)

          ;; ;; Kernel derivatives
          ;; (loop for param below (n-kern-params (kernel gp))
          ;;       do (setf (aref deriv-array param)
          ;;                (dNLL/dkernparam gp (kernel gp) 0 param dLL/dKfu dLL/dKuu)))
          
          NLL))))

  (:method ((gp variational-combined-output-parent-dependent-gp) deriv-array
            &key (latent nil))
    (compute-covariances gp)
    (make-q-means gp)
    (compute-q-covariances gp)
    (make-posteriors gp)

    (let ((NLL (negative-log-likelihood gp))
          (offset 0)
          (n-latent (n-latent gp)))
      
      (dLL/dpost gp)
      (let* ((dvarexp/dqcov (dvarexp/dqcov gp))
             (dLL/dKfu (dLL/dKfu gp)))
        (multiple-value-bind (dLL/dKuu dvarexp/dqmean) (dLL/dKuu gp dvarexp/dqcov)

          ;; If latent, do the latent derivatives only
          (if latent

              (progn
                (dNLL/dqmean gp dvarexp/dqmean deriv-array)
                (dNLL/dqchol gp dvarexp/dqcov deriv-array)
                (loop for kernel in (kernel gp) do
                  (loop for param below (n-kern-params kernel) do
                    (setf (aref deriv-array (+ param offset)) 0d0))
                  (incf offset (n-kern-params kernel))))

              (progn
                (let* ((mean-index (+ (loop for kernel in (kernel gp)
                                            sum (n-kern-params kernel))
                                      (n-output-params (output gp))))
                       (chol-index (+ mean-index (* (n-combined gp) n-latent)))
                       (offset-inc (/ (* n-latent (1+ n-latent)) 2)))
                  (dotimes (gp-ind (n-combined gp))
                    (dotimes (i n-latent)
                      (setf (aref deriv-array (+ mean-index i offset)) 0d0))
                    (incf offset n-latent))

                  (setf offset 0)
                  (dotimes (gp-ind (n-combined gp))
                    (dotimes (r (n-latent gp))
                      (loop for c below (1+ r)
                            for deriv-ind = (+ (/ (* r (1+ r)) 2) c
                                               chol-index offset)
                            do (setf (aref deriv-array deriv-ind) 0d0)))
                    (incf offset offset-inc))

                  (setf offset 0)
                  (loop for kernel in (kernel gp)
                        for dLL/dKfu-i in dLL/dKfu
                        for dLL/dKuu-i in dLL/dKuu
                        for dLL/dqf-var-i in (dLL/dqf-var gp)
                        do (loop for param below (n-kern-params kernel) do
                          (setf (aref deriv-array (+ param offset))
                                (dNLL/dkernparam gp kernel 0 param
                                                 dLL/dKfu-i dLL/dKuu-i
                                                 dLL/dqf-var-i)))
                        (incf offset (n-kern-params kernel))))))
          
          NLL)))))


(defgeneric negative-log-likelihood (gp)
  (:documentation "Computes the negative-log likelihood of the data.")
  ;; Default goes here.
  )


(defgeneric train (gp &key progress-fn relative-tol)
  (:documentation "Trains the Gaussian Process to minimize negative log likelihood.")
  (:method ((gp parent-dependent-gp) &key (progress-fn nil) (relative-tol nil))
    (declare (ignore relative-tol))

    ;; TODO: Initialize if not already initialized?
    
    (let ((deriv-array (make-array (n-gp-params gp)
                                   :element-type 'double-float
                                   :initial-element 0d0))
          (x0 (make-array (n-gp-params gp)
                          :element-type 'double-float
                          :initial-contents (param-vec gp))))
      (flet ((min-fn (instance x n step)
               (declare (ignore instance n step))
               (update-parameter-vector gp x)
               (let ((NLL (NLL-and-derivs gp deriv-array)))
                 (list NLL deriv-array))))
        (let ((solver (make-instance 'lbfgs-solver
                                     :x x0
                                     :evaluation-fn #'min-fn
                                     :progress-fn progress-fn
                                     :n (n-gp-params gp)
                                     :past 0
                                     :delta 1d-4
                                     )))
          (lbfgs-solve solver)
          (destructuring-bind (x-best NLL-best)
              (lbfgs-solution solver)
            (update-parameter-vector gp x-best)
            (- NLL-best))))))

  (:method ((gp variational-parent-dependent-gp) &key (progress-fn nil) (relative-tol 0.02))
    (let ((prev-NLL nil))
      (loop do

        ;; Latent true
        (let ((deriv-array (make-array (n-gp-params gp)
                                       :element-type 'double-float
                                       :initial-element 0d0))
              (x0 (make-array (n-gp-params gp)
                              :element-type 'double-float
                              :initial-contents (param-vec gp))))
          (flet ((min-fn (instance x n step)
                   (declare (ignore instance n step))
                   (update-parameter-vector gp x)
                   (let ((NLL (NLL-and-derivs gp deriv-array :latent t)))
                     (list NLL deriv-array))))
            (let ((solver (make-instance 'lbfgs-solver
                                         :x x0
                                         :evaluation-fn #'min-fn
                                         :progress-fn progress-fn
                                         :n (n-gp-params gp)
                                         :past 1
                                         :delta 1d-6)))
              (lbfgs-solve solver)
              (destructuring-bind (x-best NLL-best)
                  (lbfgs-solution solver)
                (update-parameter-vector gp x-best)

                ;; Quit loop if converged
                (when (and prev-NLL
                           (<= (/ (abs (- NLL-best prev-NLL))
                                  (abs NLL-best))
                               relative-tol))
                  (return-from train (- NLL-best)))

                ;(when (equal progress-fn :summary)
                  (format t "[~a] " (- NLL-best))
                ;  )
                (setf prev-NLL NLL-best)))))
        
        
        ;; Latent false
        (let ((deriv-array (make-array (n-gp-params gp)
                                       :element-type 'double-float
                                       :initial-element 0d0))
              (x0 (make-array (n-gp-params gp)
                              :element-type 'double-float
                              :initial-contents (param-vec gp))))
          (flet ((min-fn (instance x n step)
                   (declare (ignore instance n step))
                   (update-parameter-vector gp x)
                   (let ((NLL (NLL-and-derivs gp deriv-array :latent nil)))
                     (list NLL deriv-array))))
            (let ((solver (make-instance 'lbfgs-solver
                                         :x x0
                                         :evaluation-fn #'min-fn
                                         :progress-fn progress-fn
                                         :n (n-gp-params gp)
                                         :past 0
                                         :delta 1d-6)))
              (lbfgs-solve solver)
              (destructuring-bind (x-best NLL-best)
                  (lbfgs-solution solver)
                (update-parameter-vector gp x-best)

                ;; Quit loop if converged
                (when (and prev-NLL
                           (<= (/ (abs (- NLL-best prev-NLL))
                                  (abs NLL-best))
                               relative-tol))
                  (return-from train (- NLL-best)))
                
                ;(when (equal progress-fn :summary)
                (format t "[~a] " (- NLL-best))
                                        ;)
                (setf prev-NLL NLL-best)))))

        ;; ;; Latent true
        ;; (let ((deriv-array (make-array (n-gp-params gp)
        ;;                                :element-type 'double-float
        ;;                                :initial-element 0d0))
        ;;       (x0 (make-array (n-gp-params gp)
        ;;                       :element-type 'double-float
        ;;                       :initial-contents (param-vec gp))))
        ;;   (flet ((min-fn (instance x n step)
        ;;            (declare (ignore instance n step))
        ;;            (update-parameter-vector gp x)
        ;;            (let ((NLL (NLL-and-derivs gp deriv-array :latent t)))
        ;;              (list NLL deriv-array))))
        ;;     (let ((solver (make-instance 'lbfgs-solver
        ;;                                  :x x0
        ;;                                  :evaluation-fn #'min-fn
        ;;                                  :progress-fn progress-fn
        ;;                                  :n (n-gp-params gp)
        ;;                                  :past 1
        ;;                                  :delta 1d-6)))
        ;;       (lbfgs-solve solver)
        ;;       (destructuring-bind (x-best NLL-best)
        ;;           (lbfgs-solution solver)
        ;;         (update-parameter-vector gp x-best)

        ;;         ;; Quit loop if converged
        ;;         (when (and prev-NLL
        ;;                    (<= (/ (abs (- NLL-best prev-NLL))
        ;;                           (abs NLL-best))
        ;;                        relative-tol))
        ;;           (return-from train (- NLL-best)))

        ;;         ;(when (equal progress-fn :summary)
        ;;           (format t "[~a] " (- NLL-best))
        ;;         ;  )
        ;;         (setf prev-NLL NLL-best)))))

            ))))
