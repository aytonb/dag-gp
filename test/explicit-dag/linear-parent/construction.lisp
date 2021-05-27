(uiop:define-package #:dag-gp-test/explicit-dag/linear-parent/construction
    (:use #:cl
          #:fiveam
          #:mgl-mat
          #:dag-gp/quadrature
          #:dag-gp/kernel
          #:dag-gp/output
          #:dag-gp/utils/distance
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/parent-dependent-base/base
          #:dag-gp/explicit-dag/parent-dependent-base/likelihood
          #:dag-gp/explicit-dag/linear-parent/all
          #:dag-gp/explicit-dag/linear-parent/gp)
  (:export #:make-uninitialized-1d-linear-parent-gp
           #:make-uninitialized-1d-variational-linear-parent-gp
           #:make-uninitialized-1d-variational-combined-output-linear-parent-gp)
  (:documentation "Tests that data containers are made and appropriately sized."))

(in-package #:dag-gp-test/explicit-dag/linear-parent/construction)


(defun make-empty-1d-linear-parent-gp ()
  (let ((parents (list (make-instance 'linear-parent-gp
                                      :kernel (make-instance 'rbf-kernel)
                                      :output (make-instance 'gaussian-output)
                                      :parent-gps nil)
                       (make-instance 'linear-parent-gp
                                      :kernel (make-instance 'rbf-kernel)
                                      :output (make-instance 'gaussian-output)
                                      :parent-gps nil))))
  
    (make-instance 'linear-parent-gp
                   :kernel (make-instance 'rbf-kernel)
                   :output (make-instance 'gaussian-output)
                   :parent-gps parents)))

(defun make-empty-1d-variational-linear-parent-gp ()
  (let ((parents (list (make-instance 'variational-linear-parent-gp
                                      :kernel (make-instance 'rbf-kernel)
                                      :output (make-instance 'gaussian-output)
                                      :parent-gps nil)
                       (make-instance 'variational-linear-parent-gp
                                      :kernel (make-instance 'rbf-kernel)
                                      :output (make-instance 'gaussian-output)
                                      :parent-gps nil))))
  
    (make-instance 'variational-linear-parent-gp
                   :kernel (make-instance 'rbf-kernel)
                   :output (make-instance 'gaussian-output)
                   :parent-gps parents
                   :input-dim 1
                   :n-latent 2)))

(defun make-empty-1d-variational-combined-output-linear-parent-gp ()
  (let ((parents (list (make-instance 'variational-combined-output-linear-parent-gp
                                      :kernel (list (make-instance 'rbf-kernel)
                                                    (make-instance 'rbf-kernel))
                                      :output (make-instance 'n-ary-output
                                                             :n-params 3)
                                      :parent-gps nil)
                       (make-instance 'variational-combined-output-linear-parent-gp
                                      :kernel (list (make-instance 'rbf-kernel)
                                                    (make-instance 'rbf-kernel))
                                      :output (make-instance 'n-ary-output
                                                             :n-params 3)
                                      :parent-gps nil))))
  (make-instance 'variational-combined-output-linear-parent-gp
                 :kernel (list (make-instance 'rbf-kernel)
                               (make-instance 'rbf-kernel))
                 :output (make-instance 'n-ary-output :n-params 3)
                 :parent-gps parents
                 :input-dim 1
                 :n-latent 2)))


(defun make-uninitialized-1d-linear-parent-gp ()
  (let ((gp (make-empty-1d-linear-parent-gp)))
    (add-single-output-measurement gp '(0) 1d0)
    gp))

(defun make-uninitialized-1d-variational-linear-parent-gp ()
  (let ((gp (make-empty-1d-variational-linear-parent-gp)))  
    (add-single-output-measurement gp '(0) 1d0)
    gp))

(defun make-uninitialized-1d-variational-combined-output-linear-parent-gp ()
  (let ((gp (make-empty-1d-variational-combined-output-linear-parent-gp)))
    (add-single-output-measurement gp '(0) 2)
    gp))




;; (defun quick-test ()
;;   (let* ((gp (make-parameter-controlled-1d-linear-parent-gp))
;;          (deriv-array (make-array (n-gp-params gp) :element-type 'double-float))
;;          NLL)
;;     ;;(initialize-measurements gp '((0) (2) (5)))
;;     ;;(initialize-gp gp)
;;     (setf NLL (NLL-and-derivs gp deriv-array))
;;     (format t "NLL = ~a~%" NLL)
;;     (format t "deriv array = ~a~%" deriv-array)

;;     ;(format t "Prediction at (7) = ~a~%" (make-predictive-posteriors gp '((7))))
;;     ;(format t "Prediction at (6) = ~a~%" (make-predictive-posteriors gp '((6))))
    
;;     gp))



;; (defun quick-variational-combined-output-test ()
;;   (let* ((gp (make-parameter-controlled-1d-variational-combined-output-linear-parent-gp))
;;          (deriv-array (make-array (n-gp-params gp) :element-type 'double-float))
;;          NLL)
;;     (setf NLL (NLL-and-derivs gp deriv-array :latent nil))
;;     (format t "NLL = ~a~%" NLL)
;;     (format t "deriv array = ~a~%" deriv-array)

;;     (setf NLL (NLL-and-derivs gp deriv-array :latent t))
;;     (format t "NLL = ~a~%" NLL)
;;     (format t "deriv array = ~a~%" deriv-array)

;;     gp))


;; (defun train-test ()
;;   (let* ((gp (make-parameter-controlled-1d-linear-parent-gp)))
;;     (train gp :progress-fn :verbose)
;;     gp))

;; (defun variational-train-test ()
;;   (let* ((gp (make-parameter-controlled-1d-variational-linear-parent-gp)))
;;     (train gp :progress-fn :verbose)
;;     gp))
