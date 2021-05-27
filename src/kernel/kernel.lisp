(uiop:define-package #:dag-gp/kernel/kernel
    (:use #:cl)
  (:import-from #:array-operations)
  (:export #:kernel
           #:kern-params
           #:n-parameters
           #:n-kern-params
           #:default-kern-params
           #:eval-fn
           #:grad-fn
           #:dist-index
           #:operate-on-parent
           #:evaluate
           #:evaluate-matrix
           #:dk/dparam
           #:dk/dparam-matrix
           #:update-kernel
           #:initialize-kernel-params
           #:copy-kernel))

(in-package #:dag-gp/kernel/kernel)


(defclass kernel ()
  ((parameters
    :accessor kern-params
    :documentation "A vector of parameters for this kernel.")
   (n-parameters
    :accessor n-kern-params
    :documentation "The number of kernel parameters.")
   (default-kernel-parameters
    :initarg :default-kern-params
    :initform nil
    :accessor default-kern-params
    :documentation "Kernel parameters used as the starting point of optimization. If nil, parameters are randomized.")
   (evaluation-function
    :accessor eval-fn
    :documentation "A function that calls kernel evaluate.")
   (gradient-function
    :accessor grad-fn
    :documentation "A function that calls gradient evaluation.")
   (distance-index
    :initarg :dist-index
    :initform 0
    :accessor dist-index
    :documentation "The kernel uses distances at this index.")
   (operate-on-parent
    :initarg :operate-on-parent
    :initform nil
    :accessor operate-on-parent
    :documentation "If non-nil, operate on parent outputs instead.")))


(defmethod initialize-instance :after ((kernel kernel) &key)
  ;; Now will be handled by the initializer of the GP
  (setf (kern-params kernel) (make-array (n-kern-params kernel)
                                         :element-type 'double-float
                                         :adjustable t)))


(defgeneric evaluate-matrix (kernel sq-dists abs-dists
                             parent-sq-dists parent-abs-dists
                             &key add-jitter)
  (:documentation "Evaluates the kernel k(x1,x2) for all sq-dists and abs-dists.")
  (:method ((kernel kernel) sq-dists abs-dists
            parent-sq-dists parent-abs-dists
            &key (add-jitter t))
    (let ((sq-dist-part (if (operate-on-parent kernel)
                            (nth (dist-index kernel)
                                 (nth (operate-on-parent kernel) parent-sq-dists))
                            (nth (dist-index kernel) sq-dists)))
          (abs-dist-part (if (operate-on-parent kernel)
                             (nth (dist-index kernel)
                                  (nth (operate-on-parent kernel) parent-abs-dists))
                             (nth (dist-index kernel) abs-dists))))
      (flet ((eval-fn (sq-dist abs-dist)
               (if (and add-jitter
                        (equalp sq-dist 0d0))
                   (+ (evaluate kernel sq-dist abs-dist) 1d-5)
                   (evaluate kernel sq-dist abs-dist))))
        (aops:each #'eval-fn sq-dist-part abs-dist-part)))))


(defgeneric evaluate (kernel sq-dist &optional abs-dist)
  (:documentation "Evaluates the kernel k(x1,x2)."))


(defgeneric dk/dparam (kernel param sq-dist &optional abs-dist)
  (:documentation "Evaluates the kernel derivatives dk(x1,x2)/dparam."))


(defgeneric dk/dparam-matrix (kernel param sq-dists abs-dists parent-sq-dists parent-abs-dists)
  (:documentation "Evaluates the kernel derivatives dk(x1,x2)/dparam for all sq-dists and abs-dists.")
  (:method ((kernel kernel) param sq-dists abs-dists parent-sq-dists parent-abs-dists)
    (let ((sq-dist-part (if (operate-on-parent kernel)
                            (nth (dist-index kernel)
                                 (nth (operate-on-parent kernel) parent-sq-dists))
                            (nth (dist-index kernel) sq-dists)))
          (abs-dist-part (if (operate-on-parent kernel)
                             (nth (dist-index kernel)
                                  (nth (operate-on-parent kernel) parent-abs-dists))
                             (nth (dist-index kernel) abs-dists))))
      (flet ((grad-fn (sq-dist abs-dist)
               (dk/dparam kernel param sq-dist abs-dist)))
        (aops:each #'grad-fn sq-dist-part abs-dist-part)))))


;; This seems to be a micro-optimization which only pays off in the millions of
;; hits to the eval and grad-fns
;; (defgeneric update-kernel (kernel)
;;   (:documentation "Recreates the evaluation and gradient functions.")
;;   (:method ((kernel kernel))
;;     (setf (eval-fn kernel) (lambda (sq-dist &optional abs-dist)
;;                              (evaluate kernel sq-dist abs-dist))
;;           (grad-fn kernel) (loop for param below (n-kern-params kernel)
;;                                  collect (compile nil
;;                                                   `(lambda (sq-dist &optional abs-dist)
;;                                                      (declare (ignorable abs-dist))
;;                                                      ,(dk/dparam-expr kernel param)))))))


(defgeneric initialize-kernel-params (kernel)
  (:documentation "Sets the kernel parameters to their initial values."))


;; If operate-on-parent is specified, it should change the value of the operate-on-parent slot
;; only if it is specified.
(defgeneric copy-kernel (kernel &key operate-on-parent)
  (:documentation "Makes a copy of this kernel with its own independent parameters.")
  (:method ((kernel kernel) &key (operate-on-parent :copy))
    (make-instance (type-of kernel)
                   :dist-index (dist-index kernel)
                   :operate-on-parent (cond
                                        ((equal operate-on-parent :copy)
                                         (operate-on-parent kernel))
                                        ((operate-on-parent kernel)
                                         operate-on-parent)
                                        (t
                                         nil))
                   :default-kern-params (default-kern-params kernel))))

