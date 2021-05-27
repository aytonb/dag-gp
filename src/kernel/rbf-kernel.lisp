(uiop:define-package #:dag-gp/kernel/rbf-kernel
    (:use #:cl
          #:dag-gp/kernel/kernel)
  (:import-from #:array-operations)
  (:export #:rbf-kernel))

(in-package #:dag-gp/kernel/rbf-kernel)


(defclass rbf-kernel (kernel)
  ((n-parameters
    ;:accessor n-kern-params
    :initform 2)))


(defmethod evaluate ((kernel rbf-kernel) sq-dist &optional abs-dist)
  ;; To restrict to positive values, take sigma^2 and l to be exponentials.
  (declare (ignore abs-dist))
  (* (safe-exp (aref (kern-params kernel) 0))
     (exp (- (/ sq-dist (safe-exp (aref (kern-params kernel) 1)))))))


(defmethod dk/dparam ((kernel rbf-kernel) param sq-dist &optional abs-dist)
  (declare (ignore abs-dist))
  (ecase param
    (0 (* (safe-exp (aref (kern-params kernel) 0))
          (exp (- (/ sq-dist (safe-exp (aref (kern-params kernel) 1)))))))
    (1 (* (safe-exp (aref (kern-params kernel) 0))
          (exp (- (/ sq-dist (safe-exp (aref (kern-params kernel) 1)))))
          (/ sq-dist (safe-exp (aref (kern-params kernel) 1)))))))


;; (defmethod dk/dparam-expr ((kernel rbf-kernel) param)
;;   (ecase param
;;     (0 `(* ,(exp (aref (kern-params kernel) 0))
;;              (exp (/ sq-dist ,(- (exp (aref (kern-params kernel) 1)))))))
;;     (1 `(* ,(exp (aref (kern-params kernel) 0))
;;            (exp (/ sq-dist ,(- (exp (aref (kern-params kernel) 1)))))
;;            (/ sq-dist ,(exp (aref (kern-params kernel) 1)))))))


(defmethod initialize-kernel-params ((kernel rbf-kernel))
  (if (default-kern-params kernel)
      (setf (aref (kern-params kernel) 0) (aref (default-kern-params kernel) 0)
            (aref (kern-params kernel) 1) (aref (default-kern-params kernel) 1))
      (setf (aref (kern-params kernel) 0) (1- (random 2d0))
            (aref (kern-params kernel) 1) (1- (random 2d0))))) 


;; (defun time-test ()
;;   (let* ((kern (make-instance 'rbf-kernel)))
;;     (time (progn
;;             (flet ((f (sq-dist &optional abs-dist) (dk/dparam kern 1 sq-dist abs-dist)))
;;               (dotimes (i 1000000)
;;                 (f 1.5d0)))))
;;     (time (progn
;;             (update-kernel kern)
;;             (let ((f (nth 1 (grad-fn kern))))
;;               (dotimes (i 1000000)
;;                 (funcall f 1.5d0)))))))


(defun safe-exp (f)
  (cond
    ((> f 11.5d0)
     (exp 11.5d0))
    ((< f -11.5d0)
     (exp -11.5d0))
    (t
     (exp f))))
