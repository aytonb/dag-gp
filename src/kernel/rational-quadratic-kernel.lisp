(uiop:define-package #:dag-gp/kernel/rational-quadratic-kernel
    (:use #:cl
          #:dag-gp/kernel/kernel)
  (:import-from #:array-operations)
  (:export #:rational-quadratic-kernel))

(in-package #:dag-gp/kernel/rational-quadratic-kernel)


(defclass rational-quadratic-kernel (kernel)
  ((n-parameters
    :initform 3)))


(defmethod evaluate ((kernel rational-quadratic-kernel) sq-dist &optional abs-dist)
  (declare (ignore abs-dist))
  (let ((alpha (safe-exp (aref (kern-params kernel) 0)))
        (sigma (safe-exp (aref (kern-params kernel) 2))))
    (* sigma
       (expt (1+ (/ sq-dist
                    (* alpha (safe-exp (aref (kern-params kernel) 1)))))
             (- alpha)))))


(defmethod dk/dparam ((kernel rational-quadratic-kernel) param sq-dist
                      &optional abs-dist)
  (declare (ignore abs-dist))
  (let* ((alpha (safe-exp (aref (kern-params kernel) 0)))
         (sigma (safe-exp (aref (kern-params kernel) 2)))
         (len (safe-exp (aref (kern-params kernel) 1)))
         (fac (1+ (/ sq-dist (* alpha len)))))
    (ecase param
      (0 (* sigma
            alpha
            (expt fac (- alpha))
            (- (/ sq-dist (* alpha len fac))
               (log fac))))
      (1 (* sigma
            (/ sq-dist len)
            (expt fac (1- (- alpha)))))
      (2 (* sigma
            (expt (1+ (/ sq-dist
                         (* alpha (safe-exp (aref (kern-params kernel) 1)))))
                  (- alpha)))))))


(defmethod initialize-kernel-params ((kernel rational-quadratic-kernel))
  (if (default-kern-params kernel)
      (setf (aref (kern-params kernel) 0) (aref (default-kern-params kernel) 0)
            (aref (kern-params kernel) 1) (aref (default-kern-params kernel) 1)
            (aref (kern-params kernel) 2) (aref (default-kern-params kernel) 2))
      (setf (aref (kern-params kernel) 0) (1- (random 2d0))
            (aref (kern-params kernel) 1) (1- (random 2d0))
            (aref (kern-params kernel) 2) (1- (random 2d0)))))


(defun safe-exp (f)
  (cond
    ((> f 11.5d0)
     (exp 11.5d0))
    ((< f -11.5d0)
     (exp -11.5d0))
    (t
     (exp f))))
