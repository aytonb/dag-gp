(uiop:define-package #:dag-gp/output/binary-output
    (:use #:cl
          #:dag-gp/output/output)
  (:export #:binary-output
           #:log-lik
           #:dlog-lik/df
           #:child-contribution
           #:dchild/dparent-param
           #:make-quadrature-basis))

(in-package #:dag-gp/output/binary-output)


(defclass binary-output (output)
  ((n-parameters
    :initform 0
    :accessor n-output-parameters)
   (n-child-parameters
    :initform 2
    :accessor n-output-child-params)))


(defmethod log-lik ((output binary-output) f y-and-parent-value
                    parent-outputs parent-params)
  (let ((y (first y-and-parent-value))
        (parent-values (rest y-and-parent-value))
        (effective-f f))
    (loop for parent-value in parent-values
          for parent-output in parent-outputs
          for param in parent-params
          do (incf effective-f (child-contribution parent-output
                                                   parent-value
                                                   param)))

    ;; For f > 10, log(1/(1+e^-f)) ~ -e^-f
    ;; For f < -10, log(1/(1+e^-f)) ~ f
    ;; For f > 10, log(1 - 1/(1+e^-f)) ~ -f
    ;; For f < -10, log(1 - 1/(1+e^-f)) ~ -e^f
    (cond
      ((> effective-f 10d0)
       (if (equal y 1)
           (- (exp (- effective-f)))
           (- effective-f)))
      ((< effective-f -10d0)
       (if (equal y 1)
           effective-f
           (- (exp effective-f))))
      (t
       (let ((link (/ (1+ (exp (- effective-f))))))
         (if (equal y 1)
             (log link)
             (log (- 1 link))))))))


(defmethod dlog-lik/df ((output binary-output) index f y-and-parent-value
                        parent-outputs parent-params)
  (assert (equal index 0) (index))
  (let ((y (first y-and-parent-value))
        (parent-values (rest y-and-parent-value))
        (effective-f f))
    (loop for parent-value in parent-values
          for parent-output in parent-outputs
          for param in parent-params
          do (incf effective-f (child-contribution parent-output
                                                   parent-value
                                                   param)))
    (if (equal y 1)
        (/ (1+ (exp effective-f)))
        (- (/ (1+ (exp (- effective-f))))))))


(defmethod child-contribution ((output binary-output) y child-parent-params)
  (aref child-parent-params y))


(defmethod dchild/dparent-param ((output binary-output) param y child-parent-params)
  (if (equal param y)
      1d0
      0d0))


(defmethod make-quadrature-basis ((output binary-output) mean var parent-vals
                                  n-parents parent-outputs parent-params)
  (let ((parent-base 0d0)
        (probs (list (list 0 0d0) (list 1 0d0))))
    (loop for parent-val in parent-vals
          for parent-output in parent-outputs
          for parent-param in parent-params
          do (incf parent-base (child-contribution parent-output
                                                   parent-val
                                                   parent-param)))
    (loop for point in (quad-points output)
          for weight in (quad-weights output)
          for f = (+ parent-base mean (* (sqrt (* 2 var)) point))
          for p-1 = (/ (1+ (safe-exp (- f))))
          do (incf (second (first probs)) (* weight (- 1 p-1)))
             (incf (second (second probs)) (* weight p-1)))
    probs))


(defun safe-exp (f)
  (cond
    ((> f 11.5d0)
     (exp 11.5d0))
    ((< f -11.5d0)
     (exp -11.5d0))
    (t
     (exp f))))
