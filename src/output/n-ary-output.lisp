(uiop:define-package #:dag-gp/output/n-ary-output
    (:use #:cl
          #:mgl-mat
          #:dag-gp/quadrature
          #:dag-gp/output/output
          #:dag-gp/output/multi-output)
  (:export #:n-ary-output
                                        ;#:multi-output
           ;#:output-params
           ;#:n-parameters
           ;#:n-output-params
           ;#:initialize-output-params
           #:log-lik
           ;#:LL-single
           ;#:LL
           ;#:dLL/dmu+dvar
           ;#:dLL/doutputparam
           ;#:dLL-single/doutputparam
           ;#:dLL/dparentparam
                                        ;#:dLL-single/dparentparam
           #:child-contribution))

(in-package #:dag-gp/output/n-ary-output)


(defclass n-ary-output (multi-output)
  ((n-parameters
    :initform 0
    :accessor n-output-params))
  (:documentation "An output belonging to a discrete class."))


(defmethod initialize-instance :before ((output n-ary-output) &key n-params)
  (setf (n-gps output) (1- n-params)
        (n-output-child-params output) n-params))


(defmethod log-lik ((output n-ary-output) f y-and-parent-value
                    parent-outputs parent-params)
  ;; When y = n-params, then num is 1, so its log is 0
  (let ((y (first y-and-parent-value))
        (parent-values (rest y-and-parent-value))
        (num 0d0)
        (den 1d0))
    ;; Determine the effective value of f
    (loop for i below (n-gps output)
          for gp-parent-params in parent-params
          for gp-f in f
          for effective-gp-f = gp-f
          ;; Add the parents
          do (loop for parent-output in parent-outputs
                   for parent-value in parent-values
                   for param in gp-parent-params
                   do (incf effective-gp-f
                            (child-contribution parent-output
                                                parent-value
                                                param)))
             
             ;; Likelihood numerator and denominator
             (when (equalp i y)
               (setf num (log (safe-exp effective-gp-f))))
             (incf den (safe-exp effective-gp-f)))

    ;; Return the log likelihood
    (- num (log den))))


(defmethod dlog-lik/df ((output n-ary-output) index f y-and-parent-value
                        parent-outputs parent-params)
  ;; d log(S(y))/findex = 1 - S(index) if y = index
  ;;                      -S(index) otherwise
  (let* ((y (first y-and-parent-value))
         (parent-values (rest y-and-parent-value))
         num
         (den 1d0))
    (loop for i below (n-gps output)
          for gp-parent-params in parent-params
          for gp-f in f
          for effective-gp-f = gp-f
          ;; Add the parents
          do (loop for parent-output in parent-outputs
                   for parent-value in parent-values
                   for param in gp-parent-params
                   do (incf effective-gp-f
                            (child-contribution parent-output
                                                parent-value
                                                param)))
             
             ;; Likelihood numerator and denominator
             (when (equalp i index)
               (setf num (safe-exp effective-gp-f)))
             (incf den (safe-exp effective-gp-f)))
    
    (if (equal index y)
        (- 1 (/ num den))
        (- (/ num den)))))


(defmethod child-contribution ((output n-ary-output) y child-parent-params)
  (aref child-parent-params y))


(defmethod dchild/dparent-param ((output n-ary-output) param y child-parent-params)
  (if (equal param y)
      1d0
      0d0))


(defmethod make-quadrature-basis ((output n-ary-output) mean var parent-vals
                                  n-parents parent-outputs parent-params)
  (let ((parent-bases (make-list (n-gps output) :initial-element 0d0))
        probs
        f-list
        den)

    ;; Make each of the parent bases
    (setf parent-bases
          (loop for i below (n-gps output)
                for gp-parent-params in parent-params
                collect (loop for parent-val in parent-vals
                              for parent-output in parent-outputs
                              for parent-param in gp-parent-params
                              sum (child-contribution parent-output
                                                      parent-val
                                                      parent-param))))

    ;; Set up the quadrature output 
    (setf probs (loop for i upto (n-gps output) collect (list i 0d0)))

    ;; Do the quadrature, looping through points and weights
    (loop for point in (flattened (quadrature-set output))
          for weight in (weights (quadrature-set output))
          do (setf den 1d0
                   ;; Loop through the gp values
                   f-list (loop for point-i in point
                                for mean-i in mean
                                for var-i in var
                                for base-i in parent-bases
                                for f = (safe-exp (+ base-i mean-i
                                                     (* (sqrt (* 2 var-i)) point-i)))
                                do (incf den f)
                                collect f))
             ;; Increment the probabilities
             (loop for i below (n-gps output)
                   for f in f-list
                   do (incf (second (nth i probs)) (* weight (/ f den))))
             (incf (second (nth (n-gps output) probs)) (* weight (/ 1d0 den))))

    probs))


(defun safe-exp (f)
  (cond
    ((> f 11.5d0)
     (exp 11.5d0))
    ((< f -11.5d0)
     (exp -11.5d0))
    (t
     (exp f))))
