(uiop:define-package #:dag-gp/output/gaussian-output
    (:use #:cl
          #:mgl-mat
          #:dag-gp/quadrature
          #:dag-gp/lapack
          #:dag-gp/output/output)
  (:export #:gaussian-output
           #:initialize-output-params
           #:LL-single
           #:dLL/dmu+dvar
           #:dLL-single/doutputparam
           #:dLL-single/dparentparam
           #:child-contribution
           #:make-quadrature-basis))

(in-package #:dag-gp/output/gaussian-output)


(defclass gaussian-output (output)
  ((n-parameters
    :initform 1
    :accessor n-output-params)
   (n-child-parameters
    :initform 1
    :accessor n-output-child-params)))


(defmethod initialize-output-params ((output gaussian-output))
  (setf (aref (output-params output) 0) -4.5d0 ;(- -4d0 (random 1d0))
        ))


(defmethod log-lik ((output gaussian-output) f y-and-parent-value
                    parent-outputs parent-params)
  (let ((meas-value (first y-and-parent-value))
        (parent-values (rest y-and-parent-value))
        (noise-var (safe-exp (aref (output-params output) 0)))
        (effective-f f))
    (loop for parent-value in parent-values
          for parent-output in parent-outputs
          for param in parent-params
          do (incf effective-f (child-contribution parent-output
                                                   parent-value
                                                   param)))

    (* -0.5d0 (+ (/ (expt (- effective-f meas-value) 2)
                    noise-var)
                 (log noise-var)))))


(defmethod dlog-lik/df ((output gaussian-output) index f y-and-parent-value
                        parent-outputs parent-params)
  (assert (equal index 0) (index))
  (let ((y (first y-and-parent-value))
        (parent-values (rest y-and-parent-value))
        (noise-var (safe-exp (aref (output-params output) 0)))
        (effective-f f))
    (loop for parent-value in parent-values
          for parent-output in parent-outputs
          do (incf effective-f (child-contribution parent-output
                                                   parent-value
                                                   parent-params)))
    (/ (- y effective-f) noise-var)))


(defmethod dlog-lik/doutputparam ((output gaussian-output) param f y-and-parent-value
                                  parent-outputs parent-params)
  (let ((y (first y-and-parent-value))
        (parent-values (rest y-and-parent-value))
        (noise-var (safe-exp (aref (output-params output) 0)))
        (effective-f f))
    (loop for parent-value in parent-values
          for parent-output in parent-outputs
          do (incf effective-f (child-contribution parent-output
                                                   parent-value
                                                   parent-params)))
    (* 0.5d0 (1- (/ (expt (- effective-f y) 2)
                    noise-var)))))


;; In the case where the y-and-parent-dists are jointly Gaussian, this can be
;; computed in closed form
(defmethod LL-single ((output gaussian-output) mean var
                      y-and-parent-dist n-parents parent-outputs parent-params)
  (when (typep y-and-parent-dist 'quad-set)
    (return-from LL-single (call-next-method)))
  
  (destructuring-bind (y-mean y-cov) y-and-parent-dist
    (with-facets ((y-mean-array (y-mean 'backing-array :direction :input)))
      (let ((diff mean)
            (noise-var (safe-exp (aref (output-params output) 0)))
            par-vec
            par-vec-target
            LL)
        ;; Mean part of expression
        (loop for y-mu across y-mean-array
              for i from -1
              do (if (equal i -1)
                     (incf diff (- y-mu))
                     (incf diff (* (aref (nth i parent-params) 0) y-mu))))
        (setf LL (+ (/ (expt diff 2)
                       noise-var)
                    (log noise-var)))

        ;; Variance parts of the expression
        (incf LL (/ var noise-var))

        (setf par-vec (make-array (1+ n-parents) :element-type 'double-float)
              (aref par-vec 0) -1d0)
        (loop for par-param in parent-params
              for i from 1
              do (setf (aref par-vec i) (aref par-param 0)))
        (setf par-vec (array-to-mat par-vec)
              par-vec-target (make-mat (1+ n-parents) :ctype :double))
        (gemv! 1d0 y-cov par-vec 0d0 par-vec-target
               :m (1+ n-parents) :n (1+ n-parents) :lda (1+ n-parents))
        (incf LL (/ (dot par-vec par-vec-target :n (1+ n-parents))
                    noise-var))

        (* LL -0.5d0)))))


(defmethod dLL/dmu+dvar ((output gaussian-output) mean var y-and-parent-dists
                         n-parents parent-outputs parent-params)

  ;; When performing quadrature, revert to base method
  (when (typep (first y-and-parent-dists) 'quad-set)
    (return-from dLL/dmu+dvar (call-next-method)))
  
  (flet ((parameterized-dLL-single (mi vi ypari)
           (dLL-single/dmu+dvar output mi vi ypari n-parents parent-params)))
    (loop for mi in mean
          for vi in var
          for ypari in y-and-parent-dists
          for (dLL/dmu dLL/dvar) = (parameterized-dLL-single mi vi ypari)
          collect dLL/dmu into dLL/dmu-all
          collect dLL/dvar into dLL/dvar-all
          finally (return (list dLL/dmu-all dLL/dvar-all)))))


(defgeneric dLL-single/dmu+dvar (output mean var
                                 y-and-parent-dist n-parents parent-params)
  (:method ((output gaussian-output) mean var
            y-and-parent-dist n-parents parent-params)
    (destructuring-bind (y-mean y-cov) y-and-parent-dist
      (declare (ignore y-cov))
      (with-facets ((y-mean-array (y-mean 'backing-array :direction :input)))
        (let ((diff mean)
              (noise-var (safe-exp (aref (output-params output) 0)))
              dLL/dmu
              dLL/dvar)
          ;; Mean part of expression
          (loop for y-mu across y-mean-array
                for i from -1
                do (if (equal i -1)
                       (incf diff (- y-mu))
                       (incf diff (* (aref (nth i parent-params) 0) y-mu))))
          (setf dLL/dmu (- (/ diff
                              noise-var))
                dLL/dvar (- (/ 0.5d0 noise-var)))
          (list dLL/dmu dLL/dvar))))))


(defmethod dLL-single/doutputparam ((output gaussian-output) param mean var
                                    y-and-parent-dist n-parents parent-outputs
                                    parent-params)
  ;; There should only be a single parameter
  (unless (equal param 0)
    (error "Invalid parameter index in output likelihood derivative."))

  (when (typep y-and-parent-dist 'quad-set)
    (return-from dLL-single/doutputparam (call-next-method)))

  (destructuring-bind (y-mean y-cov) y-and-parent-dist
    (with-facets ((y-mean-array (y-mean 'backing-array :direction :input)))
      (let ((diff mean)
            (noise-var (safe-exp (aref (output-params output) 0)))
            par-vec
            par-vec-target
            dLL)
        ;; Mean part of expression
        (loop for y-mu across y-mean-array
              for i from -1
              do (if (equal i -1)
                     (incf diff (- y-mu))
                     (incf diff (* (aref (nth i parent-params) 0) y-mu))))
        ;; diff^2 / noise-var^2 * d noise-var/dparam
        ;;     = diff^2 / noise-var
        ;; -1 / noise-var * d noise-var/dparam = -1 
        (setf dLL (1- (/ (expt diff 2)
                         noise-var)))

        ;; Variance parts of the expression
        (incf dLL (/ var noise-var))

        (setf par-vec (make-array (1+ n-parents) :element-type 'double-float)
              (aref par-vec 0) -1d0)
        (loop for par-param in parent-params
              for i from 1
              do (setf (aref par-vec i) (aref par-param 0)))
        (setf par-vec (array-to-mat par-vec)
              par-vec-target (make-mat (1+ n-parents) :ctype :double))
        (gemv! 1d0 y-cov par-vec 0d0 par-vec-target
               :m (1+ n-parents) :n (1+ n-parents) :lda (1+ n-parents))
        (incf dLL (/ (dot par-vec par-vec-target :n (1+ n-parents))
                     noise-var))

        (* dLL 0.5d0)))))


(defmethod dLL-single/dparentparam ((output gaussian-output) index parent param mean
                                    var y-and-parent-dist n-parents parent-outputs
                                    parent-params)

  (when (typep y-and-parent-dist 'quad-set)
    (return-from dLL-single/dparentparam (call-next-method)))
  
  (destructuring-bind (y-mean y-cov) y-and-parent-dist
    (with-facets ((y-mean-array (y-mean 'backing-array :direction :input)))
      (let ((diff mean)
            (noise-var (safe-exp (aref (output-params output) 0)))
            par-vec
            par-vec-target
            d-vec
            dLL)
        ;; Mean part of expression
        (loop for y-mu across y-mean-array
              for i from -1
              do (if (equal i -1)
                     (incf diff (- y-mu))
                     (incf diff (* (aref (nth i parent-params) 0) y-mu))))
        (setf dLL (/ (* diff (aref y-mean-array (1+ parent)))
                     noise-var))

        ;; Variance parts of the expression
        (setf par-vec (make-array (1+ n-parents) :element-type 'double-float)
              (aref par-vec 0) -1d0)
        (loop for par-param in parent-params
              for i from 1
              do (setf (aref par-vec i) (aref par-param 0)))
        (setf d-vec (make-array (1+ n-parents)
                                :element-type 'double-float
                                :initial-element 0d0)
              (aref d-vec (1+ parent)) 1d0)
        (setf par-vec (array-to-mat par-vec)
              d-vec (array-to-mat d-vec)
              par-vec-target (make-mat (1+ n-parents) :ctype :double))
        (gemv! 1d0 y-cov par-vec 0d0 par-vec-target
               :m (1+ n-parents) :n (1+ n-parents) :lda (1+ n-parents))
        (incf dLL (/ (dot d-vec par-vec-target :n (1+ n-parents))
                     noise-var))

        (- dLL)))))


(defmethod child-contribution ((output gaussian-output) y child-parent-params)
  (* y (aref child-parent-params 0)))


(defmethod dchild/dparent-param ((output gaussian-output) param y child-parent-params)
  (assert (equal param 0) (param))
  y)


(defmethod make-quadrature-basis ((output gaussian-output) mean var parent-vals
                                  n-parents parent-outputs parent-params)
  (let ((effective-var (+ var (safe-exp (aref (output-params output) 0))))
        (parent-base 0d0))

    ;; Value from parents to be added to latent f
    (loop for parent-val in parent-vals
          for parent-output in parent-outputs
          for parent-param in parent-params
          do (incf parent-base (child-contribution parent-output
                                                   parent-val
                                                   parent-param)))
    
    (loop for point in (quad-points output)
          for weight in (quad-weights output)
          for f = (+ parent-base mean (* (sqrt (* 2 effective-var)) point))
          collect (list f weight))))


(defun safe-exp (f)
  (cond
    ((> f 11.5d0)
     (exp 11.5d0))
    ((< f -11.5d0)
     (exp -11.5d0))
    (t
     (exp f))))
