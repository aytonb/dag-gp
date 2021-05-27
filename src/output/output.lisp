(uiop:define-package #:dag-gp/output/output
    (:use #:cl
          #:dag-gp/quadrature)
  (:export #:output
           #:output-params
           #:n-parameters
           #:n-output-params
           #:n-child-parameters
           #:n-output-child-params
           #:quadrature-points
           #:quad-points
           #:quadrature-weights
           #:quad-weights
           #:initialize-output-params
           #:log-lik
           #:dlog-lik/df
           #:dlog-lik/doutputparam
           #:LL-single
           #:LL-single-known-f
           #:LL
           #:dLL/dmu+dvar
           #:dLL/df
           #:dLL/df-single
           #:dLL/df-single-known-f
           #:dLL/doutputparam
           #:dLL-single/doutputparam
           #:dLL/dparentparam
           #:dLL-single/dparentparam
           #:dLL-single/dparentparam-known-f
           #:child-contribution
           #:dchild/dparent-param
           #:make-quadrature-basis))

(in-package #:dag-gp/output/output)


(defclass output ()
  (;; (quadrature-points
   ;;  :initform (list -5.387480890011232862017
   ;;                  -4.603682449550744273078
   ;;                  -3.944764040115625210376
   ;;                  -3.347854567383216326915
   ;;                  -2.78880605842813048053
   ;;                  -2.254974002089275523082
   ;;                  -1.738537712116586206781
   ;;                  -1.234076215395323007886
   ;;                  -0.7374737285453943587056
   ;;                  -0.2453407083009012499038
   ;;                  0.2453407083009012499038
   ;;                  0.7374737285453943587056
   ;;                  1.234076215395323007886
   ;;                  1.738537712116586206781
   ;;                  2.254974002089275523082
   ;;                  2.78880605842813048053
   ;;                  3.347854567383216326915
   ;;                  3.944764040115625210376
   ;;                  4.603682449550744273078
   ;;                  5.387480890011232862017)
   ;;  :accessor quad-points)
   ;; (quadrature-weights
   ;;  :initform (list 1.257800702450072d-13
   ;;                  2.4820623055057025d-10
   ;;                  6.1274904435174d-8
   ;;                  4.402121308553768d-6
   ;;                  1.2882628336371142d-4
   ;;                  0.0018301030758370149d0
   ;;                  0.013997837472272182d0
   ;;                  0.061506373304376596d0
   ;;                  0.1617393381747132d0
   ;;                  0.26079306716474204d0
   ;;                  0.26079306716474204d0
   ;;                  0.1617393381747132d0
   ;;                  0.061506373304376596d0
   ;;                  0.013997837472272182d0
   ;;                  0.0018301030758370149d0
   ;;                  1.2882628336371142d-4
   ;;                  4.402121308553768d-6
   ;;                  6.1274904435174d-8
   ;;                  2.4820623055057025d-10
   ;;                  1.257800702450072d-13)
   ;;  :accessor quad-weights)
   ;; (n-quadrature
   ;;  :initform 20
   ;;  :accessor n-quad)
   (quadrature-points
    :initform (list -2.350604973674492222834d0
                    -1.335849074013696949715d0
                    -0.4360774119276165086792d0
                    0.436077411927616508679d0
                    1.335849074013696949715d0
                    2.350604973674492222834d0)
    :accessor quad-points)
   (quadrature-weights
    :initform (list 0.0025557844020562465d0
                    0.08861574604191454d0
                    0.40882846955602925d0
                    0.40882846955602925d0
                    0.08861574604191454d0
                    0.0025557844020562465d0)
    :accessor quad-weights)
   (n-quadrature
    :initform 6
    :accessor n-quad)
   (parameters
    :accessor output-params
    :documentation "A vector of parameters for this output.")
   (n-parameters
    :accessor n-output-params)
   (n-child-parameters
    :accessor n-output-child-params
    :documentation "The number of parameters needed by a child."))
  (:documentation "Outputs provide mappings between the latent GP and the observables."))


(defmethod initialize-instance :after ((output output) &key)
  (setf (output-params output) (make-array (n-output-params output)
                                           :element-type 'double-float
                                           :adjustable t)))


(defgeneric initialize-output-params (output)
  (:documentation "Sets the output parameters to their initial values.")
  (:method ((output output))
    nil))




;;;; Routines to be specialized by each output ;;;;  

(defgeneric log-lik (output f y-and-parent-value parent-outputs parent-params)
  (:documentation "Computes the log likelihood of the observation y given an output f."))


(defgeneric dlog-lik/df (output index f y-and-parent-value parent-outputs
                         parent-params)
  (:documentation "Computes the derivative of the log likelihood with respect to f_index."))


(defgeneric dlog-lik/doutputparam (output param f y-and-parent-value
                                   parent-outputs parent-params)
  (:documentation "Computes the derivative of the log likelihood with respect to the output parameter of index param."))




;; Log likelihood routines ;;;;

(defgeneric LL-single-known-f (output f y-and-parent-dist
                               parent-outputs parent-params)
  (:documentation "Computes the expected log likelihood with quadrature over y.")
  (:method ((output output) f y-and-parent-dist parent-outputs parent-params)
    (loop for y-and-parent-value in (flattened y-and-parent-dist)
          for weight in (weights y-and-parent-dist)
          sum (* weight (log-lik output f y-and-parent-value
                                 parent-outputs parent-params)))))


;; Gauss-Hermite quadrature is evaluated at points mu + sqrt(2 var)x
(defgeneric LL-single (output mean var y-and-parent-dist n-parents
                       parent-outputs parent-params)
  (:documentation "Computes the expected log likelihood of the observation y given a list of quadrature points.")
  (:method ((output output) mean var y-and-parent-dist n-parents
            parent-outputs parent-params)      
    (loop for weight in (quad-weights output)
          for quad-point in (quad-points output)
          for int-point = (+ mean (* (sqrt (* 2 var)) quad-point))
          sum (* weight (LL-single-known-f output int-point y-and-parent-dist
                                           parent-outputs parent-params)))))


;; We are computing E_{Y} [ E_{f} [p(y|f)]]
(defgeneric LL (output mean var y-and-parent-dists n-parents
                parent-outputs parent-params)
  (:documentation "Computes the log likelihood of all observations of this output.")
  (:method ((output output) mean var y-and-parent-dists n-parents
            parent-outputs parent-params)
    ;(format t "In LL with mean and var ~a ~a ~%~a~%" mean var y-and-parent-dists)
    (flet ((parameterized-LL-single (mi vi ypari)
             (LL-single output mi vi ypari n-parents parent-outputs parent-params)))
      (mapcar #'parameterized-LL-single mean var y-and-parent-dists))))




;;;; Derivatives by mean and variance ;;;;

(defgeneric dLL/dmu+dvar (output mean var y-and-parent-dists n-parents
                          parent-outputs parent-params)
  (:documentation "Computes the derivative of log likelihood with respect to the predictive mean and variance of the underlying Gaussian.")
  (:method ((output output) mean var y-and-parent-dists n-parents
            parent-outputs parent-params)
    ;; For quadrature based likelihoods, we compute the derivative with respect to
    ;; quadrature points, then use this to generate mu and var derivatives.
    (let ((dLL/df (dLL/df output mean var y-and-parent-dists n-parents
                          parent-outputs parent-params)))
      (flet ((add (x) (apply #'+ x))
             (parameterized-dLL/dvar-single (var-i dLL/df-i)
               (dLL/dvar-single output var-i dLL/df-i)))

        (list (mapcar #'add dLL/df) ;; dLL/dmu
              ;; dLL/dvar
              (mapcar #'parameterized-dLL/dvar-single var dLL/df))))))


(defgeneric dLL/dvar-single (output var dLL/df-single)
  (:documentation "Computes the derivative of the log-likelihood dLL/dvari.")
  (:method ((output output) var dLL/df-single)
    ;; dLL/dvar = dLL/df df/dvar = dLL/df x/sqrt(2 var)
    (flet ((var-deriv (quad-point dLL/df)
             (/ (* quad-point dLL/df)
                (sqrt (* 2 var)))))
      (let ((int-points (mapcar #'var-deriv
                                (quad-points output)
                                dLL/df-single)))
        (apply #'+ int-points)))))


(defgeneric dLL/df-single-known-f (output f y-and-parent-dist
                                   parent-outputs parent-params)
  (:documentation "Computes the derivative of the expected log likelihood with respect to f with quadrature over y.")
  (:method ((output output) f y-and-parent-dist parent-outputs parent-params)
    (loop for y-and-parent-value in (flattened y-and-parent-dist)
          for weight in (weights y-and-parent-dist)
          sum (* weight (dlog-lik/df output 0 f y-and-parent-value
                                     parent-outputs parent-params)))))


(defgeneric dLL/df-single (output mean var y-and-parent-dists n-parents
                           parent-outputs parent-params)
  (:documentation "Computes the derivative of the expected log likelihood with respect to f. (dLL/dfi1 dLL/dfi2 ...) where fih is f for observation i and Gauss-Hermite point h.")
  (:method ((output output) mean var y-and-parent-dists n-parents
            parent-outputs parent-params)
    (flet ((parameterized-dLL/df-single-known-f (f weight)
             (* weight (dLL/df-single-known-f output f y-and-parent-dists
                                              parent-outputs parent-params))))
      (let ((int-points (mapcar (lambda (x) (+ mean (* (sqrt (* 2 var)) x)))
                                (quad-points output))))
        (mapcar #'parameterized-dLL/df-single-known-f
                int-points
                (quad-weights output))))))


(defgeneric dLL/df (output mean var y-and-parent-dists n-parents
                    parent-outputs parent-params)
  (:documentation "Computes the derivative of log likelihood with respect to f for all observations. ((dLL/df11 dLL/df12 ...) (dLL/df21 dLL/df22 ...) ...) where fih is f for observation i and Gauss-Hermite point h.")
  (:method ((output output) mean var y-and-parent-dists n-parents
            parent-outputs parent-params)
    (flet ((parameterized-dLL/df-single (mi vi ypari)
             (dLL/df-single output mi vi ypari n-parents
                            parent-outputs parent-params)))
      (mapcar #'parameterized-dLL/df-single mean var y-and-parent-dists))))




;;;; Derivatives by output parameters ;;;;

(defgeneric dLL-single/doutputparam-known-f (output param f y-and-parent-dist
                                              parent-outputs parent-params)
  (:documentation "Computes the derivative of the expected log likelihood with respected to an output parameter with quadrature over f.")
  (:method ((output output) param f y-and-parent-dist parent-outputs parent-params)
    (loop for y-and-parent-value in (flattened y-and-parent-dist)
          for weight in (weights y-and-parent-dist)
          sum (* weight (dlog-lik/doutputparam output param f y-and-parent-value
                                               parent-outputs parent-params)))))


(defgeneric dLL-single/doutputparam (output param mean var y-and-parent-dist
                                     n-parents parent-outputs parent-params)
  (:documentation "Computes the derivative of the log likelihood of a single output observation.")
  (:method ((output output) param mean var y-and-parent-dist n-parents
            parent-outputs parent-params)      
    (loop for weight in (quad-weights output)
          for quad-point in (quad-points output)
          for int-point = (+ mean (* (sqrt (* 2 var)) quad-point))
          sum (* weight (dLL-single/doutputparam-known-f
                         output param int-point y-and-parent-dist
                         parent-outputs parent-params)))))


(defgeneric dLL/doutputparam (output param mean var y-and-parent-dists
                              n-parents parent-outputs parent-params)
  (:documentation "Computes the derivative of log likelihood with respect to the output parameters.")
  (:method ((output output) param mean var y-and-parent-dists
            n-parents parent-outputs parent-params)
    (flet ((parameterized-dLL-single (mi vi ypari)
             (dLL-single/doutputparam output param mi vi ypari n-parents
                                      parent-outputs parent-params)))
      (apply #'+ (mapcar #'parameterized-dLL-single mean var y-and-parent-dists)))))




;;;; Derivatives by parent parameters ;;;;

(defgeneric dLL-single/dparentparam-known-f (output index parent param f
                                             y-and-parent-dist parent-outputs
                                             parent-params)
  (:documentation "Computes the derivative of the expected log likelihood with respect to f with quadrature over y.")
  (:method ((output output) index parent param f y-and-parent-dist
            parent-outputs parent-params)
    (loop for y-and-parent-value in (flattened y-and-parent-dist)
          for weight in (weights y-and-parent-dist)
          sum (* weight
                 (dlog-lik/df output index f y-and-parent-value
                              parent-outputs parent-params)
                 (dchild/dparent-param (nth parent parent-outputs)
                                       param
                                       (nth (1+ parent) y-and-parent-value)
                                       (nth parent parent-params))))))


(defgeneric dLL-single/dparentparam (output index parent param mean var
                                     y-and-parent-dist n-parents parent-outputs
                                     parent-params)
  (:documentation "Computes the derivative of the log likelihood of a single output observation.")
  (:method ((output output) index parent param mean var y-and-parent-dist
            n-parents parent-outputs parent-params)
    (loop for weight in (quad-weights output)
          for quad-point in (quad-points output)
          for int-point = (+ mean (* (sqrt (* 2 var)) quad-point))
          sum (* weight (dLL-single/dparentparam-known-f
                         output index parent param int-point y-and-parent-dist
                         parent-outputs parent-params)))))


(defgeneric dLL/dparentparam (output index parent param mean var y-and-parent-dists
                              n-parents parent-outputs parent-params)
  (:documentation "Computes the derivative of log likelihood with respect to the parent parameters.")
  (:method ((output output) index parent param mean var y-and-parent-dists
            n-parents parent-outputs parent-params)
    (flet ((parameterized-dLL-single (mi vi ypari)
             (dLL-single/dparentparam output index parent param mi vi ypari n-parents
                                      parent-outputs parent-params)))
      (apply #'+ (mapcar #'parameterized-dLL-single mean var y-and-parent-dists)))))




(defgeneric child-contribution (output y child-parent-params)
  (:documentation "The effect on a child's value of f, given y and params."))


(defgeneric dchild/dparent-param (output param y child-parent-params)
  (:documentation "The derivative of the effect on a child's value of f."))


(defgeneric make-quadrature-basis (output mean var parent-vals
                                   n-parents parent-outputs parent-params)
  (:documentation "For given parent variables, compute a quadrature basis over output values."))
