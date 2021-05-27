(uiop:define-package #:dag-gp/explicit-dag/linear-parent/gp
    (:use #:cl
          #:dag-gp/output
          #:dag-gp/explicit-dag/parent-dependent-base/all)
  (:export #:linear-parent-gp-base
           #:linear-parent-gp
           #:variational-linear-parent-gp
           #:variational-combined-output-linear-parent-gp

           #:var-parent-distributions
           #:a-mat
           #:n-parent-params
           #:parent-params

           #:true-ff-squared-dist
           #:true-ff-abs-dist
           #:true-Kff
           #:y-cov
           #:a-deriv-mats
           #:a-deriv-mats-updated-p))

(in-package #:dag-gp/explicit-dag/linear-parent/gp)


;; Since we will use EM, we define true observations, as well as distributions
;; over parents.
(defclass linear-parent-gp-base ()
  (
   ;; For now, I will assume this is a list of (mu, sigma) at all obs-locs
   (var-parent-distributions
    :initform nil
    :accessor var-parent-distributions)
   (a-mat
    :initform nil
    :accessor a-mat)

   ;; Parent scaling parameters
   (n-parent-parameters
    :initform nil
    :accessor n-parent-params)
   (parent-parameters
    :initform nil
    :accessor parent-params)))


(defclass linear-parent-gp (linear-parent-gp-base parent-dependent-gp)
  ((true-ff-squared-distances
    :initform nil
    :accessor true-ff-squared-dist)
   (true-ff-abs-distances
    :initform nil
    :accessor true-ff-abs-dist)
   (true-ff-covariance
    :initform nil
    :accessor true-Kff)
   (y-cov
    :initform nil
    :accessor y-cov
    :documentation "The covariance matrix of y - A y-par.")
   (a-deriv-mats
    :initform nil
    :accessor a-deriv-mats
    :documentation "Derivative of the a matrix with respect to a parent parameter.")
   (a-deriv-mats-updated-p
    :initform nil
    :accessor a-deriv-mats-updated-p
    :documentation "T iff a-deriv-mats are up to date and can be used.")))


(defclass variational-linear-parent-gp
    (linear-parent-gp-base variational-parent-dependent-gp)
  ())


(defclass variational-combined-output-linear-parent-gp
    (linear-parent-gp-base variational-combined-output-parent-dependent-gp)
  ())


(defmethod count-params ((gp linear-parent-gp-base))
  (let ((parent-param-count 0))
    (setf (n-parent-params gp) nil)
    (loop for parent-output in (parent-outputs gp) do
      (setf (n-parent-params gp)
            (nconc (n-parent-params gp)
                   (list (n-output-child-params parent-output))))
      (incf parent-param-count (n-output-child-params parent-output)))
    
    (+ (call-next-method) parent-param-count)))

(defmethod count-params ((gp variational-combined-output-linear-parent-gp))
  (let ((single-output-est (call-next-method))
        (increment (* (1- (n-gps (output gp)))
                      (apply #'+ (n-parent-params gp)))))
    ;(incf (n-parent-params gp) increment)
    (+ single-output-est increment)))


(defmethod initialize-specialized-parameters ((gp linear-parent-gp-base))
  (let ((offset 0)
        parent-param)
    (setf (parent-params gp) nil)
    (loop for n-parent-params in (reverse (n-parent-params gp)) do
      (incf offset n-parent-params)
      (setf parent-param (make-array n-parent-params
                                     :element-type 'double-float
                                     :displaced-to (param-vec gp)
                                     :displaced-index-offset (- (n-gp-params gp)
                                                                offset)))
      (dotimes (i n-parent-params)
        (if (default-parent-param gp)
            (setf (aref parent-param i) (default-parent-param gp))
            (setf (aref parent-param i) (1- (* 2 (random 1d0))))))
      (push parent-param (parent-params gp)))))

(defmethod initialize-specialized-parameters ((gp linear-parent-gp))
  (setf (a-deriv-mats gp) nil
        (a-deriv-mats-updated-p gp) nil)
  (call-next-method))

(defmethod initialize-specialized-parameters
    ((gp variational-combined-output-linear-parent-gp))
  (let ((offset 0)
        parent-param)
    (setf (parent-params gp) nil)
    (dotimes (gp-ind (n-gps (output gp)))

      (let ((gp-parent-params nil))
        (loop for n-parent-params in (reverse (n-parent-params gp)) do
          (incf offset n-parent-params)
          (setf parent-param (make-array n-parent-params
                                         :element-type 'double-float
                                         :displaced-to (param-vec gp)
                                         :displaced-index-offset (- (n-gp-params gp)
                                                                    offset)))
          (dotimes (i n-parent-params)
            (if (default-parent-param gp)
                (setf (aref parent-param i) (default-parent-param gp))
                (setf (aref parent-param i) (1- (* 2 (random 1d0))))))
          (push parent-param gp-parent-params))
        (push gp-parent-params (parent-params gp))))))
