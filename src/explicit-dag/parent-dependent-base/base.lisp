(uiop:define-package #:dag-gp/explicit-dag/parent-dependent-base/base
    (:use #:cl
          #:dag-gp/utils/distance
          #:dag-gp/kernel
          #:dag-gp/output)
  (:export #:parent-dependent-gp
           #:variational-parent-dependent-gp
           #:variational-combined-output-parent-dependent-gp

           #:input-dim
           
           #:obs-locs
           #:obs
           #:obs-mat
           #:obs-mat-copy
           #:n-obs
           
           #:param-vec
           #:n-gp-params

           #:output

           #:n-latent
           #:u-locs
           #:q-mean-params
           #:q-mean
           #:q-mean-copy
           #:q-chol
           #:q-chol-mat
           #:q-cov
           #:middle
           #:qf-mean
           #:reshaped-qf-mean
           #:dLL/dqf-mu
           #:dLL/dqf-var

           #:parent-gps
           #:parent-outputs
           #:parent-dist-fns
           #:parent-squared-dist
           #:parent-abs-dist

           #:child-obs-locs
           #:child-obs
           
           #:dist-fns
           #:ff-squared-dist
           #:ff-abs-dist
           #:fu-squared-dist
           #:fu-abs-dist
           #:uu-squared-dist
           #:uu-abs-dist

           #:pred-parent-squared-dist
           #:pred-parent-abs-dist
           #:pred-obs-parent-squared-dist
           #:pred-obs-parent-abs-dist
           #:pred-ff-squared-dist
           #:pred-ff-abs-dist
           #:pred-obs-ff-squared-dist
           #:pred-obs-ff-abs-dist
           #:pred-fu-squared-dist
           #:pred-fu-abs-dist

           #:kernel
           #:Kff
           #:reshaped-Kff
           #:Kff-inv
           #:Kfu
           #:KfuM
           #:Kuu
           #:Kuu-chol
           #:Kuu-inv

           #:pred-Kff
           #:pred-obs-Kff
           #:pred-Kfu

           #:default-parent-param

           #:n-combined

           #:closed-downwards-p
           #:true-obs-locs
           #:true-obs
           #:n-true-obs

           #:update-parameter-vector
           #:count-params
           #:initialize-specialized-parameters))

(in-package #:dag-gp/explicit-dag/parent-dependent-base/base)


(defclass parent-dependent-gp ()
  ;; Parents
  ((parent-gps
    :initform nil
    :initarg :parent-gps
    :accessor parent-gps)
   (parent-outputs
    :initform nil
    :accessor parent-outputs)

   ;; Raw outputs
   (observation-locations
    :initform nil
    :accessor obs-locs)
   (observations
    :initform nil
    :accessor obs)
   (observations-mat
    :initform nil
    :accessor obs-mat)
   (observations-mat-copy
    :initform nil
    :accessor obs-mat-copy)
   (n-observations
    :initform 0
    :accessor n-obs)

   ;; Output
   (output
    :initarg :output
    :initform nil
    :accessor output)

   ;; Distances between inputs
   (distance-functions
    :initarg :dist-fns
    :accessor dist-fns
    :initform (list #'squared-distance))
   (ff-squared-distances
    :accessor ff-squared-dist
    :initform nil)
   (ff-absolute-distances
    :accessor ff-abs-dist
    :initform nil)
   
   ;; Predictive distances
   (predictive-ff-squared-distances
    :accessor pred-ff-squared-dist
    :initform nil)
   (predictive-ff-absolute-distances
    :accessor pred-ff-abs-dist
    :initform nil)
   (predictive-observed-ff-squared-distances
    :accessor pred-obs-ff-squared-dist
    :initform nil)
   (predictive-observed-ff-absolute-distances
    :accessor pred-obs-ff-abs-dist
    :initform nil)
   

   ;; Kernels and covariances
   (kernel
    :initarg :kernel
    :accessor kernel)
   (ff-covariance
    :initform nil
    :accessor Kff
    :documentation "The covariance Kff.")
   (ff-covariance-inverse
    :initform nil
    :accessor Kff-inv
    :documentation "The matrix Kff^-1.")

   (likelihood-var-deriv
    :initform nil
    :accessor dLL/dqf-var)

   ;; Predictive covariances
   (predictive-ff-covariance
    :initform nil
    :accessor pred-Kff
    :documentation "The predictive covariance Kff.")
   (predictive-observed-ff-covariance
    :initform nil
    :accessor pred-obs-Kff
    :documentation "Covariance between predicted and observed locations.")

   (default-parent-param
    :initarg :default-parent-param
    :initform nil
    :accessor default-parent-param)

   ;; Parameter vectors
   (parameter-indices
    :accessor param-indices
    :documentation "Indices of various parameters in the parameter vector.")
   (n-gp-parameters
    :accessor n-gp-params
    :documentation "Number of total parameters in the Gaussian Process.")
   (parameter-vector
    :accessor param-vec
    :documentation "A single vector of all parameters.")

   (closed-downwards-p
    :accessor closed-downwards-p
    :initarg :closed-downwards-p
    :initform nil)
   (true-observation-locations
    :initform nil
    :accessor true-obs-locs)
   (true-observations
    :initform nil
    :accessor true-obs)
   (n-true-observations
    :initform 0
    :accessor n-true-obs)))


(defclass variational-parent-dependent-gp (parent-dependent-gp)
  ((input-dimension
    :initarg :input-dim
    :accessor input-dim)

   ;; Latent variables
   (n-latent
    :initarg :n-latent
    :accessor n-latent
    :documentation "The number of latent locations u.")
   (latent-locations
    :initform nil
    :accessor u-locs
    :documentation "The latent variable locations.")

   (posterior-mean-params
    :initform nil
    :accessor q-mean-params)
   (posterior-mean
    :initform nil
    :accessor q-mean)
   (posterior-mean-copy
    :initform nil
    :accessor q-mean-copy)
   (posterior-cholesky
    :initform nil
    :accessor q-chol)
   (posterior-cholesky-mat
    :initform nil
    :accessor q-chol-mat)
   (posterior-covariances
    :initform nil
    :accessor q-cov)
   (middle
    :initform nil
    :accessor middle
    :documentation "The middle matrices Su - Kuu.")
   (posterior-f-mean
    :initform nil
    :accessor qf-mean)
   (likelihood-mean-deriv
    :initform nil
    :accessor dLL/dqf-mu)

   ;; Distances between inputs
   ;; (distance-functions
   ;;  :initarg :dist-fns
   ;;  :accessor dist-fns
   ;;  :initform (list #'squared-distance))
   (fu-squared-distances
    :accessor fu-squared-dist
    :initform nil)
   (fu-absolute-distances
    :accessor fu-abs-dist
    :initform nil)
   (uu-squared-distances
    :accessor uu-squared-dist
    :initform nil)
   (uu-absolute-distances
    :accessor uu-abs-dist
    :initform nil)

   ;; Predictive distances
   (predictive-fu-squared-distances
    :accessor pred-fu-squared-dist
    :initform nil)
   (predictive-fu-absolute-distances
    :accessor pred-fu-abs-dist
    :initform nil)


   ;; Covariances
   (fu-covariance
    :initform nil
    :accessor Kfu
    :documentation "The covariance Kfu.")
   (fu-covariance-mid
    :accessor KfuM
    :documentation "The product Kfu Kuu^-1 (q-cov - Kuu).")
   (uu-covariance
    :initform nil
    :accessor Kuu
    :documentation "The covariance Kuu.")
   (uu-covariance-cholesky
    :initform nil
    :accessor Kuu-chol
    :documentation "chol(Kuu)")
   (uu-covariance-inverse
    :initform nil
    :accessor Kuu-inv
    :documentation "Kuu^-1")

   ;; Predictive covariances
   (predictive-fu-covariance
    :initform nil
    :accessor pred-Kfu
    :documentation "The predictive covariance Kfu.")))


(defclass variational-combined-output-parent-dependent-gp
    (variational-parent-dependent-gp)
  ((n-combined
    :initarg :n-combined
    :accessor n-combined
    :documentation "The number of independent GPs used to make the output.")

   (reshaped-posterior-f-mean
    :initform nil
    :accessor reshaped-qf-mean
    :documentation "Posterior f mean reshaped to be ((loc0ind0 loc0ind1 ...) (loc1ind0 loc1ind1 ...)).")
   (reshaped-ff-covariance
    :initform nil
    :accessor reshaped-Kff
    :documentation "Kff reshaped to be ((loc0ind0 loc0ind1 ...) (loc1ind0 loc1ind1 ...)).")))


;; On creation, if parents are specified, collect outputs
(defmethod initialize-instance :after ((gp parent-dependent-gp) &key)
  (when (parent-gps gp)
    (setf (parent-outputs gp) (loop for parent-gp in (parent-gps gp)
                                    collect (output parent-gp)))))


(defmethod initialize-instance :after
    ((gp variational-combined-output-parent-dependent-gp) &key)
  (setf (n-combined gp) (n-gps (output gp))))


(defgeneric count-params (gp)
  (:documentation "Gives the number of parameters in the GP.")

  (:method ((gp parent-dependent-gp))
    (+ (n-kern-params (kernel gp))
       (n-output-params (output gp))))

  (:method ((gp variational-parent-dependent-gp))
    (+ (call-next-method)
       (n-latent gp)
       (/ (* (n-latent gp) (1+ (n-latent gp))) 2)))

  (:method ((gp variational-combined-output-parent-dependent-gp))
    (+ (loop for kern in (kernel gp)
             sum (n-kern-params kern))
       (n-output-params (output gp))
       (* (n-combined gp)
          (+ (n-latent gp)
             (/ (* (n-latent gp) (1+ (n-latent gp))) 2))))))


(defgeneric initialize-specialized-parameters (gp)
  (:documentation "Initializes any parameters that are only for this GP subclass.")
  (:method ((gp parent-dependent-gp))
    nil))


(defgeneric update-parameter-vector (gp new-params)
  (:documentation "Updates parameters in the GP.")
  (:method ((gp parent-dependent-gp) new-params)
    (adjust-array (param-vec gp) (n-gp-params gp) :displaced-to new-params)))
