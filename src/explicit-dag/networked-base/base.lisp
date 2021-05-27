(uiop:define-package #:dag-gp/explicit-dag/networked-base/base
    (:use #:cl
          #:dag-gp/utils/distance
          #:dag-gp/kernel
          #:dag-gp/explicit-dag/parent-dependent-base/all)
  (:export #:networked-base
           #:variational-networked-base
           
           #:constituent-gps
           #:output-dim
           #:outputs
           #:constituent-locs
           #:constituent-obs
           #:constituent-loc-obs
           #:ref-dist-fns
           #:ref-parent-param
           #:factors
           #:n-latent-per-constituent
           #:networked-input-dim
           #:all-gaussian-p
           #:factor-ordering
           #:factor-ordering-up-to-date

           #:make-kernel-from-reference
           #:make-constituent-gps
           #:configure-for-factor))

(in-package #:dag-gp/explicit-dag/networked-base/base)


(defclass networked-base ()
  ((constituent-gps
    :initarg constituent-gps
    :initform nil
    :accessor constituent-gps
    :documentation "The individual GPs making up this networked base.")
   (output-dimension
    :initarg :output-dim
    :initform nil
    :accessor output-dim)

   (outputs
    :initarg :outputs
    :initform nil
    :accessor outputs)

   (constituent-locations
    :initarg :constituent-locs
    :initform nil
    :accessor constituent-locs)
   (constituent-obs
    :initform (make-hash-table :test #'equalp)
    :accessor constituent-obs)
   (constituent-loc-obs
    :initform (make-hash-table :test #'equalp)
    :accessor constituent-loc-obs)

   ;; Reference kernel
   (ref-distance-fns
    :initarg :ref-dist-fns
    :initform (list #'squared-distance)
    :accessor ref-dist-fns)
   (reference-kernel
    :initarg :ref-kernel
    :accessor ref-kernel
    :initform nil
    :documentation "This kernel will be applied to every variable.")
   (reference-parent-param
    :initarg :ref-parent-param
    :accessor ref-parent-param
    :initform nil)

   ;; Track factors
   (factors
    :initform nil
    :accessor factors
    :documentation "An a-list of factors, var -> parents.")
   (factor-ordering
    :initform nil
    :accessor factor-ordering
    :documentation "An ordered list of factors of parents to children.")
   (factor-ordering-up-to-date
    :initform nil
    :accessor factor-ordering-up-to-date
    :documentation "t iff the factor ordering is correct for the current DAG.")))


(defclass variational-networked-base (networked-base)
  ((n-latent-per-constituent
    :initarg :n-latent
    :accessor n-latent-per-constituent)
   (input-dimension
    :initarg :input-dim
    :accessor networked-input-dim)

   (all-gaussian-p
    :accessor all-gaussian-p)))


(defmethod initialize-instance :after ((gp networked-base) &key)
  (when (and (constituent-gps gp)
             (or (output-dim gp)
                 (ref-kernel gp)))
    (error "If constituent gps are specified, then ref-kernel and output-dim cannot be specified."))

  (unless (constituent-gps gp)
    (make-constituent-gps gp)))


(defgeneric make-kernel-from-reference (gp parents)
  (:documentation "Makes a new kernel from the reference.")
  (:method ((gp networked-base) parents)
    (copy-kernel (ref-kernel gp) :operate-on-parent nil)))


(defgeneric make-constituent-gps (gp)
  (:documentation "Makes the constituent gps of the appropriate form."))


;; Initialize with custom-obs-locs

;; Initialize-constituent-gp
(defgeneric initialize-constituent-gp (gp constituent-gp &key use-all-locs)
  (:documentation "Calls logic to initialize a constituent GP.")
  (:method ((gp networked-base) constituent-gp &key (use-all-locs t))
    (initialize-gp constituent-gp :use-all-locs use-all-locs))
  (:method ((gp variational-networked-base) constituent-gp &key (use-all-locs t))
    (initialize-gp constituent-gp :initialize-latent nil
                                  :use-all-locs use-all-locs)))


(defgeneric configure-for-factor (gp index parents &key use-all-locs)
  (:documentation "Configures the gp numbered index so that it will have the specified parents.")
  (:method ((gp networked-base) index parents &key (use-all-locs t))
    (let* ((index-gp (nth index (constituent-gps gp)))
           (factor-kernel
             (if (typep index-gp 'variational-combined-output-parent-dependent-gp)
                 (loop for i below (n-combined index-gp)
                       collect (make-kernel-from-reference gp nil))
                 (make-kernel-from-reference gp nil))))
      ;; Define the parent gps
      (setf (parent-gps index-gp) nil
            (parent-outputs index-gp) nil)
      (loop for parent in parents do
        (setf (parent-gps index-gp)
              (nconc (parent-gps index-gp)
                     (list (nth parent (constituent-gps gp))))
              (parent-outputs index-gp)
              (nconc (parent-outputs index-gp)
                     (list (output (nth parent (constituent-gps gp)))))))

      ;; Update the factors
      (setf (cdr (assoc index (factors gp))) parents
            (factor-ordering-up-to-date gp) nil)

      ;; Set the constituent gp kernel
      (setf (kernel index-gp) factor-kernel)

      ;; Initialize the gp
      (initialize-constituent-gp gp index-gp :use-all-locs use-all-locs))))
