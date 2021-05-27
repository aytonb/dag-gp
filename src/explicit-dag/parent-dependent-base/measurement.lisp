(uiop:define-package #:dag-gp/explicit-dag/parent-dependent-base/measurement
    (:use #:cl
          #:dag-gp/explicit-dag/parent-dependent-base/base)
  (:export #:add-single-output-measurement))

(in-package #:dag-gp/explicit-dag/parent-dependent-base/measurement)


(defgeneric add-single-output-measurement (gp loc y)
  (:documentation "Adds the observation `y` at location `loc` to the gp.")
  (:method ((gp parent-dependent-gp) loc y)
    ;; Add to the observation locations and observations.
    (setf (obs-locs gp)
          (nconc (obs-locs gp) (list loc)))
    (incf (n-obs gp))
    (setf (obs gp)
          (nconc (obs gp) (list y)))))
