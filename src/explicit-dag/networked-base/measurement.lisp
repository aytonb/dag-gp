(uiop:define-package #:dag-gp/explicit-dag/networked-base/measurement
    (:use #:cl
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/networked-base/base)
  (:export #:add-measurement))

(in-package #:dag-gp/explicit-dag/networked-base/measurement)


(defgeneric add-measurement (gp loc output y)
  (:documentation "Adds the observation `y` with output dimension `output` at location `loc` to the gp.")
  (:method ((gp networked-base) loc output y)
    (add-single-output-measurement (nth output (constituent-gps gp))
                                   loc
                                   y)

    (unless (member loc (constituent-locs gp) :test #'equalp)
      (setf (constituent-locs gp)
            (nconc (constituent-locs gp) (list loc))))

    (setf (gethash loc (constituent-obs gp))
          (sort (nconc (gethash loc (constituent-obs gp)) (list output)) #'<)

          (gethash loc (constituent-loc-obs gp))
          (acons output y (gethash loc (constituent-loc-obs gp))))))
