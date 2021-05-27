(uiop:define-package #:dag-gp/explicit-dag/all
    (:use #:cl
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/linear-parent/all
          #:dag-gp/explicit-dag/networked-base/all
          #:dag-gp/explicit-dag/dag-gp/all)
  (:export #:parent-scaled-gp
           #:train
           #:predict

           #:dag-gp
           #:parent-scaled-gps
           #:configure-for-factor
           #:add-measurement))

(in-package #:dag-gp/explicit-dag/all)
