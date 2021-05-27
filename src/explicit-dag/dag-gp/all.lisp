(uiop:define-package #:dag-gp/explicit-dag/dag-gp/all
    (:use #:cl
          #:dag-gp/explicit-dag/dag-gp/gp
          #:dag-gp/explicit-dag/dag-gp/predict
          #:dag-gp/explicit-dag/dag-gp/likelihood)
  (:export #:dag-gp
           #:variational-dag-gp

           #:update-combined-distributions
           #:predict-combined))

(in-package #:dag-gp/explicit-dag/dag-gp/all)
