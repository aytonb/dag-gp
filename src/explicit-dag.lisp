(uiop:define-package #:dag-gp/explicit-dag
    (:use #:cl
          #:dag-gp/explicit-dag/all)
  (:reexport #:dag-gp/explicit-dag/all)
  (:documentation "Gaussian Processes with explicit DAG structure."))

(in-package #:dag-gp/explicit-dag)
