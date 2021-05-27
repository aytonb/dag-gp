(uiop:define-package #:dag-gp/quadrature
    (:use #:cl
          #:dag-gp/quadrature/all)
  (:reexport #:dag-gp/quadrature/all)
  (:documentation "Quadrature over distributions produced by Gaussian Processes."))

(in-package #:dag-gp/quadrature)
