(uiop:define-package #:dag-gp/lapack
    (:use #:cl
          #:dag-gp/lapack/all)
  (:reexport #:dag-gp/lapack/all)
  (:documentation "Lapack functions needed for Gaussian Processes."))

(in-package #:dag-gp/lapack)
