(uiop:define-package #:dag-gp/utils
    (:use #:cl
          #:dag-gp/utils/all)
  (:reexport #:dag-gp/utils/all)
  (:documentation "Utilities to support Gaussian Processes."))

(in-package #:dag-gp/utils)
