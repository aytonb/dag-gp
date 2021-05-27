(uiop:define-package #:dag-gp/kernel
    (:use #:cl
          #:dag-gp/kernel/all)
  (:reexport #:dag-gp/kernel/all)
  (:documentation "Gaussian Process kernels."))

(in-package #:dag-gp/kernel)
