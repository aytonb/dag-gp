(uiop:define-package #:dag-gp/kernel/all
    (:use #:cl
          #:dag-gp/kernel/kernel
          #:dag-gp/kernel/rbf-kernel
          #:dag-gp/kernel/rational-quadratic-kernel
          #:dag-gp/kernel/sum-kernel)
  (:export
   ;; Kernel slots
   #:kern-params
   #:n-kern-params
   #:dist-index

   ;; Kernel generic functions
   #:evaluate
   #:evaluate-matrix
   #:dk/dparam
   #:dk/dparam-matrix
   #:initialize-kernel-params
   #:copy-kernel

   ;; Specific kernels
   #:kernel
   #:rbf-kernel
   #:rational-quadratic-kernel
   #:sum-kernel))

(in-package #:dag-gp/kernel/all)
