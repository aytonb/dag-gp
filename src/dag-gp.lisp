(uiop:define-package #:dag-gp/dag-gp
    (:nicknames #:dag-gp)
  (:use #:cl
        #:dag-gp/quadrature
        #:dag-gp/utils
        #:dag-gp/kernel
        #:dag-gp/output
        #:dag-gp/lapack
        #:dag-gp/explicit-dag
        #:dag-gp/experiment)
  (:export
   ;; dag-gp/kernel
   #:rbf-kernel
   #:rational-quadratic-kernel
   #:sum-kernel
   #:kern-params
   #:n-kern-params

   ;; dag-gp/output
   #:gaussian-output
   #:binary-output
   #:n-ary-output

   ;; dag-gp/explicit-dag
   #:dag-gp
   #:add-measurement
   #:configure-for-factor
   #:train
   #:predict)
  (:documentation "Package for exporting public symbols."))

(in-package #:dag-gp/dag-gp)
