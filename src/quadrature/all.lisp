(uiop:define-package #:dag-gp/quadrature/all
    (:use #:cl
          #:dag-gp/quadrature/quad-set)
  (:export #:quad-set
           #:nested
           #:flattened
           #:weights
           #:add-quad-variable
           #:accumulate-variables
           #:flatten
           #:marginalize
           #:run-on-quadrature
           #:copy-quad-set
           #:gauss-hermite-generator))

(in-package #:dag-gp/quadrature/all)
