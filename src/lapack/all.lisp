(uiop:define-package #:dag-gp/lapack/all
    (:use #:cl
          #:dag-gp/lapack/lapack
          #:dag-gp/lapack/lapack-functions)
  (:export #:potrf!
           #:zero-potrf!
           #:potri!
           #:symmetric-potri!
           #:potrs!
           #:gemv!
           #:ger!
           #:symm!
           #:fixed-dot))

(in-package #:dag-gp/lapack/all)
