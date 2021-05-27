(uiop:define-package #:dag-gp-test/explicit-dag/parent-scaled/prediction
    (:use #:cl
          #:fiveam
          #:mgl-mat
          #:dag-gp/explicit-dag/parent-scaled/base
          #:dag-gp/explicit-dag/parent-scaled/measurement
          #:dag-gp/explicit-dag/parent-scaled/distance-matrix
          #:dag-gp/explicit-dag/parent-scaled/covariance
          #:dag-gp/explicit-dag/parent-scaled/likelihood
          #:dag-gp/explicit-dag/parent-scaled/predict
          #:dag-gp-test/explicit-dag/parent-scaled/computation
          #:dag-gp-test/utils/all)
  (:export )
  (:documentation "Tests that prediction quantities are computed appropriately."))

(in-package #:dag-gp-test/explicit-dag/parent-scaled/prediction)


(def-suite explicit-dag/parent-scaled/prediction)
(in-suite explicit-dag/parent-scaled/prediction)


(test 2-parent-predict ()
  (let* ((gp (make-parameter-controlled-2-parent-scaled-gp))
         (deriv (make-array (n-gp-params gp) :element-type 'double-float)))
    (make-observed-distance-matrices gp)
    (compute-covariances gp)
    (make-measurement-mat gp)
    (LL-and-derivs gp deriv)

    (destructuring-bind (pred-mean pred-cov)
        (predict gp '((3) (6)))
      (is (typep pred-mean 'mat))
      (is (equalp (mat-dimensions pred-mean) (list 2)))
      (with-facets ((pred-mean-array (pred-mean 'array :direction :input)))
        (is (approximately-equal (aref pred-mean-array 0) 0.689413650373851d0 1d-5))
        (is (approximately-equal (aref pred-mean-array 1) 0.299215888690553d0 1d-5)))

      (is (typep pred-cov 'mat))
      (is (equalp (mat-dimensions pred-cov) (list 2 2)))
      (with-facets ((pred-cov-array (pred-cov 'array :direction :input)))
        (is (approximately-equal (aref pred-cov-array 0 0) 5.606641726744854d0 1d-5))
        (is (approximately-equal (aref pred-cov-array 0 1) -0.003216138674423d0 1d-5))
        (is (approximately-equal (aref pred-cov-array 1 0) -0.003216138674423d0 1d-5))
        (is (approximately-equal (aref pred-cov-array 1 1) 1.618827886261300d0 1d-5))))))
