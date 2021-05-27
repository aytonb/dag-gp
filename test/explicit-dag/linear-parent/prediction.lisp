(uiop:define-package #:dag-gp-test/explicit-dag/linear-parent/prediction
    (:use #:cl
          #:fiveam
          #:mgl-mat
          #:dag-gp/quadrature
          #:dag-gp/kernel
          #:dag-gp/output
          #:dag-gp/utils/distance
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/parent-dependent-base/base
          #:dag-gp/explicit-dag/parent-dependent-base/likelihood
          #:dag-gp/explicit-dag/linear-parent/all
          #:dag-gp/explicit-dag/linear-parent/gp
          #:dag-gp-test/explicit-dag/linear-parent/computation
          #:dag-gp-test/utils/all)
  (:export )
  (:documentation "Tests predictive capability."))

(in-package #:dag-gp-test/explicit-dag/linear-parent/prediction)


(def-suite explicit-dag/linear-parent/prediction)
(in-suite explicit-dag/linear-parent/prediction)


;; See variational_linear_parent_test.m
(test variational-predict
  (let ((gp (make-parameter-controlled-1d-variational-linear-parent-gp)))
    (destructuring-bind (mean var) (make-predictive-posteriors gp '((6)))
      (with-facets ((mean-array (mean 'backing-array :direction :input))
                    (var-array (var 'array :direction :input)))
        (is (approximately-equal (aref mean-array 0) 0.004927925829804d0 1d-8))
        (is (approximately-equal (aref var-array 0 0) 3.490072987630492d0 1d-8))))
    (destructuring-bind (mean var) (make-predictive-posteriors gp '((7)))
      (with-facets ((mean-array (mean 'backing-array :direction :input))
                    (var-array (var 'array :direction :input)))
        (is (approximately-equal (aref mean-array 0) 1.522759025637386d-4 1d-8))
        (is (approximately-equal (aref var-array 0 0) 3.490342699199990d0 1d-8)))))

  ;; Test prediction is not order dependent
  (let ((gp (make-parameter-controlled-1d-variational-linear-parent-gp)))
    (destructuring-bind (mean var) (make-predictive-posteriors gp '((7)))
      (with-facets ((mean-array (mean 'backing-array :direction :input))
                    (var-array (var 'array :direction :input)))
        (is (approximately-equal (aref mean-array 0) 1.522759025637386d-4 1d-8))
        (is (approximately-equal (aref var-array 0 0) 3.490342699199990d0 1d-8))))
    (destructuring-bind (mean var) (make-predictive-posteriors gp '((6)))
      (with-facets ((mean-array (mean 'backing-array :direction :input))
                    (var-array (var 'array :direction :input)))
        (is (approximately-equal (aref mean-array 0) 0.004927925829804d0 1d-8))
        (is (approximately-equal (aref var-array 0 0) 3.490072987630492d0 1d-8))))))
