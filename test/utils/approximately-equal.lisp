(uiop:define-package #:dag-gp-test/utils/approximately-equal
    (:use #:cl)
  (:export #:approximately-equal))

(in-package #:dag-gp-test/utils/approximately-equal)


(defun approximately-equal (x y tol)
  (<= (abs (- x y)) tol))
