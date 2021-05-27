(uiop:define-package #:dag-gp-test/explicit-dag/parent-scaled/construction
    (:use #:cl
          #:fiveam
          #:dag-gp/explicit-dag/parent-scaled/base
          #:dag-gp/explicit-dag/parent-scaled/measurement
          #:dag-gp/kernel
          #:dag-gp/utils/all)
  (:export #:make-uninitialized-1d-parent-scaled-gp)
  (:documentation "Tests that data containers are made and appropriately sized."))

(in-package #:dag-gp-test/explicit-dag/parent-scaled/construction)


(def-suite explicit-dag/parent-scaled/construction)
(in-suite explicit-dag/parent-scaled/construction)


(defun make-empty-1d-parent-scaled-gp ()
  (make-instance 'parent-scaled-gp
                 :kernel (make-instance 'rbf-kernel)))


(defun make-uninitialized-1d-parent-scaled-gp ()
  (let ((gp (make-empty-1d-parent-scaled-gp)))
    (add-single-output-measurement gp '(0) 1d0)
    (add-single-output-measurement gp '(2) 2d0)
    (add-single-output-measurement gp '(5) 0.5d0)
    gp))


(test empty-construction
  (let ((gp (make-empty-1d-parent-scaled-gp)))
    ;; obs, n-obs, and obs-locs should be empty
    (is (equal (obs gp) nil))
    (is (equal (n-obs gp) 0))
    (is (equal (obs-locs gp) nil))

    ;; n-gp-params should be 3
    (is (equal (n-gp-params gp) 3))))


(test uninitialized-construction
  (let ((gp (make-uninitialized-1d-parent-scaled-gp)))
    (is (equalp (obs-locs gp) '((0) (2) (5))))
    (is (equalp (obs gp) '(1d0 2d0 0.5d0)))
    (is (equal (n-obs gp) 3))))
