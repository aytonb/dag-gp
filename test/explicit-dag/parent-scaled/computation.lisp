(uiop:define-package #:dag-gp-test/explicit-dag/parent-scaled/computation
    (:use #:cl
          #:fiveam
          #:mgl-mat
          #:dag-gp/explicit-dag/parent-scaled/base
          #:dag-gp/explicit-dag/parent-scaled/measurement
          #:dag-gp/explicit-dag/parent-scaled/distance-matrix
          #:dag-gp/explicit-dag/parent-scaled/covariance
          #:dag-gp/explicit-dag/parent-scaled/likelihood
          #:dag-gp/kernel
          #:dag-gp/utils/distance
          #:dag-gp-test/explicit-dag/parent-scaled/construction
          #:dag-gp-test/utils/all)
  (:export #:make-parameter-controlled-1d-parent-scaled-gp
           #:make-parameter-controlled-2-parent-scaled-gp)
  (:documentation "Tests that inference quantities are computed appropriately."))

(in-package #:dag-gp-test/explicit-dag/parent-scaled/computation)


(def-suite explicit-dag/parent-scaled/computation)
(in-suite explicit-dag/parent-scaled/computation)


(defun make-parameter-controlled-1d-parent-scaled-gp ()
  (let ((gp (make-uninitialized-1d-parent-scaled-gp)))
    ;; Noise param
    (setf (aref (param-vec gp) 0) -4.0d0)

    ;; Kernel parameters
    (setf (subseq (kern-params (kernel gp)) 0 2)
          (list 0.5d0 2.0d0))

    gp))


(test distance-matrices
  (let ((gp (make-parameter-controlled-1d-parent-scaled-gp)))
    (make-observed-distance-matrices gp)

    ;; ff distances
    (is (listp (ff-squared-dist gp)))
    (is (equal (list-length (ff-squared-dist gp)) 1))
    (is (arrayp (car (ff-squared-dist gp))))
    (is (equalp (array-dimensions (car (ff-squared-dist gp))) (list 3 3)))
    (is (equal (aref (car (ff-squared-dist gp)) 0 0) 0d0))
    (is (equal (aref (car (ff-squared-dist gp)) 0 1) 4d0))
    (is (equal (aref (car (ff-squared-dist gp)) 0 2) 25d0))
    (is (equal (aref (car (ff-squared-dist gp)) 1 0) 4d0))
    (is (equal (aref (car (ff-squared-dist gp)) 1 1) 0d0))
    (is (equal (aref (car (ff-squared-dist gp)) 1 2) 9d0))
    (is (equal (aref (car (ff-squared-dist gp)) 2 0) 25d0))
    (is (equal (aref (car (ff-squared-dist gp)) 2 1) 9d0))
    (is (equal (aref (car (ff-squared-dist gp)) 2 2) 0d0))

    ;; Parent distances
    (is (equal (parent-squared-dist gp) nil))))


(test covariances
  (let ((gp (make-parameter-controlled-1d-parent-scaled-gp)))
    (make-observed-distance-matrices gp)
    (compute-covariances gp)

    (is (typep (Kff gp) 'mat))
    (with-facets ((Kff ((Kff gp) 'array :direction :input)))
      (is (approximately-equal (aref Kff 0 0) 1.667046909588862d0 1d-5))
      (is (approximately-equal (aref Kff 0 1) 0.959501756450728d0 1d-5))
      (is (approximately-equal (aref Kff 0 2) 0.055945231210373d0 1d-5))
      (is (approximately-equal (aref Kff 1 0) 0.959501756450728d0 1d-5))
      (is (approximately-equal (aref Kff 1 1) 1.667046909588862d0 1d-5))
      (is (approximately-equal (aref Kff 1 2) 0.487718175521200d0 1d-5))
      (is (approximately-equal (aref Kff 2 0) 0.055945231210373d0 1d-5))
      (is (approximately-equal (aref Kff 2 1) 0.487718175521200d0 1d-5))
      (is (approximately-equal (aref Kff 2 2) 1.667046909588862d0 1d-5)))))


(test update-parameter-vector
  (let ((gp (make-uninitialized-1d-parent-scaled-gp))
        (params-1 (make-array 3 :element-type 'double-float
                                :initial-contents (list -4.0d0 0.5d0 2.0d0)))
        (deriv (make-array 3 :element-type 'double-float)))
    (make-observed-distance-matrices gp)
    (compute-covariances gp)
    (make-measurement-mat gp)

    (update-parameter-vector gp params-1)
    (is (approximately-equal (LL-and-derivs gp deriv) 1.720044034075352d0 1d-5))
    (is (approximately-equal (aref deriv 0) 0.007932819548154d0 1d-5))
    (is (approximately-equal (aref deriv 1) 0.277573296409508d0 1d-5))
    (is (approximately-equal (aref deriv 2) -0.273902900804248d0 1d-5))))


(defun quick-test ()
  (let ((gp (make-uninitialized-1d-parent-scaled-gp)))
    (train gp :progress-fn :verbose)))


(defun make-parameter-controlled-2-parent-scaled-gp ()
  (let* ((parent-gp-1 (make-instance 'parent-scaled-gp
                                     :kernel (make-instance 'rbf-kernel)))
         (parent-gp-2 (make-instance 'parent-scaled-gp
                                     :kernel (make-instance 'rbf-kernel)))
         (kernel (make-instance 'sum-kernel
                                :constituents (list (make-instance 'rbf-kernel)
                                                    (make-instance 'rbf-kernel
                                                                   :operate-on-parent 0)
                                                    (make-instance 'rbf-kernel
                                                                   :operate-on-parent 1))))
         (gp (make-instance 'parent-scaled-gp
                            :kernel kernel
                            :parent-gps (list parent-gp-1
                                              parent-gp-2)
                            :parent-dist-fns (list (list #'squared-distance)
                                                   (list #'squared-distance)))))

    (add-single-output-measurement gp '(0) 1d0)
    (add-single-output-measurement gp '(2) 2d0)
    (add-single-output-measurement gp '(5) 0.5d0)

    (setf (child-obs parent-gp-1) (make-hash-table :test #'equalp)
          (gethash '(0) (child-obs parent-gp-1)) 3d0
          (gethash '(2) (child-obs parent-gp-1)) 5d0
          (gethash '(5) (child-obs parent-gp-1)) 2d0
          (gethash '(3) (child-obs parent-gp-1)) 1d0
          (gethash '(6) (child-obs parent-gp-1)) 2d0
          (child-obs parent-gp-2) (make-hash-table :test #'equalp)
          (gethash '(0) (child-obs parent-gp-2)) -1d0
          (gethash '(2) (child-obs parent-gp-2)) 4d0
          (gethash '(5) (child-obs parent-gp-2)) 2.5d0
          (gethash '(3) (child-obs parent-gp-2)) 5d0
          (gethash '(6) (child-obs parent-gp-2)) 1d0)

    ;; Noise param
    (setf (aref (param-vec gp) 0) -4.0d0)

    ;; Kernel parameters
    (setf (subseq (kern-params (kernel gp)) 0 6)
          (list 0.5d0 2.0d0 1.5d0 0.3d0 0.4d0 1.2d0))
    
    gp))


(test 2-parent-distance-matrices
  (let ((gp (make-parameter-controlled-2-parent-scaled-gp)))
    (make-observed-distance-matrices gp)

    ;; ff distances
    (is (listp (ff-squared-dist gp)))
    (is (equal (list-length (ff-squared-dist gp)) 1))
    (is (arrayp (car (ff-squared-dist gp))))
    (is (equalp (array-dimensions (car (ff-squared-dist gp))) (list 3 3)))
    (is (equal (aref (car (ff-squared-dist gp)) 0 0) 0d0))
    (is (equal (aref (car (ff-squared-dist gp)) 0 1) 4d0))
    (is (equal (aref (car (ff-squared-dist gp)) 0 2) 25d0))
    (is (equal (aref (car (ff-squared-dist gp)) 1 0) 4d0))
    (is (equal (aref (car (ff-squared-dist gp)) 1 1) 0d0))
    (is (equal (aref (car (ff-squared-dist gp)) 1 2) 9d0))
    (is (equal (aref (car (ff-squared-dist gp)) 2 0) 25d0))
    (is (equal (aref (car (ff-squared-dist gp)) 2 1) 9d0))
    (is (equal (aref (car (ff-squared-dist gp)) 2 2) 0d0))

    ;; Parent distances
    (is (listp (parent-squared-dist gp)))
    (is (equal (list-length (parent-squared-dist gp)) 2))
    (is (listp (first (parent-squared-dist gp))))
    (is (equal (list-length (first (parent-squared-dist gp))) 1))
    (is (arrayp (car (first (parent-squared-dist gp)))))
    (is (equalp (array-dimensions (car (first (parent-squared-dist gp)))) (list 3 3)))
    (is (equal (aref (car (first (parent-squared-dist gp))) 0 0) 0d0))
    (is (equal (aref (car (first (parent-squared-dist gp))) 0 1) 4d0))
    (is (equal (aref (car (first (parent-squared-dist gp))) 0 2) 1d0))
    (is (equal (aref (car (first (parent-squared-dist gp))) 1 0) 4d0))
    (is (equal (aref (car (first (parent-squared-dist gp))) 1 1) 0d0))
    (is (equal (aref (car (first (parent-squared-dist gp))) 1 2) 9d0))
    (is (equal (aref (car (first (parent-squared-dist gp))) 2 0) 1d0))
    (is (equal (aref (car (first (parent-squared-dist gp))) 2 1) 9d0))
    (is (equal (aref (car (first (parent-squared-dist gp))) 2 2) 0d0))
    (is (listp (second (parent-squared-dist gp))))
    (is (equal (list-length (second (parent-squared-dist gp))) 1))
    (is (arrayp (car (second (parent-squared-dist gp)))))
    (is (equalp (array-dimensions (car (second (parent-squared-dist gp)))) (list 3 3)))
    (is (equal (aref (car (second (parent-squared-dist gp))) 0 0) 0d0))
    (is (equal (aref (car (second (parent-squared-dist gp))) 0 1) 25d0))
    (is (equal (aref (car (second (parent-squared-dist gp))) 0 2) 12.25d0))
    (is (equal (aref (car (second (parent-squared-dist gp))) 1 0) 25d0))
    (is (equal (aref (car (second (parent-squared-dist gp))) 1 1) 0d0))
    (is (equal (aref (car (second (parent-squared-dist gp))) 1 2) 2.25d0))
    (is (equal (aref (car (second (parent-squared-dist gp))) 2 0) 12.25d0))
    (is (equal (aref (car (second (parent-squared-dist gp))) 2 1) 2.25d0))
    (is (equal (aref (car (second (parent-squared-dist gp))) 2 2) 0d0))))


(test 2-parent-covariances
  (let ((gp (make-parameter-controlled-2-parent-scaled-gp)))
    (make-observed-distance-matrices gp)
    (compute-covariances gp)

    (is (typep (Kff gp) 'mat))
    (with-facets ((Kff ((Kff gp) 'array :direction :input)))
      (is (approximately-equal (aref Kff 0 0) 7.640560677568197d0 1d-5))
      (is (approximately-equal (aref Kff 0 1) 1.191780026824015d0 1d-5))
      (is (approximately-equal (aref Kff 0 2) 2.229740255020798d0 1d-5))
      (is (approximately-equal (aref Kff 1 0) 1.191780026824015d0 1d-5))
      (is (approximately-equal (aref Kff 1 1) 7.640560677568197d0 1d-5))
      (is (approximately-equal (aref Kff 1 2) 1.250951658801889d0 1d-5))
      (is (approximately-equal (aref Kff 2 0) 2.229740255020798d0 1d-5))
      (is (approximately-equal (aref Kff 2 1) 1.250951658801889d0 1d-5))
      (is (approximately-equal (aref Kff 2 2) 7.640560677568197d0 1d-5)))))


(test 2-parent-update-parameter-vector
  (let ((gp (make-parameter-controlled-2-parent-scaled-gp))
        (params-1 (make-array 7 :element-type 'double-float
                                :initial-contents (list -4.0d0 0.5d0 2.0d0
                                                        1.5d0 0.3d0 0.4d0 1.2d0)))
        (deriv (make-array 7 :element-type 'double-float)))
    (make-observed-distance-matrices gp)
    (compute-covariances gp)
    (make-measurement-mat gp)

    (update-parameter-vector gp params-1)
    (is (approximately-equal (LL-and-derivs gp deriv) 3.279026145023809d0 1d-5))
    (is (approximately-equal (aref deriv 0) 0.003265956943049d0 1d-5))
    (is (approximately-equal (aref deriv 1) 0.245925045597065d0 1d-5))
    (is (approximately-equal (aref deriv 2) -0.037925311148350d0 1d-5))
    (is (approximately-equal (aref deriv 3) 0.705688132393654d0 1d-5))
    (is (approximately-equal (aref deriv 4) -0.089943888014823d0 1d-5))
    (is (approximately-equal (aref deriv 5) 0.251589582111419d0 1d-5))
    (is (approximately-equal (aref deriv 6) -0.014420166263814d0 1d-5))))


