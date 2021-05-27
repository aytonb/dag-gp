(uiop:define-package #:dag-gp/utils/distance
    (:use #:cl)
  (:export #:squared-distance
           #:euclidean-distance))

(in-package #:dag-gp/utils/distance)


(defun squared-distance (vec-1 vec-2)
  (when (and (not (listp vec-1))
             (not (listp vec-2)))
    (return-from squared-distance (expt (- vec-1 vec-2) 2)))
  (when (not (equal (list-length vec-1) (list-length vec-2)))
    (error "Vectors provided to squared-distance must be the same length."))
  (let ((distance 0))
    (loop for i below (list-length vec-1) do
      (setf distance (+ distance (expt (- (nth i vec-1) (nth i vec-2)) 2))))
    (return-from squared-distance distance)))


(defun euclidean-distance (vec-1 vec-2)
  (return-from euclidean-distance (expt (squared-distance vec-1 vec-2) 0.5)))
