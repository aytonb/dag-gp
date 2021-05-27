(uiop:define-package #:dag-gp/explicit-dag/linear-parent/covariance
    (:use #:cl
          #:mgl-mat
          #:dag-gp/output
          #:dag-gp/kernel
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/linear-parent/gp)
  (:export #:compute-predictive-covariances))

(in-package #:dag-gp/explicit-dag/linear-parent/covariance)


(defmethod compute-predictive-covariances ((gp linear-parent-gp) pred-locs
                                           &key (use-all-locs t))
  (let ((obs-locs (if (closed-downwards-p gp)
                      (true-obs-locs gp)
                      (obs-locs gp))))
    (compute-covariances gp :use-all-locs use-all-locs)

    (setf (pred-Kff gp) (evaluate-matrix (kernel gp)
                                         (pred-ff-squared-dist gp)
                                         (pred-ff-abs-dist gp)
                                         nil
                                         nil
                                         :add-jitter t))
    (let ((noise-var (exp (aref (output-params (output gp)) 0))))
      (loop for i below (list-length pred-locs) do
        (incf (aref (pred-Kff gp) i i) noise-var)))

    (setf (pred-Kff gp) (array-to-mat (pred-Kff gp) :ctype :double))
  
    (setf (pred-obs-Kff gp) (evaluate-matrix (kernel gp)
                                             (pred-obs-ff-squared-dist gp)
                                             (pred-obs-ff-abs-dist gp)
                                             nil
                                             nil
                                             :add-jitter nil))

    ;; If the prediction and observation are the same location, add noise and jitter
    (let ((noise-var (+ (exp (aref (output-params (output gp)) 0))
                        1d-5))
          (pos))
      (loop for pred-loc in pred-locs
            for pred-index from 0
            do (setf pos (position pred-loc obs-locs :test #'equalp))
               (when pos
                 (incf (aref (pred-obs-Kff gp) pred-index pos) noise-var))))
    

    (setf (pred-obs-Kff gp) (array-to-mat (pred-obs-Kff gp) :ctype :double))
  ;; (setf (true-Kff gp) (evaluate-matrix (kernel gp)
  ;;                                      (true-ff-squared-dist gp)
  ;;                                      (true-ff-abs-dist gp)
  ;;                                      nil
  ;;                                      nil
  ;;                                      :add-jitter t))
  
  ;; (let ((noise-var (exp (aref (output-params (output gp)) 0))))
  ;;   (loop for i below (n-true-obs gp) do
  ;;     (incf (aref (true-Kff gp) i i) noise-var)))
  ))
