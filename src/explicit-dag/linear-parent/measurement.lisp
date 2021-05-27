(uiop:define-package #:dag-gp/explicit-dag/linear-parent/measurement
    (:use #:cl
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/linear-parent/gp)
  (:export #:initialize-measurements))

(in-package #:dag-gp/explicit-dag/linear-parent/measurement)


(defmethod add-single-output-measurement ((gp linear-parent-gp-base) loc y)
  (setf (true-obs-locs gp)
        (nconc (true-obs-locs gp) (list loc)))
  (incf (n-true-obs gp))
  (setf (true-obs gp)
        (nconc (true-obs gp) (list y))))


(defgeneric initialize-measurements (gp all-locs)
  (:documentation "Prepares observation lists with the values at all locations that have observations.")

  (:method ((gp linear-parent-gp-base) all-locs)
    (let ((sorted-true-obs nil)
          (sorted-true-obs-locs nil))

    ;; (setf (obs-locs gp) (copy-list (true-obs-locs gp))
    ;;       (n-obs gp) (n-true-obs gp)
    ;;       (obs gp) (copy-list (true-obs gp)))

    ;; (loop for loc in all-locs do
    ;;   (unless (member loc (true-obs-locs gp) :test #'equalp)
    ;;     (setf (obs-locs gp)
    ;;           (nconc (obs-locs gp) (list loc)))
    ;;     (incf (n-obs gp))
    ;;     (setf (obs gp)
    ;;           (nconc (obs gp) (list :dist)))))

      (setf (obs-locs gp) nil
            (n-obs gp) 0
            (obs gp) nil)
      (loop for loc in all-locs do
        (if (member loc (true-obs-locs gp) :test #'equalp)
            (setf (obs gp)
                  (nconc (obs gp)
                         (list (nth (position loc (true-obs-locs gp) :test #'equalp)
                                    (true-obs gp)))))
            (setf (obs gp) (nconc (obs gp) (list :dist))))
        (setf (obs-locs gp) (nconc (obs-locs gp) (list loc)))
        (incf (n-obs gp)))

      ;; TODO: Only needs to be done once, prior to training
      (when (closed-downwards-p gp)
        (loop for loc in all-locs do
          (when (member loc (true-obs-locs gp) :test #'equalp)
            (setf sorted-true-obs
                  (nconc sorted-true-obs
                         (list (nth (position loc (true-obs-locs gp) :test #'equalp)
                                    (true-obs gp))))
                  sorted-true-obs-locs
                  (nconc sorted-true-obs-locs
                         (list loc)))))
        
        (setf (true-obs gp) sorted-true-obs
              (true-obs-locs gp) sorted-true-obs-locs)))

  ;; (:method ((gp linear-parent-gp) all-locs)
  ;;   (call-next-method)

  ;;   (setf (transform-indices gp) nil)
  ;;   (loop for loc in (obs-locs gp) do
  ;;     (setf (transform-indices gp)
  ;;           (nconc (transform-indices gp)
  ;;                  (list (position loc all-locs :test #'equalp)))))
  ;;   (setf (inverse-transform-indices gp) nil)
  ;;   (loop for loc in all-locs do
  ;;     (setf (inverse-transform-indices gp)
  ;;           (nconc (inverse-transform-indices gp)
  ;;                  (list (position loc (obs-locs gp) :test #'equalp))))))
  ))
