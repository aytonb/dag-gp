(uiop:define-package #:dag-gp/explicit-dag/linear-parent/distance-matrix
    (:use #:cl
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/linear-parent/gp)
  (:import-from #:array-operations)
  (:export #:make-predictive-distance-matrices))

(in-package #:dag-gp/explicit-dag/linear-parent/distance-matrix)


(defmethod make-predictive-distance-matrices ((gp linear-parent-gp) pred-locs n-pred)
  (setf (true-ff-squared-dist gp) nil
        (true-ff-abs-dist gp) nil
        (pred-ff-squared-dist gp) nil
        (pred-ff-abs-dist gp) nil
        (pred-obs-ff-squared-dist gp) nil
        (pred-obs-ff-abs-dist gp) nil)

  (loop for dist-fn in (dist-fns gp) do
    (let ((true-sq-d (make-array (list (n-true-obs gp) (n-true-obs gp))
                                 :element-type 'double-float))
          (pred-sq-d (make-array (list n-pred n-pred)
                                 :element-type 'double-float))
          (pred-obs-sq-d (make-array (list n-pred (n-true-obs gp))
                                     :element-type 'double-float)))
      (loop for row below (n-true-obs gp)
            for loc-1 in (true-obs-locs gp)
            do (loop for col below row
                     for loc-2 in (true-obs-locs gp)
                     do (setf (aref true-sq-d row col)
                              (coerce (funcall dist-fn loc-1 loc-2)
                                      'double-float)
                              (aref true-sq-d col row)
                              (aref true-sq-d row col)))
               (setf (aref true-sq-d row row)
                     (coerce (funcall dist-fn loc-1 loc-1)
                             'double-float)))
               
      (loop for row below n-pred
            for loc-1 in pred-locs
            do (loop for col below row
                     for loc-2 in pred-locs
                     do (setf (aref pred-sq-d row col)
                              (coerce (funcall dist-fn loc-1 loc-2)
                                      'double-float)
                              (aref pred-sq-d col row)
                              (aref pred-sq-d row col)))
               (setf (aref pred-sq-d row row)
                     (coerce (funcall dist-fn loc-1 loc-1)
                             'double-float))
               (loop for col below (n-true-obs gp)
                     for loc-2 in (true-obs-locs gp)
                     do (setf (aref pred-obs-sq-d row col)
                              (coerce (funcall dist-fn loc-1 loc-2)
                                      'double-float))))
      
      (setf (true-ff-squared-dist gp) (nconc (true-ff-squared-dist gp)
                                             (list true-sq-d))
            (true-ff-abs-dist gp) (nconc (true-ff-abs-dist gp)
                                         (list (aops:vectorize* 'double-float (true-sq-d)
                                                 (sqrt true-sq-d))))
            (pred-ff-squared-dist gp) (nconc (pred-ff-squared-dist gp)
                                             (list pred-sq-d))
            (pred-ff-abs-dist gp) (nconc (pred-ff-abs-dist gp)
                                         (list (aops:vectorize* 'double-float (pred-sq-d)
                                                 (sqrt pred-sq-d))))
            (pred-obs-ff-squared-dist gp) (nconc (pred-obs-ff-squared-dist gp)
                                                 (list pred-obs-sq-d))
            (pred-obs-ff-abs-dist gp) (nconc (pred-obs-ff-abs-dist gp)
                                             (list (aops:vectorize* 'double-float (pred-obs-sq-d)
                                                     (sqrt pred-obs-sq-d))))))))
