(uiop:define-package #:dag-gp/explicit-dag/parent-dependent-base/distance-matrix
    (:use #:cl
          #:dag-gp/explicit-dag/parent-dependent-base/base)
  (:import-from #:array-operations)
  (:export #:make-observed-distance-matrices
           #:make-unobserved-distance-matrices
           #:make-predictive-distance-matrices))

(in-package #:dag-gp/explicit-dag/parent-dependent-base/distance-matrix)


(defgeneric make-observed-distance-matrices (gp &key use-all-locs)
  (:documentation "Constructs the matrices of squared and absolute distances for input and parent distances.")
  
  (:method ((gp parent-dependent-gp) &key (use-all-locs t))
    (let ((n-obs (if (and (closed-downwards-p gp)
                          (not use-all-locs))
                     (n-true-obs gp)
                     (n-obs gp)))
          (obs-locs (if (and (closed-downwards-p gp)
                             (not use-all-locs))
                        (true-obs-locs gp)
                        (obs-locs gp))))
    ;; Input (x) distances
    (setf (ff-squared-dist gp) nil
          (ff-abs-dist gp) nil)
    
    (loop for dist-fn in (dist-fns gp) do
      (let ((sq-d (make-array (list n-obs n-obs)
                              :element-type 'double-float)))
        (loop for row below n-obs
              for loc-1 in obs-locs
              do (loop for col below row
                       for loc-2 in obs-locs
                       do (setf (aref sq-d row col)
                                (coerce (funcall dist-fn loc-1 loc-2)
                                        'double-float)
                                (aref sq-d col row)
                                (aref sq-d row col)))
                 (setf (aref sq-d row row)
                       (coerce (funcall dist-fn loc-1 loc-1)
                               'double-float)))
        
        (setf (ff-squared-dist gp) (nconc (ff-squared-dist gp)
                                          (list sq-d))
              (ff-abs-dist gp) (nconc (ff-abs-dist gp)
                                      (list (aops:vectorize* 'double-float (sq-d)
                                              (sqrt sq-d)))))))))

  ;; In the variational case, we only need diagonal elements
  (:method ((gp variational-parent-dependent-gp) &key (use-all-locs t))
    (declare (ignore use-all-locs))
    ;; Input (x) distances
    (setf (ff-squared-dist gp) nil
          (ff-abs-dist gp) nil)
    
    (loop for dist-fn in (dist-fns gp) do
      (let ((sq-d (make-array (n-obs gp)
                              :element-type 'double-float)))
        (loop for row below (n-obs gp)
              for loc-1 in (obs-locs gp)
              do (setf (aref sq-d row)
                       (coerce (funcall dist-fn loc-1 loc-1)
                               'double-float)))
        
        (setf (ff-squared-dist gp) (nconc (ff-squared-dist gp)
                                          (list sq-d))
              (ff-abs-dist gp) (nconc (ff-abs-dist gp)
                                      (list (aops:vectorize* 'double-float (sq-d)
                                              (sqrt sq-d)))))))))


(defgeneric make-unobserved-distance-matrices (gp)
  (:documentation "Constructs the matrices containing squared and absolute distances between latent observations and observed and latent observations.")
  (:method ((gp variational-parent-dependent-gp))
    (setf (uu-squared-dist gp) nil
          (uu-abs-dist gp) nil
          (fu-squared-dist gp) nil
          (fu-abs-dist gp) nil)

    ;; uu distances
    (loop for dist-fn in (dist-fns gp) do
      (let ((sq-d (make-array (list (n-latent gp) (n-latent gp))
                              :element-type 'double-float)))
        (loop for row below (n-latent gp)
              for loc-1 = (loop for j below (input-dim gp)
                                collect (aref (u-locs gp) row j))
              do (loop for col below row
                       for loc-2 = (loop for j below (input-dim gp)
                                         collect (aref (u-locs gp) col j))
                       do (setf (aref sq-d row col)
                                (coerce (funcall dist-fn loc-1 loc-2)
                                        'double-float)
                                (aref sq-d col row)
                                (aref sq-d row col)))
                 (setf (aref sq-d row row)
                       (coerce (funcall dist-fn loc-1 loc-1)
                               'double-float)))
        
        (setf (uu-squared-dist gp) (nconc (uu-squared-dist gp)
                                        (list sq-d))
              (uu-abs-dist gp) (nconc (uu-abs-dist gp)
                                      (list (aops:vectorize* 'double-float (sq-d)
                                              (sqrt sq-d)))))))

    ;; fu distances 
    (loop for dist-fn in (dist-fns gp) do
      (let ((sq-d (make-array (list (n-obs gp) (n-latent gp))
                              :element-type 'double-float)))
        (loop for col below (n-latent gp)
              for loc-2 = (loop for j below (input-dim gp)
                                collect (aref (u-locs gp) col j))
              do (loop for row below (n-obs gp)
                       for loc-1 in (obs-locs gp)
                       do (setf (aref sq-d row col)
                                (coerce (funcall dist-fn loc-1 loc-2)
                                        'double-float))))
    
        (setf (fu-squared-dist gp) (nconc (fu-squared-dist gp)
                                          (list sq-d))
              (fu-abs-dist gp) (nconc (fu-abs-dist gp)
                                      (list (aops:vectorize* 'double-float (sq-d)
                                              (sqrt sq-d)))))))))


(defgeneric make-predictive-distance-matrices (gp pred-locs n-pred
                                               &key use-all-locs)
  (:documentation "Constructs the matrices containing squared distances and absolute distances for prediction.")

  (:method ((gp parent-dependent-gp) pred-locs n-pred &key (use-all-locs t))
    (setf (pred-ff-squared-dist gp) nil
          (pred-ff-abs-dist gp) nil
          (pred-obs-ff-squared-dist gp) nil
          (pred-obs-ff-abs-dist gp) nil)
    (let ((n-obs (if (and (closed-downwards-p gp)
                          (not use-all-locs))
                     (n-true-obs gp)
                     (n-obs gp)))
          (obs-locs (if (and (closed-downwards-p gp)
                             (not use-all-locs))
                        (true-obs-locs gp)
                        (obs-locs gp))))
      (loop for dist-fn in (dist-fns gp) do
        (let ((pred-sq-d (make-array (list n-pred n-pred)
                                     :element-type 'double-float))
              (pred-obs-sq-d (make-array (list n-pred n-obs)
                                         :element-type 'double-float)))
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
                   (loop for col below n-obs
                         for loc-2 in obs-locs
                         do (setf (aref pred-obs-sq-d row col)
                                  (coerce (funcall dist-fn loc-1 loc-2)
                                          'double-float))))
        
          (setf (pred-ff-squared-dist gp) (nconc (pred-ff-squared-dist gp)
                                                 (list pred-sq-d))
                (pred-ff-abs-dist gp) (nconc (pred-ff-abs-dist gp)
                                             (list (aops:vectorize* 'double-float (pred-sq-d)
                                                     (sqrt pred-sq-d))))
                (pred-obs-ff-squared-dist gp) (nconc (pred-obs-ff-squared-dist gp)
                                                     (list pred-obs-sq-d))
                (pred-obs-ff-abs-dist gp) (nconc (pred-obs-ff-abs-dist gp)
                                                 (list (aops:vectorize* 'double-float (pred-obs-sq-d)
                                                         (sqrt pred-obs-sq-d)))))))))
  
  (:method ((gp variational-parent-dependent-gp) pred-locs n-pred &key use-all-locs)
    (declare (ignore use-all-locs))
    (setf (pred-ff-squared-dist gp) nil
          (pred-ff-abs-dist gp) nil
          (pred-fu-squared-dist gp) nil
          (pred-fu-abs-dist gp) nil)

    ;; ff distances
    (loop for dist-fn in (dist-fns gp) do
      (let ((sq-d (make-array (list n-pred n-pred)
                              :element-type 'double-float)))
        (loop for row below n-pred
              for loc-1 in pred-locs
              do (loop for col below row
                       for loc-2 in pred-locs
                       do (setf (aref sq-d row col)
                                (coerce (funcall dist-fn loc-1 loc-2)
                                        'double-float)
                                (aref sq-d col row)
                                (aref sq-d row col)))
                 (setf (aref sq-d row row)
                       (coerce (funcall dist-fn loc-1 loc-1)
                               'double-float)))
        
        (setf (pred-ff-squared-dist gp) (nconc (pred-ff-squared-dist gp)
                                               (list sq-d))
              (pred-ff-abs-dist gp) (nconc (pred-ff-abs-dist gp)
                                           (list (aops:vectorize* 'double-float (sq-d)
                                                   (sqrt sq-d)))))))

    ;; fu distances 
    (loop for dist-fn in (dist-fns gp) do
      (let ((sq-d (make-array (list n-pred (n-latent gp))
                              :element-type 'double-float)))
        (loop for col below (n-latent gp)
              for loc-2 = (loop for j below (input-dim gp)
                                collect (aref (u-locs gp) col j))
              do (loop for row below n-pred
                       for loc-1 in pred-locs
                       do (setf (aref sq-d row col)
                                (coerce (funcall dist-fn loc-1 loc-2)
                                        'double-float))))
    
        (setf (pred-fu-squared-dist gp) (nconc (pred-fu-squared-dist gp)
                                               (list sq-d))
              (pred-fu-abs-dist gp) (nconc (pred-fu-abs-dist gp)
                                           (list (aops:vectorize* 'double-float (sq-d)
                                                   (sqrt sq-d)))))))))


