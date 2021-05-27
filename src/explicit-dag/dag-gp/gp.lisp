(uiop:define-package #:dag-gp/explicit-dag/dag-gp/gp
    (:use #:cl
          #:mgl-mat
          #:dag-gp/output
          #:dag-gp/quadrature
          #:dag-gp/explicit-dag/networked-base/all
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/linear-parent/all)
  (:export #:dag-gp-base
           #:dag-gp
           #:variational-dag-gp
           #:combined-dists
           #:dag-closed-downwards-p
           #:impute-indices

           #:make-constituent-gps
           #:configure-for-factor
           #:sub-distribution-for-factor
           #:full-sub-distribution-for-factor))

(in-package #:dag-gp/explicit-dag/dag-gp/gp)


;; A DAG-GP is a networked GP where each constituent is a linear parent GP.
(defclass dag-gp-base (networked-base)
  ((combined-distributions
    :initform nil
    :accessor combined-dists
    :documentation "Distributions for the combined output dimensions.")
   (closed-downwards-p
    :initarg :closed-downwards-p
    :initform nil
    :accessor dag-closed-downwards-p)
   (impute-indices
    :initarg :impute
    :initform nil
    :accessor impute-indices)
   (possible-descendants
    :initform nil
    :accessor possible-descendants
    :documentation "An a-list of index to indices of possible descendants. Only used 
when closed-downwards-p.")))


(defclass dag-gp (dag-gp-base)
  ())


(defclass variational-dag-gp (dag-gp-base variational-networked-base)
  ((combined-distributions
    :initform (make-hash-table :test #'equalp)
    :accessor combined-dists)))


(defmethod make-constituent-gps ((gp dag-gp))
  (loop for output in (outputs gp) do
    (setf (constituent-gps gp)
          (nconc (constituent-gps gp)
                 (list (make-instance 'linear-parent-gp
                                      :dist-fns (ref-dist-fns gp)
                                      :output output
                                      :default-parent-param (ref-parent-param gp)
                                      :closed-downwards-p
                                      (dag-closed-downwards-p gp)))))))


(defmethod make-constituent-gps ((gp variational-dag-gp))
  (loop for output in (outputs gp) do
    (setf (constituent-gps gp)
          (nconc (constituent-gps gp)
                 (list
                  (if (typep output 'multi-output)
                      (make-instance 'variational-combined-output-linear-parent-gp
                                     :n-latent (n-latent-per-constituent gp)
                                     :dist-fns (ref-dist-fns gp)
                                     :output output
                                     :n-combined (n-gps output)
                                     :input-dim (networked-input-dim gp)
                                     :default-parent-param (ref-parent-param gp)
                                     :closed-downwards-p (dag-closed-downwards-p gp))
                      (make-instance 'variational-linear-parent-gp
                                     :n-latent (n-latent-per-constituent gp)
                                     :dist-fns (ref-dist-fns gp)
                                     :output output
                                     :input-dim (networked-input-dim gp)
                                     :default-parent-param (ref-parent-param gp)
                                     :closed-downwards-p
                                     (dag-closed-downwards-p gp))))))))


;; (defmethod initialize-gp ((gp dag-gp-base) &key (initialize-latent t)
;;                                              (use-all-locs t))
;;   (declare (ignore initialize-latent use-all-locs))
;;   (loop for constituent-gp in (constituent-gps gp)
;;         do (initialize-measurements constituent-gp
;;                                     (constituent-locs gp)))
;;   (call-next-method))
  

(defmethod configure-for-factor ((gp dag-gp) index parents &key (use-all-locs t))
  ;; Initialize the measurements of the constituent for all DAG-GP measurements
  (declare (ignore use-all-locs))
  (let ((constituent-gp (nth index (constituent-gps gp)))
        parent-local-obs
        parent-obs rest-parent-obs
        parent-loc rest-parent-locs)
    (initialize-measurements constituent-gp
                             (constituent-locs gp))
    
    ;; Perform the rest of the initialization
    (call-next-method)
    
    ;; Copy over distributions for locations
    (setf (var-parent-distributions constituent-gp) nil)
    (when (combined-dists gp)
      (if (closed-downwards-p constituent-gp)
          (progn
            (setf (var-parent-distributions constituent-gp)
                  (list* (make-mat (n-true-obs constituent-gp)
                                   :ctype :double
                                   :initial-contents (true-obs constituent-gp))
                         (loop for parent in parents
                               for parent-gp = (nth parent (constituent-gps gp))
                               do (setf parent-local-obs
                                        (make-array (n-true-obs constituent-gp)
                                                    :element-type 'double-float)
                                        parent-obs (first (true-obs parent-gp))
                                        rest-parent-obs (rest (true-obs parent-gp))
                                        parent-loc (first (true-obs-locs parent-gp))
                                        rest-parent-locs (rest (true-obs-locs parent-gp)))
                                  (loop for loc in (true-obs-locs constituent-gp)
                                        for i from 0
                                        ;; Find the local loc in the parent observed locs
                                        do (loop while (not (equalp loc parent-loc)) do
                                          (setf parent-obs (first rest-parent-obs)
                                                rest-parent-obs (rest rest-parent-obs)
                                                parent-loc (first rest-parent-locs)
                                                rest-parent-locs (rest rest-parent-locs)))
                                           ;; Add the parent observation to the array
                                           (setf (aref parent-local-obs i) parent-obs))
                               collect (array-to-mat parent-local-obs
                                                     :ctype :double)))))
          
          (setf (var-parent-distributions constituent-gp)
                (full-sub-distribution-for-factor (combined-dists gp)
                                                  index
                                                  parents
                                                  (n-obs constituent-gp)))))))

;; TODO: There is code duplication between here and update-combined-distributions
;; in /dag-gp/predict. Setting the var-parent-distributions may not be nexessary.
(defmethod configure-for-factor ((gp variational-dag-gp) index parents
                                 &key (use-all-locs t))
  (declare (ignore use-all-locs))
  ;; Initialize the measurements of the constituent for all DAG-GP measurements
  (let ((constituent-gp (nth index (constituent-gps gp))))
    (initialize-measurements constituent-gp
                             (constituent-locs gp))
    
    ;; Perform the rest of the initialization
    (call-next-method)
    
    ;; Copy over distributions for locations
    (setf (var-parent-distributions constituent-gp) nil)
    (if (all-gaussian-p gp)
        (loop for loc in (obs-locs constituent-gp) do
          (when (gethash loc (combined-dists gp))
            (setf (var-parent-distributions constituent-gp)
                  (nconc (var-parent-distributions constituent-gp)
                         (list (sub-distribution-for-factor
                                (gethash loc (combined-dists gp))
                                index
                                parents))))))

        (loop for loc in (obs-locs constituent-gp)
              for pred = (gethash loc (combined-dists gp))
              do (when pred
                   (let ((new-quad (copy-quad-set pred)))
                     ;; Marginalize everything unneeded
                     (loop for var below (output-dim gp) do
                       (unless (or (equal var index)
                                   (member var parents))
                         (marginalize new-quad var)))
                     (flatten new-quad (list* index parents))
                     (setf (var-parent-distributions constituent-gp)
                           (nconc (var-parent-distributions constituent-gp)
                                  (list new-quad)))))))))


(defun full-sub-distribution-for-factor (distribution index parents n-obs)
  (with-facets ((mean ((first distribution) 'backing-array :direction :input))
                (cov ((second distribution) 'array :direction :input)))
    (let* ((dist-size (* (1+ (list-length parents)) n-obs))
           (new-mean (make-array dist-size :element-type 'double-float))
           (new-cov (make-array (list dist-size dist-size)
                                :element-type 'double-float))
           (index-offset (* index n-obs)))
      (loop for row below (1+ (list-length parents))
            for row-par-index = (when (> row 0) (nth (1- row) parents))
            for row-offset = (* row n-obs)
            for row-par-offset = (when (> row 0) (* row-par-index n-obs))
            do (if (equal row 0)
                   (loop for i below n-obs do
                     (setf (aref new-mean i) (aref mean (+ index-offset i))))
                   (loop for i below n-obs do
                     (setf (aref new-mean (+ row-offset i))
                           (aref mean (+ row-par-offset i)))))
               (loop for col below row
                     for col-par-index = (when (> col 0) (nth (1- col) parents))
                     for col-offset = (* col n-obs)
                     for col-par-offset = (when (> col 0) (* col-par-index n-obs)) 
                     do (if (equal col 0)
                            (loop for i below n-obs do
                              (loop for j below n-obs do
                                (setf (aref new-cov (+ row-offset i)
                                            (+ col-offset j))
                                    (aref cov (+ row-par-offset i)
                                          (+ index-offset j)))))
                            (loop for i below n-obs do
                              (loop for j below n-obs do
                                (setf (aref new-cov (+ row-offset i)
                                            (+ col-offset j))
                                      (aref cov (+ row-par-offset i)
                                            (+ col-par-offset j))))))
                        (loop for i below n-obs do
                          (loop for j below n-obs do
                            (setf (aref new-cov (+ col-offset j)
                                        (+ row-offset i))
                                  (aref new-cov (+ row-offset i)
                                        (+ col-offset j))))))
               (if (equal row 0)
                   (loop for i below n-obs do
                     (loop for j below n-obs do
                       (setf (aref new-cov i j)
                             (aref cov (+ index-offset i) (+ index-offset j)))))
                    (loop for i below n-obs do
                      (loop for j below n-obs do
                        (setf (aref new-cov (+ row-offset i) (+ row-offset j))
                              (aref cov (+ row-par-offset i)
                                    (+ row-par-offset j)))))))
      (list (array-to-mat new-mean :ctype :double)
            (array-to-mat new-cov :ctype :double)))))

(defun sub-distribution-for-factor (distribution index parents)
  ;; Creates new distributions for the factor
  (with-facets ((mean ((first distribution) 'backing-array :direction :input))
                (cov ((second distribution) 'array :direction :input)))
    (let* ((dist-size (1+ (list-length parents)))
           (new-mean (make-array dist-size :element-type 'double-float))
           (new-cov (make-array (list dist-size dist-size) :element-type 'double-float)))
      (loop for row below dist-size
            for row-par-index = (when (> row 0) (nth (1- row) parents))
            do (if (equal row 0)
                   (setf (aref new-mean 0) (aref mean index))
                   (setf (aref new-mean row) (aref mean row-par-index)))
               (loop for col below row
                     for col-par-index = (when (> col 0) (nth (1- col) parents))
                     do (if (equal col 0)
                            (setf (aref new-cov row col)
                                  (aref cov row-par-index index))
                            (setf (aref new-cov row col)
                                  (aref cov row-par-index col-par-index)))
                        (setf (aref new-cov col row)
                              (aref new-cov row col)))
               (if (equal row 0)
                   (setf (aref new-cov 0 0)
                         (aref cov index index))
                   (setf (aref new-cov row row)
                         (aref cov row-par-index row-par-index))))
      (list (array-to-mat new-mean :ctype :double)
            (array-to-mat new-cov :ctype :double)))))
