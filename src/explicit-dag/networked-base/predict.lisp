(uiop:define-package #:dag-gp/explicit-dag/networked-base/predict
    (:use #:cl
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/networked-base/base)
  (:export #:predict
           #:solve-factor-ordering))

(in-package #:dag-gp/explicit-dag/networked-base/predict)


(defgeneric predict (gp pred-locs)
  (:documentation "Predicts the combined output of the multi-output GP.")
  (:method ((gp networked-base) pred-locs)
    (loop for constituent-gp in (constituent-gps gp)
          collect (make-predictive-posteriors constituent-gp pred-locs))))


(defgeneric solve-factor-ordering (gp)
  (:documentation "Sets the factor ordering to be up to date.")
  (:method ((gp networked-base))
    (let ((vars (loop for i below (list-length (constituent-gps gp)) collect i)))
      (setf (factor-ordering gp) nil)
      (dotimes (i (list-length (constituent-gps gp)))
        (block next-factor
          (loop for var in vars
                for parents = (cdr (assoc var (factors gp)))
                do (block check-parents
                     (loop for parent in parents do
                       ;; If a parent hasn't yet been selected, can't use var
                       (unless (member parent (factor-ordering gp))
                         (return-from check-parents)))
                     ;; If all parents have been selected, use this as next factor
                     (setf (factor-ordering gp)
                           (nconc (factor-ordering gp) (list var))
                           vars (remove var vars))
                     (return-from next-factor))))))))
