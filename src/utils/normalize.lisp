(uiop:define-package #:dag-gp/utils/normalize
    (:use #:cl)
  (:export #:normalize-data))

(in-package #:dag-gp/utils/normalize)


(defun normalize-data (data col)
  (let ((n-data (array-dimension data 0))
        (mean 0)
        (stdev 0))
    (loop for i below n-data do
      (incf mean (aref data i col)))
    (setf mean (/ mean n-data))

    (loop for i below n-data do
      (incf stdev (expt (- (aref data i col) mean) 2)))
    (setf stdev (sqrt (/ stdev n-data)))

    (loop for i below n-data do
      (setf (aref data i col)
            (/ (- (aref data i col) mean)
               stdev)))
    (list mean stdev)))
