(uiop:define-package #:dag-gp/experiment/jura/dag-gp-experiment
    (:use #:cl
          #:mgl-mat
          #:dag-gp/kernel
          #:dag-gp/output
          #:dag-gp/experiment/jura/read
          #:dag-gp/explicit-dag/dag-gp/all
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/networked-base/all
          #:dag-gp/utils/distance
          )
  (:export ))

(in-package #:dag-gp/experiment/jura/dag-gp-experiment)


;; Before running this test, ensure jura_sample.dat and jura_validation.dat,
;; downloaded from https://sites.google.com/site/goovaertspierre/pierregoovaertswebsite/download/jura-data are in this folder.

;; Run this test by entering this package in a lisp process using
;; `(in-package :dag-gp/experiment/jura/dag-gp-experiment)`
;; and running
;; `(explicit-ni-zn-cd-test)`


;; Test in which Nickel and Zinc are observed, and Cadmium is predicted
(defun explicit-ni-zn-cd-test ()
  (let ((dag-gp (make-instance 'dag-gp
                               :output-dim 3
                               :ref-kernel (make-instance 'rbf-kernel)
                               :outputs (list (make-instance 'gaussian-output)
                                              (make-instance 'gaussian-output)
                                              (make-instance 'gaussian-output))))
        (jura-sample (read-jura-sample))
        (jura-validation (read-jura-validation))
        (cd-mean 0) (cd-stdev 0) (cd-err 0)
        (ni-mean 0) (ni-stdev 0)
        (zn-mean 0) (zn-stdev 0) (zn-col 10)) ;; Was 10

    (dotimes (i 259)
      (incf ni-mean (log (aref jura-sample i 8))))
    (dotimes (i 100)
      (incf ni-mean (log (aref jura-validation i 8))))
    (setf ni-mean (/ ni-mean 359))
    (dotimes (i 259)
      (incf ni-stdev (expt (- (log (aref jura-sample i 8)) ni-mean) 2)))
    (dotimes (i 100)
      (incf ni-stdev (expt (- (log (aref jura-validation i 8)) ni-mean) 2))) ;; Was sample
    (setf ni-stdev (sqrt (/ ni-stdev 359)))

    (dotimes (i 259)
      (incf zn-mean (log (aref jura-sample i zn-col))))
    (dotimes (i 100)
      (incf zn-mean (log (aref jura-validation i zn-col))))
    (setf zn-mean (/ zn-mean 359))
    (dotimes (i 259)
      (incf zn-stdev (expt (- (log (aref jura-sample i zn-col)) zn-mean) 2)))
    (dotimes (i 100)
      (incf zn-stdev (expt (- (log (aref jura-validation i zn-col)) zn-mean) 2))) ;; Was sample
    (setf zn-stdev (sqrt (/ zn-stdev 359)))

    (dotimes (i 259)
      (incf cd-mean (log (aref jura-sample i 4))))
    (setf cd-mean (/ cd-mean 259))
    (dotimes (i 259)
      (incf cd-stdev (expt (- (log (aref jura-sample i 4)) cd-mean) 2)))
    (setf cd-stdev (sqrt (/ cd-stdev 259)))
    
    ;; Add Nickel (col 8)
    (dotimes (i 259)
      (add-measurement dag-gp
                       (list (aref jura-sample i 0)
                             (aref jura-sample i 1))
                       0
                       (/ (- (log (aref jura-sample i 8)) ni-mean) ni-stdev)))
    (dotimes (i 100)
      (add-measurement dag-gp
                       (list (aref jura-validation i 0)
                             (aref jura-validation i 1))
                       0
                       (/ (- (log (aref jura-validation i 8)) ni-mean) ni-stdev)))

    ;; Add Zinc (col 10)
    (dotimes (i 259)
      (add-measurement dag-gp
                       (list (aref jura-sample i 0)
                             (aref jura-sample i 1))
                       1
                       (/ (- (log (aref jura-sample i zn-col)) zn-mean) zn-stdev)))
    (dotimes (i 100)
      (add-measurement dag-gp
                       (list (aref jura-validation i 0)
                             (aref jura-validation i 1))
                       1
                       (/ (- (log (aref jura-validation i zn-col)) zn-mean) zn-stdev)))

    ;; Add Cadmium (col 4)
    (dotimes (i 259)
      (add-measurement dag-gp
                       (list (aref jura-sample i 0)
                             (aref jura-sample i 1))
                       2
                       (/ (- (log (aref jura-sample i 4)) cd-mean) cd-stdev)))

    (initialize-gp dag-gp)
    
    (configure-for-factor dag-gp 0 nil)
    (configure-for-factor dag-gp 1 nil)
    (configure-for-factor dag-gp 2 nil)

    
    (train dag-gp :progress-fn :summary
                  :relative-tol 0.005)
    
    (dotimes (i 100)
      (let ((pred-loc (list (aref jura-validation i 0)
                            (aref jura-validation i 1))))
        (with-facets ((pred-array ((first (predict dag-gp
                                                   (list pred-loc)))
                                   'array :direction :input)))
          (incf cd-err (abs (- (exp (+ cd-mean (* cd-stdev (aref pred-array 2))))
                               (aref jura-validation i 4)))))))
    (setf cd-err (/ cd-err 100))

    (format t "mae = ~a~%" cd-err)
    cd-err))



;; Results:

;; Trial 1:
;; mae = 0.3921593886655393d0

;; Trial 2:
;; mae = 0.39215939456775184d0

;; Trial 3:
;; mae = 0.39215939279855233d0

;; Trial 4:
;; mae = 0.4169293178999542d0

;; Trial 5:
;; mae = 0.39215939446304d0

;; Trial 6:
;; mae = 0.39215939446304d0

;; Trial 7:
;; mae = 0.39215939431057434d0

;; Trial 8:
;; mae = 0.3921590777372937d0

;; Trial 9:
;; mae = 0.39215939453347043d0

;; Trial 10:
;; mae = 0.39215939453347043d0


;; Average mae = 0.3946 +/- 0.0023

