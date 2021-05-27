(uiop:define-package #:dag-gp/experiment/andro/dag-gp-experiment
    (:use #:cl
          #:mgl-mat
          #:dag-gp/kernel
          #:dag-gp/output
          #:dag-gp/experiment/andro/read
          #:dag-gp/explicit-dag/dag-gp/all
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/networked-base/all
          #:dag-gp/utils/distance)
  (:export ))

(in-package #:dag-gp/experiment/andro/dag-gp-experiment)


;; Before running this test, ensure andro.arff, downloaded from
;; http://mulan.sourceforge.net/datasets-mtr.html is in this folder.

;; Run this test by entering this package in a lisp process using
;; `(in-package :dag-gp/experiment/andro/dag-gp-experiment)`
;; and running
;; `(andro-test)`


;; Prediction test
(defun andro-test ()
  (let ((dag-gp (make-instance 'dag-gp
                               :output-dim 6
                               :ref-kernel (make-instance 'rbf-kernel)
                               :outputs (list (make-instance 'gaussian-output)
                                              (make-instance 'gaussian-output)
                                              (make-instance 'gaussian-output)
                                              (make-instance 'gaussian-output)
                                              (make-instance 'gaussian-output)
                                              (make-instance 'gaussian-output))))
        (andro (read-andro))
        (andro-truth (make-array '(54 7) :initial-element nil))
        mean-stdev
        mean-temp stdev-temp 
        mean-ph stdev-ph
        mean-cond stdev-cond 
        mean-sal stdev-sal (cae-sal 0) (cse-sal 0) (cse-norm-sal 0) (cnll-sal 0)
        mean-oxy stdev-oxy (cae-oxy 0) (cse-oxy 0) (cse-norm-oxy 0) (cnll-oxy 0)
        mean-turb stdev-turb
        truth-mean-sal truth-mean-oxy
        prediction var)

    (dotimes (i 54)
      ;; Block out oxygen on inputs inputs 30 - 39
      (when (and (>= (aref andro i 0) 30)
                 (<= (aref andro i 0) 39))
        (setf (aref andro-truth i 5) (aref andro i 5)
              (aref andro i 5) nil))

      ;; Block out salinity on inputs 20 - 29
      (when (and (>= (aref andro i 0) 20)
                 (<= (aref andro i 0) 29))
        (setf (aref andro-truth i 4) (aref andro i 4)
              (aref andro i 4) nil)))
    

    (setf mean-stdev (mean-and-stdev andro 1)
          mean-temp (first mean-stdev)
          stdev-temp (second mean-stdev)
          mean-stdev (mean-and-stdev andro 2)
          mean-ph (first mean-stdev)
          stdev-ph (second mean-stdev)
          mean-stdev (mean-and-stdev andro 3)
          mean-cond (first mean-stdev)
          stdev-cond (second mean-stdev)
          mean-stdev (mean-and-stdev andro 4)
          mean-sal (first mean-stdev)
          stdev-sal (second mean-stdev)
          mean-stdev (mean-and-stdev andro 5)
          mean-oxy (first mean-stdev)
          stdev-oxy (second mean-stdev)
          mean-stdev (mean-and-stdev andro 6)
          mean-turb (first mean-stdev)
          stdev-turb (second mean-stdev)
          mean-stdev (mean-and-stdev andro-truth 4)
          truth-mean-sal (first mean-stdev)
          mean-stdev (mean-and-stdev andro-truth 5)
          truth-mean-oxy (first mean-stdev))

    (dotimes (i 54)
      (when (aref andro i 1)
        (add-measurement dag-gp
                         (list (aref andro i 0))
                         0
                         (/ (- (aref andro i 1) mean-temp) stdev-temp)))
      (when (aref andro i 2)
        (add-measurement dag-gp
                         (list (aref andro i 0))
                         1
                         (/ (- (aref andro i 2) mean-ph) stdev-ph)))
      (when (aref andro i 3)
        (add-measurement dag-gp
                         (list (aref andro i 0))
                         2
                         (/ (- (aref andro i 3) mean-cond) stdev-cond)))
      (when (aref andro i 4)
        (add-measurement dag-gp
                         (list (aref andro i 0))
                         3
                         (/ (- (aref andro i 4) mean-sal) stdev-sal)))
      (when (aref andro i 5)
        (add-measurement dag-gp
                         (list (aref andro i 0))
                         4
                         (/ (- (aref andro i 5) mean-oxy) stdev-oxy)))
      (when (aref andro i 6)
        (add-measurement dag-gp
                         (list (aref andro i 0))
                         5
                         (/ (- (aref andro i 6) mean-turb) stdev-turb))))


    (time
     (progn
       (initialize-gp dag-gp)
    
       (configure-for-factor dag-gp 0 nil)
       (configure-for-factor dag-gp 1 nil)
       (configure-for-factor dag-gp 2 nil)
       (configure-for-factor dag-gp 3 nil)
       (configure-for-factor dag-gp 4 nil)
       (configure-for-factor dag-gp 5 nil)
       
       (train dag-gp :progress-fn :summary
                     :search :abc
                     :max-restarts 2
                     :relative-tol 0.005
                     )))

    ;; Predict at locations 0-9
    (loop for i from 30 upto 39 do 
      (destructuring-bind (mean cov) (predict dag-gp
                                              (list (list (aref andro i 0))))
        (with-facets ((pred-array (mean 'array :direction :input))
                      (pred-var (cov 'array :direction :input)))
          (setf prediction (+ mean-oxy (* stdev-oxy (aref pred-array 4)))
                var (* (expt stdev-oxy 2)
                       (aref pred-var 4 4)))
          (incf cae-oxy (abs (- prediction
                                (aref andro-truth i 5))))
          (incf cse-oxy (expt (- prediction
                                 (aref andro-truth i 5))
                              2))
          (incf cse-norm-oxy (expt (- truth-mean-oxy
                                      (aref andro-truth i 5))
                                   2))
          (incf cnll-oxy (* 0.5
                            (+ (/ (expt (- prediction
                                           (aref andro-truth i 5))
                                        2)
                                  var)
                               (log (* 2 PI var)))))
          (format t "pred = ~a, truth = ~a~%"
                  (+ mean-oxy (* stdev-oxy (aref pred-array 4)))
                  (aref andro-truth i 5)))))

    (loop for i from 20 upto 29 do
      (destructuring-bind (mean cov) (predict dag-gp
                                              (list (list (aref andro i 0))))
        (with-facets ((pred-array (mean 'array :direction :input))
                      (pred-var (cov 'array :direction :input)))
          (setf prediction (+ mean-sal (* stdev-sal (aref pred-array 3)))
                var (* (expt stdev-sal 2)
                       (aref pred-var 3 3)))
          (incf cae-sal (abs (- prediction
                                (aref andro-truth i 4))))
          (incf cse-sal (expt (- prediction
                                 (aref andro-truth i 4))
                              2))
          (incf cse-norm-sal (expt (- truth-mean-sal
                                      (aref andro-truth i 4))
                                   2))
          (incf cnll-sal (* 0.5
                            (+ (/ (expt (- prediction
                                           (aref andro-truth i 4))
                                        2)
                                  var)
                               (log (* 2 PI var)))))
          (format t "pred = ~a, truth = ~a~%"
                  (+ mean-sal (* stdev-sal (aref pred-array 3)))
                  (aref andro-truth i 4)))))

    (format t "mae-sal = ~a~%" (/ cae-sal 10))
    (format t "mae-oxy = ~a~%" (/ cae-oxy 10))
    (format t "smse-sal = ~a~%" (/ cse-sal cse-norm-sal))
    (format t "smse-oxy = ~a~%" (/ cse-oxy cse-norm-oxy))
    (format t "average = ~a~%" (/ (+ (/ cse-oxy cse-norm-oxy)
                                     (/ cse-sal cse-norm-sal))
                                  2))
    (format t "mnll-sal = ~a~%" (/ cnll-sal 10))
    (format t "mnll-oxy = ~a~%" (/ cnll-oxy 10))
    (format t "average = ~a~%" (/ (+ (/ cnll-sal 10)
                                     (/ cnll-oxy 10))
                                  2))
    
    dag-gp))


(defun mean-and-stdev (array col)
  (let ((count 0) (mean 0) (stdev 0))
    (dotimes (i (array-dimension array 0))
      (when (aref array i col)
        (incf mean (aref array i col))
        (incf count)))
    (setf mean (/ mean count))
    (dotimes (i (array-dimension array 0))
      (when (aref array i col)
        (incf stdev (expt (- (aref array i col) mean) 2))))
    (setf stdev (sqrt (/ stdev count)))

    (list mean stdev)))


(defun log-lik (y mean var)
  (* -0.5 (+ (/ (expt (- y mean)
                      2)
                var)
             (log var))))


;; Results:

;; Trial 1:
;; mae-sal = 0.22352588590366054d0
;; mae-oxy = 1.2855665030153531d0
;; smse-sal = 0.053245664923093086d0
;; smse-oxy = 0.03209897394426666d0
;; average = 0.04267231943367987d0
;; mnll-sal = 0.8903801948055804d0
;; mnll-oxy = 1.8017765052819121d0
;; average = 1.3460783500437463d0

;; Trial 2:
;; mae-sal = 0.22352635318781786d0
;; mae-oxy = 1.2855658577575368d0
;; smse-sal = 0.05324581741039371d0
;; smse-oxy = 0.032098948002350566d0
;; average = 0.042672382706372136d0
;; mnll-sal = 0.8903823346637545d0
;; mnll-oxy = 1.801776136797825d0
;; average = 1.3460792357307896d0

;; Trial 3:
;; mae-sal = 0.22352598958163092d0
;; mae-oxy = 1.2855658433985127d0
;; smse-sal = 0.05324570637437844d0
;; smse-oxy = 0.03209894613118842d0
;; average = 0.04267232625278343d0
;; mnll-sal = 0.890379849493257d0
;; mnll-oxy = 1.8017760733184538d0
;; average = 1.3460779614058553d0

;; Trial 4:
;; mae-sal = 0.22352614443008356d0
;; mae-oxy = 1.2855657417221622d0
;; smse-sal = 0.053245750258311395d0
;; smse-oxy = 0.03209893349694201d0
;; average = 0.0426723418776267d0
;; mnll-sal = 0.8903812128510413d0
;; mnll-oxy = 1.8017758891332527d0
;; average = 1.346078550992147d0

;; Trial 5:
;; mae-sal = 0.2235263415457414d0
;; mae-oxy = 1.2855664613163085d0
;; smse-sal = 0.053245804638716636d0
;; smse-oxy = 0.03209897897652526d0
;; average = 0.042672391807620946d0
;; mnll-sal = 0.8903833033794276d0
;; mnll-oxy = 1.8017765940005834d0
;; average = 1.3460799486900055d0

;; Trial 6:
;; mae-sal = 0.22352618754616352d0
;; mae-oxy = 1.2855658459077204d0
;; smse-sal = 0.05324576290358012d0
;; smse-oxy = 0.03209894510431085d0
;; average = 0.042672354003945485d0
;; mnll-sal = 0.8903812303374957d0
;; mnll-oxy = 1.8017760816097905d0
;; average = 1.346078655973643d0

;; Trial 7:
;; mae-sal = 0.22352604860327147d0
;; mae-oxy = 1.2855656863249834d0
;; smse-sal = 0.053245715145426975d0
;; smse-oxy = 0.03209894357065874d0
;; average = 0.042672329358042854d0
;; mnll-sal = 0.8903818189007702d0
;; mnll-oxy = 1.8017760761468782d0
;; average = 1.3460789475238242d0

;; Trial 8:
;; mae-sal = 0.22352592675132393d0
;; mae-oxy = 1.2855655360635794d0
;; smse-sal = 0.053245684199567475d0
;; smse-oxy = 0.032098938043361246d0
;; average = 0.04267231112146436d0
;; mnll-sal = 0.8903804611276301d0
;; mnll-oxy = 1.8017759126128916d0
;; average = 1.346078186870261d0

;; Trial 9:
;; mae-sal = 0.22352607398349064d0
;; mae-oxy = 1.2855663271754096d0
;; smse-sal = 0.053245723777732644d0
;; smse-oxy = 0.032098976110730065d0
;; average = 0.04267234994423136d0
;; mnll-sal = 0.8903817757187721d0
;; mnll-oxy = 1.8017765547390845d0
;; average = 1.3460791652289283d0

;; Trial 10:
;; mae-sal = 0.22352616193830882d0
;; mae-oxy = 1.2855662028125465d0
;; smse-sal = 0.05324575389723923d0
;; smse-oxy = 0.03209896375533515d0
;; average = 0.042672358826287185d0
;; mnll-sal = 0.8903812975079475d0
;; mnll-oxy = 1.8017763367437254d0
;; average = 1.3460788171258364d0


;; Average:
;; mae-sal = 0.22352611134714925 +/- 4.753931010585078e-08
;; mae-oxy = 1.2855660005494112 +/- 1.0309455251719639e-07
;; smse-sal = 0.05324573835284396 +/- 1.4740476963737092e-08
;; smse-oxy = 0.0320989547135669 +/- 5.060972143322196e-09
;; average = 0.04267234653320544 +/- 7.932949914435463e-09
;; mnll-sal = 0.8903813478785676 +/- 3.0981101317155966e-07
;; mnll-oxy = 1.8017762160384396 +/- 7.867832648509625e-08
;; average = 1.3460787819585036 +/- 1.739517515241746e-07


;; New best DAG = ((1 0 2 4) (2 3) (0 3 4 5) (3) (4 5) (5)): 369.49014953682445d0
