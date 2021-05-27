(uiop:define-package #:dag-gp/experiment/exchange/dag-gp-experiment
    (:use #:cl
          #:mgl-mat
          #:dag-gp/kernel
          #:dag-gp/output
          #:dag-gp/experiment/exchange/read
          #:dag-gp/explicit-dag/dag-gp/all
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/networked-base/all
          #:dag-gp/utils/distance)
  (:export ))

(in-package #:dag-gp/experiment/exchange/dag-gp-experiment)


;; Use AU(1), AG(2), CAD(4), EUR(5), CHF(8), AUD(9), HKD(10), NZD(11)

;; Before running this test, download exchange.dat into this folder from
;; http://fx.sauder.ubc.ca/.

;; Run this test by entering this package in a lisp process using
;; `(in-package #:dag-gp/experiment/exchange/dag-gp-experiment)`
;; and running
;; `(eight-exact-closed-downwards-test nil)`


(defun eight-exact-closed-downwards-test (dag)
  (flet ((allowable-fn (variable parents)
           (if (member variable '(0 1 3 4 7))
               (if parents
                   nil
                   t)
               (if (subsetp parents '(0 1 3 4 7))
                   t
                   nil)))
         (max-set-fn (variable)
           (if (member variable '(0 1 3 4 7))
               nil
               '(0 1 3 4 7))))
    (let ((dag-gp (if dag
                      dag
                      (make-instance 'dag-gp
                                     :output-dim 8
                                     :ref-kernel (make-instance 'rational-quadratic-kernel)
                                     :outputs (list (make-instance 'gaussian-output)
                                                    (make-instance 'gaussian-output)
                                                    (make-instance 'gaussian-output)
                                                    (make-instance 'gaussian-output)
                                                    (make-instance 'gaussian-output)
                                                    (make-instance 'gaussian-output)
                                                    (make-instance 'gaussian-output)
                                                    (make-instance 'gaussian-output))
                                     :closed-downwards-p t
                                     :impute '(0 1 3 4 7))))
          (exchange (read-exchange))
          (exchange-truth (make-array '(251 14) :initial-element nil))
          mean-stdev
          au-mean au-stdev
          ag-mean ag-stdev
          cad-mean cad-stdev (cad-cae 0) (cad-cse 0) (cad-cse-norm 0) (cad-cnll 0)
          eur-mean eur-stdev
          chf-mean chf-stdev
          aud-mean aud-stdev (aud-cae 0) (aud-cse 0) (aud-cse-norm 0) (aud-cnll 0)
          hkd-mean hkd-stdev (hkd-cae 0) (hkd-cse 0) (hkd-cse-norm 0) (hkd-cnll 0)
          nzd-mean nzd-stdev
          cad-truth-mean aud-truth-mean hkd-truth-mean
          prediction var)
      
      (dotimes (i 251)
        ;; Block out CAD on days 50-100
        (when (and (>= i 49)
                   (<= i 99))
          (setf (aref exchange-truth i 4) (aref exchange i 4)
                (aref exchange i 4) nil))
        
        ;; Block out HKD on days 100-150
        (when (and (>= i 99)
                   (<= i 149))
          (setf (aref exchange-truth i 10) (aref exchange i 10)
                (aref exchange i 10) nil))
        
        ;; Block out AUD on days 150-200
        (when (and (>= i 149)
                   (<= i 199))
          (setf (aref exchange-truth i 9) (aref exchange i 9)
                (aref exchange i 9) nil)))
      
      
      (setf mean-stdev (mean-and-stdev exchange 1)
            au-mean (first mean-stdev)
            au-stdev (second mean-stdev)
            mean-stdev (mean-and-stdev exchange 2)
            ag-mean (first mean-stdev)
            ag-stdev (second mean-stdev)
            mean-stdev (mean-and-stdev exchange 4)
            cad-mean (first mean-stdev)
            cad-stdev (second mean-stdev)
            mean-stdev (mean-and-stdev exchange 5)
            eur-mean (first mean-stdev)
            eur-stdev (second mean-stdev)
            mean-stdev (mean-and-stdev exchange 8)
            chf-mean (first mean-stdev)
            chf-stdev (second mean-stdev)
            mean-stdev (mean-and-stdev exchange 9)
            aud-mean (first mean-stdev)
            aud-stdev (second mean-stdev)
            mean-stdev (mean-and-stdev exchange 10)
            hkd-mean (first mean-stdev)
            hkd-stdev (second mean-stdev)
            mean-stdev (mean-and-stdev exchange 11)
            nzd-mean (first mean-stdev)
            nzd-stdev (second mean-stdev)
            mean-stdev (mean-and-stdev exchange-truth 4)
            cad-truth-mean (first mean-stdev)
            mean-stdev (mean-and-stdev exchange-truth 10)
            hkd-truth-mean (first mean-stdev)
            mean-stdev (mean-and-stdev exchange-truth 9)
            aud-truth-mean (first mean-stdev))
      
      (unless dag
        (dotimes (i 251)
          (when (aref exchange i 1)
            (add-measurement dag-gp
                             (list (aref exchange i 0))
                             0
                             (/ (- (aref exchange i 1) au-mean) au-stdev)))
          (when (aref exchange i 2)
            (add-measurement dag-gp
                             (list (aref exchange i 0))
                             1
                             (/ (- (aref exchange i 2) ag-mean) ag-stdev)))
          (when (aref exchange i 4)
            (add-measurement dag-gp
                             (list (aref exchange i 0))
                             2
                             (/ (- (aref exchange i 4) cad-mean) cad-stdev)))
          (when (aref exchange i 5)
            (add-measurement dag-gp
                             (list (aref exchange i 0))
                             3
                             (/ (- (aref exchange i 5) eur-mean) eur-stdev)))
          (when (aref exchange i 8)
            (add-measurement dag-gp
                             (list (aref exchange i 0))
                             4
                             (/ (- (aref exchange i 8) chf-mean) chf-stdev)))
          (when (aref exchange i 9)
            (add-measurement dag-gp
                             (list (aref exchange i 0))
                             5
                             (/ (- (aref exchange i 9) aud-mean) aud-stdev)))
          (when (aref exchange i 10)
            (add-measurement dag-gp
                             (list (aref exchange i 0))
                             6
                             (/ (- (aref exchange i 10) hkd-mean) hkd-stdev)))
          (when (aref exchange i 11)
            (add-measurement dag-gp
                             (list (aref exchange i 0))
                             7
                             (/ (- (aref exchange i 11) nzd-mean) nzd-stdev))))
        
        (initialize-gp dag-gp)
        
        (configure-for-factor dag-gp 0 '())
        (configure-for-factor dag-gp 1 '())
        (configure-for-factor dag-gp 2 '())
        (configure-for-factor dag-gp 3 '())
        (configure-for-factor dag-gp 4 '())
        (configure-for-factor dag-gp 5 '())
        (configure-for-factor dag-gp 6 '())
        (configure-for-factor dag-gp 7 '())
        
        (time
         (train dag-gp :progress-fn :summary
                       :search :abc
                       :allowable-fn #'allowable-fn
                       :max-set-fn #'max-set-fn
                       :relative-tol 0.005)))


      (destructuring-bind (pred-mean pred-cov)
          (predict dag-gp (loop for i from 49 upto 99
                                collect (list (aref exchange i 0))))
        (with-facets ((mean-array (pred-mean 'array :direction :input))
                      (cov-array (pred-cov 'array :direction :input)))
          (dotimes (i 51)
            (setf prediction (+ cad-mean
                                (* cad-stdev
                                   (aref mean-array (+ (* 2 51) i))))
                  var (* (expt cad-stdev 2)
                         (aref cov-array (+ (* 2 51) i) (+ (* 2 51) i))))
            (incf cad-cae (abs (- prediction
                                  (aref exchange-truth (+ 49 i) 4))))
            (incf cad-cse (expt (- prediction
                                   (aref exchange-truth (+ 49 i) 4))
                                2))
            (incf cad-cse-norm (expt (- cad-truth-mean
                                        (aref exchange-truth (+ 49 i) 4))
                                     2))
            (incf cad-cnll (* 0.5
                              (+ (/ (expt (- prediction
                                             (aref exchange-truth (+ 49 i) 4))
                                          2)
                                    var)
                                 (log (* 2 PI var)))))
            (format t "pred = ~a, truth = ~a~%"
                    prediction
                    (aref exchange-truth (+ 49 i) 4)))))
      
      (destructuring-bind (pred-mean pred-cov)
          (predict dag-gp (loop for i from 99 upto 149
                                collect (list (aref exchange i 0))))
        (with-facets ((mean-array (pred-mean 'array :direction :input))
                      (cov-array (pred-cov 'array :direction :input)))
          (dotimes (i 51)
            (setf prediction (+ hkd-mean
                                (* hkd-stdev
                                   (aref mean-array (+ (* 6 51) i))))
                  var (* (expt hkd-stdev 2)
                         (aref cov-array (+ (* 6 51) i) (+ (* 6 51) i))))
            (incf hkd-cae (abs (- prediction
                                  (aref exchange-truth (+ 99 i) 10))))
            (incf hkd-cse (expt (- prediction
                                   (aref exchange-truth (+ 99 i) 10))
                                2))
            (incf hkd-cse-norm (expt (- hkd-truth-mean
                                        (aref exchange-truth (+ 99 i) 10))
                                     2))
            (incf hkd-cnll (* 0.5
                              (+ (/ (expt (- prediction
                                             (aref exchange-truth (+ 99 i) 10))
                                          2)
                                    var)
                                 (log (* 2 PI var)))))
            (format t "pred = ~a, truth = ~a~%"
                    (+ hkd-mean (* hkd-stdev (aref mean-array (+ (* 6 51) i))))
                    (aref exchange-truth (+ 99 i) 10)))))
      
      (destructuring-bind (pred-mean pred-cov)
          (predict dag-gp (loop for i from 149 upto 199
                                collect (list (aref exchange i 0))))
        (with-facets ((mean-array (pred-mean 'array :direction :input))
                      (cov-array (pred-cov 'array :direction :input)))
          (dotimes (i 51)
            (setf prediction (+ aud-mean
                                (* aud-stdev
                                   (aref mean-array (+ (* 5 51) i))))
                  var (* (expt aud-stdev 2)
                         (aref cov-array (+ (* 5 51) i) (+ (* 5 51) i))))
            (incf aud-cae (abs (- prediction
                                  (aref exchange-truth (+ 149 i) 9))))
            (incf aud-cse (expt (- prediction
                                   (aref exchange-truth (+ 149 i) 9))
                                2))
            (incf aud-cse-norm (expt (- aud-truth-mean
                                        (aref exchange-truth (+ 149 i) 9))
                                     2))
            (incf aud-cnll (* 0.5
                              (+ (/ (expt (- prediction
                                             (aref exchange-truth (+ 149 i) 9))
                                          2)
                                    var)
                                 (log (* 2 PI var)))))
            (format t "pred = ~a, truth = ~a~%"
                    (+ aud-mean (* aud-stdev (aref mean-array (+ (* 5 51) i))))
                    (aref exchange-truth (+ 149 i) 9)))))
      
      (format t "cad-mae = ~a~%" (/ cad-cae 51))
      (format t "hkd-mae = ~a~%" (/ hkd-cae 51))
      (format t "aud-mae = ~a~%" (/ aud-cae 51))
      (format t "cad-smse = ~a~%" (/ cad-cse cad-cse-norm))
      (format t "hkd-smse = ~a~%" (/ hkd-cse hkd-cse-norm))
      (format t "aud-smse = ~a~%" (/ aud-cse aud-cse-norm))
      (format t "average = ~a~%" (/ (+ (/ cad-cse cad-cse-norm)
                                       (/ hkd-cse hkd-cse-norm)
                                       (/ aud-cse aud-cse-norm))
                                    3))
      (format t "cad-mnll = ~a~%" (/ cad-cnll 51))    
      (format t "hkd-mnll = ~a~%" (/ hkd-cnll 51))
      (format t "aud-mnll = ~a~%" (/ aud-cnll 51))
      (format t "average = ~a~%" (/ (+ (/ cad-cnll 51)
                                       (/ hkd-cnll 51)
                                       (/ aud-cnll 51))
                                    3))
    
      dag-gp)))


;; Results:

;; Trial 1:
;; cad-mae = 0.014632362486878254d0
;; hkd-mae = 4.4398363512112526d-5
;; aud-mae = 0.004325905148611235d0
;; cad-smse = 0.6009600915558493d0
;; hkd-smse = 0.6352903741767465d0
;; aud-smse = 0.029431796087479888d0
;; average = 0.42189408727335853d0
;; cad-mnll = -2.827045989132631d0
;; hkd-mnll = -7.698941150565871d0
;; aud-mnll = -3.6686816901529435d0
;; average = -4.731556276617149d0

;; Trial 2:
;; cad-mae = 0.014632365091005628d0
;; hkd-mae = 4.4398360282911304d-5
;; aud-mae = 0.004325909854868997d0
;; cad-smse = 0.6009602842998284d0
;; hkd-smse = 0.6352905315070654d0
;; aud-smse = 0.02943184082192411d0
;; average = 0.4218942188762726d0
;; cad-mnll = -2.827046022633872d0
;; hkd-mnll = -7.6989407591087025d0
;; aud-mnll = -3.6686818658785247d0
;; average = -4.7315562158737d0

;; Trial 3:
;; cad-mae = 0.014632365397232094d0
;; hkd-mae = 4.4398360153023375d-5
;; aud-mae = 0.004325906098814898d0
;; cad-smse = 0.6009603114600593d0
;; hkd-smse = 0.6352905412846133d0
;; aud-smse = 0.029431804179673775d0
;; average = 0.42189421897478213d0
;; cad-mnll = -2.827045996810027d0
;; hkd-mnll = -7.698940679205318d0
;; aud-mnll = -3.6686819419248375d0
;; average = -4.731556205980061d0

;; Trial 4:
;; cad-mae = 0.014632367635414985d0
;; hkd-mae = 4.439836131172614d-5
;; aud-mae = 0.004325914957500114d0
;; cad-smse = 0.6009604833501668d0
;; hkd-smse = 0.63529051505511d0
;; aud-smse = 0.02943189535925227d0
;; average = 0.4218942979215097d0
;; cad-mnll = -2.827045971410316d0
;; hkd-mnll = -7.698940694150362d0
;; aud-mnll = -3.6686812378338183d0
;; average = -4.731555967798165d0

;; Trial 5:
;; cad-mae = 0.014632364424568715d0
;; hkd-mae = 4.439835998394185d-5
;; aud-mae = 0.004325905217854595d0
;; cad-smse = 0.6009602554375743d0
;; hkd-smse = 0.6352905543587485d0
;; aud-smse = 0.02943179461105399d0
;; average = 0.4218942014691256d0
;; cad-mnll = -2.8270459086442394d0
;; hkd-mnll = -7.698940784931685d0
;; aud-mnll = -3.668682028326343d0
;; average = -4.731556240634089d0

;; Trial 6:
;; cad-mae = 0.014632367709935324d0
;; hkd-mae = 4.439836392336417d-5
;; aud-mae = 0.004325896843518147d0
;; cad-smse = 0.6009605104440535d0
;; hkd-smse = 0.6352904517608424d0
;; aud-smse = 0.029431711152145844d0
;; average = 0.4218942244523472d0
;; cad-mnll = -2.827046014897484d0
;; hkd-mnll = -7.69894061147074d0
;; aud-mnll = -3.6686815239223676d0
;; average = -4.731556050096864d0

;; Trial 7:
;; cad-mae = 0.014632366787722935d0
;; hkd-mae = 4.4398360289129644d-5
;; aud-mae = 0.004325906917822805d0
;; cad-smse = 0.6009604085475719d0
;; hkd-smse = 0.6352906198287471d0
;; aud-smse = 0.029431811374455278d0
;; average = 0.4218942799169248d0
;; cad-mnll = -2.827045822536991d0
;; hkd-mnll = -7.698940352700186d0
;; aud-mnll = -3.668681359282767d0
;; average = -4.731555844839981d0

;; Trial 8:
;; cad-mae = 0.014632376975395522d0
;; hkd-mae = 4.439837619256834d-5
;; aud-mae = 0.0043259070287034775d0
;; cad-smse = 0.6009612442653485d0
;; hkd-smse = 0.635290175482052d0
;; aud-smse = 0.029431813022392207d0
;; average = 0.4218944109232643d0
;; cad-mnll = -2.8270451088500623d0
;; hkd-mnll = -7.698941038360025d0
;; aud-mnll = -3.6686815628835805d0
;; average = -4.731555903364556d0

;; Trial 9:
;; cad-mae = 0.014632364006713533d0
;; hkd-mae = 4.439836046916482d-5
;; aud-mae = 0.0040928287762904405d0
;; cad-smse = 0.6009601878940071d0
;; hkd-smse = 0.6352905335317119d0
;; aud-smse = 0.02715037494538284d0
;; average = 0.4211336987903673d0
;; cad-mnll = -2.8270461905113224d0
;; hkd-mnll = -7.6989406899120585d0
;; aud-mnll = -3.6783870737501694d0
;; average = -4.734791318057851d0

;; Trial 10:
;; cad-mae = 0.014632368839957565d0
;; hkd-mae = 4.439834528094813d-5
;; aud-mae = 0.00432590963307372d0
;; cad-smse = 0.600960583028561d0
;; hkd-smse = 0.6352909423761359d0
;; aud-smse = 0.029431842117016212d0
;; average = 0.421894455840571d0
;; cad-mnll = -2.82704566204375d0
;; hkd-mnll = -7.6989401530895645d0
;; aud-mnll = -3.668681304375625d0
;; average = -4.73155570650298d0


;; Averages:
;; cad-smse: 0.600960436028302 +/- 9.668361493356946e-08
;; hkd-smse: 0.6352905239361772 +/- 5.7861869523875287e-08
;; aud-smse: 0.029203668367077645 +/- 0.00021643613102310515
;; average: 0.42181820944385223 +/- 7.215376522112862e-05
;; cad-mnll: -2.82704586874707 +/- 9.021655901398994e-08
;; hkd-mnll: -7.698940691349452 +/- 8.692468232670391e-08
;; aud-mnll: -3.669652158833098 +/- 0.0009207408804492791
;; average: -4.73187957297654 +/- 0.0003069248860965737



(defun full-predict (dag-gp)
  (let ((exchange (read-exchange))
        (exchange-truth (read-exchange))
        mean-stdev
        cad-mean cad-stdev
        hkd-mean hkd-stdev
        aud-mean aud-stdev)

    (dotimes (i 251)
      ;; Block out CAD on days 50-100
      (when (and (>= i 49)
                 (<= i 99))
        (setf (aref exchange i 4) nil))

      ;; Block out HKD on days 100-150
      (when (and (>= i 99)
                 (<= i 149))
        (setf (aref exchange i 10) nil))
      
      ;; Block out AUD on days 150-200
      (when (and (>= i 149)
                 (<= i 199))
        (setf (aref exchange i 9) nil)))

    (setf mean-stdev (mean-and-stdev exchange 4)
          cad-mean (first mean-stdev)
          cad-stdev (second mean-stdev)
          mean-stdev (mean-and-stdev exchange 9)
          aud-mean (first mean-stdev)
          aud-stdev (second mean-stdev)
          mean-stdev (mean-and-stdev exchange 10)
          hkd-mean (first mean-stdev)
          hkd-stdev (second mean-stdev))

    (loop for i from 1 upto 365 by 0.2 do
      (destructuring-bind (mean cov) (predict dag-gp
                                              (list (list i)))
        (with-facets ((pred-array (mean 'array :direction :input))
                      (cov-array (cov 'array :direction :input)))
        (format t "i = ~a, pred = ~a, var = ~a~%"
                i
                (+ cad-mean (* cad-stdev (aref pred-array 2)))
                (* (expt cad-stdev 2) (aref cov-array 2 2))))))
    (loop for i from 1 upto 365 by 0.2 do
      (destructuring-bind (mean cov) (predict dag-gp
                                              (list (list i)))
        (with-facets ((pred-array (mean 'array :direction :input))
                      (cov-array (cov 'array :direction :input)))
        (format t "i = ~a, pred = ~a, var = ~a~%"
                i
                (+ hkd-mean (* hkd-stdev (aref pred-array 6)))
                (* (expt hkd-stdev 2) (aref cov-array 6 6))))))
    (loop for i from 1 upto 365 by 0.2 do
      (destructuring-bind (mean cov) (predict dag-gp
                                              (list (list i)))
        (with-facets ((pred-array (mean 'array :direction :input))
                      (cov-array (cov 'array :direction :input)))
          (format t "i = ~a, pred = ~a, var = ~a~%"
                  i
                  (+ aud-mean (* aud-stdev (aref pred-array 5)))
                  (* (expt aud-stdev 2) (aref cov-array 5 5))))))))


(defun mean-and-stdev (exchange col)
  (let ((count 0) (mean 0) (stdev 0))
    (dotimes (i 251)
      (when (aref exchange i col)
        (incf mean (aref exchange i col))
        (incf count)))
    (setf mean (/ mean count))
    (dotimes (i 251)
      (when (aref exchange i col)
        (incf stdev (expt (- (aref exchange i col) mean) 2))))
    (setf stdev (sqrt (/ stdev count)))

    (list mean stdev)))

