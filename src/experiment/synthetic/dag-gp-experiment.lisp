(uiop:define-package #:dag-gp/experiment/synthetic/dag-gp-experiment
    (:use #:cl
          #:mgl-mat
          #:dag-gp/kernel
          #:dag-gp/output
          #:dag-gp/explicit-dag/dag-gp/all
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/networked-base/all
          #:dag-gp/utils/distance)
  (:import-from #:array-operations)
  (:export ))

(in-package #:dag-gp/experiment/synthetic/dag-gp-experiment)


;; Run this test by entering this package in a lisp process using
;; `(in-package :dag-gp/experiment/synthetic/dag-gp-experiment)`
;; and running
;; `(synthetic-test nil)`


(defun make-synthetic-data ()
  (let ((datT (make-array
               '(4 41)
               :element-type t
               :initial-contents
   (list (list 0d0    0.25d0  0.5d0   0.75d0  1d0    1.25d0  1.5d0   1.75d0  2d0  
               2.25d0  2.5d0   2.75d0  3d0    3.25d0  3.5d0   3.75d0  4d0    4.25d0
               4.5d0   4.75d0  5d0    5.25d0  5.5d0   5.75d0  6d0    6.25d0  6.5d0 
               6.75d0  7d0    7.25d0  7.5d0   7.75d0  8d0    8.25d0  8.5d0   8.75d0
               9d0    9.25d0  9.5d0   9.75d0 10d0)
         (list 1.831564d-02 4.677062d-02 1.053992d-01 2.096114d-01 3.678794d-01 5.697828d-01 7.788008d-01 9.394131d-01 1.000000d00 9.394131d-01 7.788008d-01 5.697828d-01 3.678794d-01 2.096114d-01 1.053992d-01 4.677062d-02 1.831564d-02 6.329715d-03 1.930454d-03 5.195747d-04 1.234098d-04 2.586810d-05 4.785117d-06 7.811489d-07 1.125352d-07 1.430724d-08 1.605228d-09 1.589391d-10 1.388794d-11 1.070923d-12 7.287724d-14 4.376619d-15 2.319523d-16 1.084855d-17 4.477732d-19 1.631014d-20 5.242886d-22 1.487292d-23 3.723363d-25 8.225981d-27 1.603811d-28)
(list 1.510425d00 1.693480d00 1.867413d00 2.099768d00 2.447516d00 2.926411d00 3.501289d00 4.097623d00 4.624334d00 4.997418d00 5.158373d00 5.085090d00 4.794323d00 4.335891d00 3.780355d00 3.203652d00 2.672848d00 2.236445d00 1.920733d00 1.731688d00 1.660425d00 1.689835d00 1.800739d00 1.980012d00 2.252180d00 2.775637d00 3.858196d00 5.415512d00 6.431518d00 6.147785d00 5.322615d00 4.971123d00 5.174975d00 5.620957d00 6.141623d00 6.699234d00 7.290001d00 7.914531d00 8.573750d00 9.268594d00 1.000000d01)
(list -1.072404d00 -4.609765d-01 -1.934721d-02 4.192228d-01 9.142737d-01 1.463938d00 2.005294d00 2.433344d00 2.648744d00 2.611859d00 2.366882d00 2.018455d00 1.678683d00 1.421433d00 1.268203d00 1.202577d00 1.194166d00 1.215921d00 1.250377d00 1.288589d00 1.326829d00 1.363850d00 1.399369d00 1.433409d00 1.466065d00 1.497442d00 1.527634d00 1.556728d00 1.584801d00 1.611922d00 1.638154d00 1.663553d00 1.688171d00 1.712053d00 1.735243d00 1.757780d00 1.779699d00 1.801033d00 1.821814d00 1.842068d00 1.861822d00)))))
    (aops:permute '(1 0) datT)))


(defun synthetic-test (dag)
  (let ((dag-gp (if dag
                    dag
                    (make-instance 'dag-gp
                                   :output-dim 3
                                   :ref-kernel (make-instance 'rbf-kernel)
                                   :outputs (list (make-instance 'gaussian-output)
                                                  (make-instance 'gaussian-output)
                                                  (make-instance 'gaussian-output)))))
        (dat (make-synthetic-data))
        (dat-truth (make-synthetic-data))
        mean-stdev
        mean-0 stdev-0
        mean-1 stdev-1
        mean-2 stdev-2
        (cae-2 0) (cse-2 0))
      
      (dotimes (i 41)
        ;; Block out y1 from 6-8 (24-32)
        (when (and (>= i 24)
                   (<= i 32))
          (setf (aref dat i 3) nil)))
      
      
      (setf mean-stdev (mean-and-stdev dat 1)
            mean-0 (first mean-stdev)
            stdev-0 (second mean-stdev)
            mean-stdev (mean-and-stdev dat 2)
            mean-1 (first mean-stdev)
            stdev-1 (second mean-stdev)
            mean-stdev (mean-and-stdev dat 3)
            mean-2 (first mean-stdev)
            stdev-2 (second mean-stdev))
      
      (unless dag
        (dotimes (i 41)
          (when (aref dat i 1)
            (add-measurement dag-gp
                             (list (aref dat i 0))
                             0
                             (/ (- (aref dat i 1) mean-0) stdev-0)))
          (when (aref dat i 2)
            (add-measurement dag-gp
                             (list (aref dat i 0))
                             1
                             (/ (- (aref dat i 2) mean-1) stdev-1)))
          (when (aref dat i 3)
            (add-measurement dag-gp
                             (list (aref dat i 0))
                             2
                             (/ (- (aref dat i 3) mean-2) stdev-2))))
        
        (initialize-gp dag-gp)
        
        (configure-for-factor dag-gp 0 '())
        (configure-for-factor dag-gp 1 '())
        (configure-for-factor dag-gp 2 '())
        
        (time
         (train dag-gp :progress-fn :summary
                       :search :abc
                       :relative-tol 0.005)))


    (loop for i from 24 upto 32 do
      (with-facets ((pred-array
                     ((first (predict dag-gp
                                      (list (list (aref dat i 0)))))
                      'array :direction :input)))
        (incf cae-2 (abs (- (+ mean-2 (* stdev-2 (aref pred-array 2)))
                            (aref dat-truth i 3))))
        (incf cse-2 (expt (- (+ mean-2 (* stdev-2 (aref pred-array 2)))
                             (aref dat-truth i 3))
                          2))
        (format t "pred = ~a, truth = ~a~%"
                (+ mean-2 (* stdev-2 (aref pred-array 2)))
                (aref dat-truth i 3))))
      
    (format t "cae-2 = ~a~%" cae-2)
    (format t "cse-2 = ~a~%" cse-2)

    ;; (with-open-file (f (asdf:system-relative-pathname
    ;;                     :gaussian-process
    ;;                     "./src/experiment/synthetic/full_predict.txt")
    ;;                    :direction :output
    ;;                    :if-does-not-exist :create
    ;;                    :if-exists :supersede)
    ;;   (loop for i from 0.001 upto 10.05001 by 0.05 do
    ;;     (destructuring-bind (mean cov) (predict dag-gp
    ;;                                             (list (list i)))
    ;;       (with-facets ((pred-array (mean 'array :direction :input))
    ;;                     (cov-array (cov 'array :direction :input)))
    ;;         (format f "i = ~a, pred = ~a, var = ~a~%"
    ;;                 i
    ;;                 (+ mean-0 (* stdev-0 (aref pred-array 0)))
    ;;                 (* (expt stdev-0 2) (aref cov-array 0 0))))))
    ;;   (loop for i from 0.001 upto 10.05001 by 0.05 do
    ;;     (destructuring-bind (mean cov) (predict dag-gp
    ;;                                             (list (list i)))
    ;;       (with-facets ((pred-array (mean 'array :direction :input))
    ;;                     (cov-array (cov 'array :direction :input)))
    ;;         (format f "i = ~a, pred = ~a, var = ~a~%"
    ;;                 i
    ;;                 (+ mean-1 (* stdev-1 (aref pred-array 1)))
    ;;                 (* (expt stdev-1 2) (aref cov-array 1 1))))))
    ;;   (loop for i from 0.001 upto 10.05001 by 0.05 do
    ;;     (destructuring-bind (mean cov) (predict dag-gp
    ;;                                             (list (list i)))
    ;;       (with-facets ((pred-array (mean 'array :direction :input))
    ;;                     (cov-array (cov 'array :direction :input)))
    ;;         (format f "i = ~a, pred = ~a, var = ~a~%"
    ;;                 i
    ;;                 (+ mean-2 (* stdev-2 (aref pred-array 2)))
    ;;                 (* (expt stdev-2 2) (aref cov-array 2 2)))))))
    
    dag-gp))


(defun mean-and-stdev (exchange col)
  (let ((count 0) (mean 0) (stdev 0))
    (dotimes (i 41)
      (when (aref exchange i col)
        (incf mean (aref exchange i col))
        (incf count)))
    (setf mean (/ mean count))
    (dotimes (i 41)
      (when (aref exchange i col)
        (incf stdev (expt (- (aref exchange i col) mean) 2))))
    (setf stdev (sqrt (/ stdev count)))

    (list mean stdev)))


;; Typical A*BC search:
;; Training 0 <- (1 2): 130.00959479988154d0
;; Training 1 <- (0 2): 85.29163205349307d0
;; Training 2 <- (0 1): 116.52636429697876d0
;; Training 2 <- (0): 116.66486738050975d0
;; Training 1 <- NIL: 82.94880985458846d0
;; Training 0 <- NIL: 129.81315310483996d0
;; Training 2 <- NIL: 109.86083574394713d0
;; Training 1 <- (0): 82.90093421407701d0
;; Training 1 <- (2): 84.87184727888557d0
;; Evaluated 9 likelihoods
;; New best DAG = ((1) (2 0) (0)): 327.4268303399382d0

;; Results:

;; Trial 1:
;; pred = 1.4625189225694952d0, truth = 1.466065d0
;; pred = 1.4890249856414348d0, truth = 1.497442d0
;; pred = 1.514392288169552d0, truth = 1.527634d0
;; pred = 1.5406937823982245d0, truth = 1.556728d0
;; pred = 1.5690945317018645d0, truth = 1.584801d0
;; pred = 1.5993097617241865d0, truth = 1.611922d0
;; pred = 1.6299669544637783d0, truth = 1.638154d0
;; pred = 1.6594915987245904d0, truth = 1.663553d0
;; pred = 1.686884650326213d0, truth = 1.688171d0
;; cae-2 = 0.08309252428066083d0
;; cse-2 = 0.001006798953298686d0

;; Trial 2:
;; pred = 1.462214762615275d0, truth = 1.466065d0
;; pred = 1.4883916503029886d0, truth = 1.497442d0
;; pred = 1.5134671668124693d0, truth = 1.527634d0
;; pred = 1.5396348961433113d0, truth = 1.556728d0
;; pred = 1.568120618862707d0, truth = 1.584801d0
;; pred = 1.5985956981439344d0, truth = 1.611922d0
;; pred = 1.6295634799248924d0, truth = 1.638154d0
;; pred = 1.659327650256598d0, truth = 1.663553d0
;; pred = 1.6868421516498993d0, truth = 1.688171d0
;; cae-2 = 0.08831192528792475d0
;; cse-2 = 0.0011388484091725871d0

;; Trial 3:
;; pred = 1.4652699998372563d0, truth = 1.466065d0
;; pred = 1.4982470010562445d0, truth = 1.497442d0
;; pred = 1.5326410867716391d0, truth = 1.527634d0
;; pred = 1.5671913753723214d0, truth = 1.556728d0
;; pred = 1.5993724602476895d0, truth = 1.584801d0
;; pred = 1.6271147824728605d0, truth = 1.611922d0
;; pred = 1.65026049744067d0, truth = 1.638154d0
;; pred = 1.6706386184660438d0, truth = 1.663553d0
;; pred = 1.6907611512111802d0, truth = 1.688171d0
;; cae-2 = 0.06861697320139282d0
;; cse-2 = 7.824634397218649d-4

;; Trial 4:
;; pred = 1.4652673834432226d0, truth = 1.466065d0
;; pred = 1.4982235157406403d0, truth = 1.497442d0
;; pred = 1.5326264652240686d0, truth = 1.527634d0
;; pred = 1.567214547979527d0, truth = 1.556728d0
;; pred = 1.5994298810881824d0, truth = 1.584801d0
;; pred = 1.6271774934879017d0, truth = 1.611922d0
;; pred = 1.6503053779845458d0, truth = 1.638154d0
;; pred = 1.6706646885493794d0, truth = 1.663553d0
;; pred = 1.6907778461453151d0, truth = 1.688171d0
;; cae-2 = 0.06881243275633775d0
;; cse-2 = 7.87901347788586d-4

;; Trial 5:
;; pred = 1.4622247301863982d0, truth = 1.466065d0
;; pred = 1.4884101679430009d0, truth = 1.497442d0
;; pred = 1.5135387471599455d0, truth = 1.527634d0
;; pred = 1.5397835515773264d0, truth = 1.556728d0
;; pred = 1.568321118254234d0, truth = 1.584801d0
;; pred = 1.598793409985571d0, truth = 1.611922d0
;; pred = 1.6297148606425953d0, truth = 1.638154d0
;; pred = 1.6594200202154008d0, truth = 1.663553d0
;; pred = 1.6868850725211053d0, truth = 1.688171d0
;; cae-2 = 0.0873783215144226d0
;; cse-2 = 0.0011160127301167787d0

;; Trial 6:
;; pred = 1.4652825391835638d0, truth = 1.466065d0
;; pred = 1.4982602662498907d0, truth = 1.497442d0
;; pred = 1.5326871609418524d0, truth = 1.527634d0
;; pred = 1.5672913855279798d0, truth = 1.556728d0
;; pred = 1.599506817792396d0, truth = 1.584801d0
;; pred = 1.627238330686d0, truth = 1.611922d0
;; pred = 1.6503423213412864d0, truth = 1.638154d0
;; pred = 1.6706808104902826d0, truth = 1.663553d0
;; pred = 1.6907822381274227d0, truth = 1.688171d0
;; cae-2 = 0.0691667919735468d0
;; cse-2 = 7.954318405899661d-4

;; Trial 7:
;; pred = 1.4622013804853482d0, truth = 1.466065d0
;; pred = 1.488286098097608d0, truth = 1.497442d0
;; pred = 1.5132249901786803d0, truth = 1.527634d0
;; pred = 1.5392545508228828d0, truth = 1.556728d0
;; pred = 1.567648763350235d0, truth = 1.584801d0
;; pred = 1.5981169607153523d0, truth = 1.611922d0
;; pred = 1.6291724465261985d0, truth = 1.638154d0
;; pred = 1.6590870381230178d0, truth = 1.663553d0
;; pred = 1.6867505363574744d0, truth = 1.688171d0
;; cae-2 = 0.0907272353432027d0
;; cse-2 = 0.001199108252565766d0

;; Trial 8:
;; pred = 1.465246418488785d0, truth = 1.466065d0
;; pred = 1.4981894093566033d0, truth = 1.497442d0
;; pred = 1.5325635949469374d0, truth = 1.527634d0
;; pred = 1.5671193930277951d0, truth = 1.556728d0
;; pred = 1.5993240489119012d0, truth = 1.584801d0
;; pred = 1.627092523534204d0, truth = 1.611922d0
;; pred = 1.6502567357402858d0, truth = 1.638154d0
;; pred = 1.670644004086624d0, truth = 1.663553d0
;; pred = 1.6907683814036893d0, truth = 1.688171d0
;; cae-2 = 0.06837167251925491d0
;; cse-2 = 7.780793273470693d-4

;; Trial 9:
;; pred = 1.4625491816735243d0, truth = 1.466065d0
;; pred = 1.4891723177669483d0, truth = 1.497442d0
;; pred = 1.5146760814013764d0, truth = 1.527634d0
;; pred = 1.541085861757256d0, truth = 1.556728d0
;; pred = 1.5695403792501592d0, truth = 1.584801d0
;; pred = 1.5997412487464182d0, truth = 1.611922d0
;; pred = 1.6303120269565237d0, truth = 1.638154d0
;; pred = 1.6596998626900874d0, truth = 1.663553d0
;; pred = 1.6869585139248362d0, truth = 1.688171d0
;; cae-2 = 0.08073452583287022d0
;; cse-2 = 9.524033435523182d-4

;; Trial 10:
;; pred = 1.4652732779744988d0, truth = 1.466065d0
;; pred = 1.4982224651007612d0, truth = 1.497442d0
;; pred = 1.5326019462955738d0, truth = 1.527634d0
;; pred = 1.5671581989071346d0, truth = 1.556728d0
;; pred = 1.5993492906194429d0, truth = 1.584801d0
;; pred = 1.627092533386255d0, truth = 1.611922d0
;; pred = 1.650236678569123d0, truth = 1.638154d0
;; pred = 1.6706230980511723d0, truth = 1.663553d0
;; pred = 1.690761333655656d0, truth = 1.688171d0
;; cae-2 = 0.06843226661061963d0
;; cse-2 = 7.79190568654067d-4


;; Average mae = 0.008596051881335923 +/- 0.00031824519032049866
