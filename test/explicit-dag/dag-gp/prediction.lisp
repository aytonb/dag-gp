(uiop:define-package #:dag-gp-test/explicit-dag/dag-gp/prediction
    (:use #:cl
          #:fiveam
          #:mgl-mat
          #:dag-gp/kernel
          #:dag-gp/output
          #:dag-gp/quadrature
          #:dag-gp/explicit-dag/parent-dependent-base/base
          #:dag-gp/explicit-dag/linear-parent/gp
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/networked-base/all
          #:dag-gp/explicit-dag/dag-gp/all
          #:dag-gp/explicit-dag/dag-gp/likelihood
          #:dag-gp-test/explicit-dag/dag-gp/construction
          #:dag-gp-test/utils/all)
  (:export ))

(in-package #:dag-gp-test/explicit-dag/dag-gp/prediction)


(def-suite explicit-dag/dag-gp/prediction)
(in-suite explicit-dag/dag-gp/prediction)


;; See dag_gp_test.m
(test dag-gp-predict
  (let ((gp (make-parameter-controlled-1d-dag-gp)))
    (destructuring-bind (mean cov) (predict gp '((0) (2) (5)))
      (with-facets ((mean-array (mean 'backing-array :direction :input))
                    (cov-array (cov 'array :direction :input)))
        (is (approximately-equal (aref mean-array 0) 1d0 1d-8))
        (is (approximately-equal (aref mean-array 1) 2d0 1d-8))
        (is (approximately-equal (aref mean-array 2) 0.164087930640108d0 1d-8))
        (is (approximately-equal (aref mean-array 3) 0.907532008711676d0 1d-8))
        (is (approximately-equal (aref mean-array 4) 1.5d0 1d-8))
        (is (approximately-equal (aref mean-array 5) 4d0 1d-8))
        (is (approximately-equal (aref mean-array 6) -0.041447748330703d0 1d-8))
        (is (approximately-equal (aref mean-array 7) -0.171209251981919d0 1d-8))
        (is (approximately-equal (aref mean-array 8) 2.5d0 1d-8))

        (is (approximately-equal (aref cov-array 0 0) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 1 0) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 1 1) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 2 0) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 2 1) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 2 2) 1.483170408489018d0 1d-8))
        (is (approximately-equal (aref cov-array 3 0) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 3 1) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 3 2) 0.002409284594968d0 1d-8))
        (is (approximately-equal (aref cov-array 3 3) 9.090057277323430d0 1d-8))
        (is (approximately-equal (aref cov-array 4 0) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 4 1) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 4 2) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 4 3) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 4 4) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 5 0) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 5 1) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 5 2) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 5 3) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 5 4) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 5 5) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 6 0) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 6 1) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 6 2) 0.001083126767770d0 1d-8))
        (is (approximately-equal (aref cov-array 6 3) 3.538035283175222d0 1d-8))
        (is (approximately-equal (aref cov-array 6 4) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 6 5) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 6 6) 2.381185869532273d0 1d-8))
        (is (approximately-equal (aref cov-array 7 0) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 7 1) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 7 2) 0.006003951470681d0 1d-8))
        (is (approximately-equal (aref cov-array 7 3) 0.084810158281693d0 1d-8))
        (is (approximately-equal (aref cov-array 7 4) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 7 5) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 7 6) 0.181237398150780d0 1d-8))
        (is (approximately-equal (aref cov-array 7 7) 1.004628983004166d0 1d-8))
        (is (approximately-equal (aref cov-array 8 0) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 8 1) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 8 2) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 8 3) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 8 4) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 8 5) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 8 6) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 8 7) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 8 8) 0d0 1d-8))))

    (destructuring-bind (mean cov) (predict gp '((1) (3) (5)))
      (with-facets ((mean-array (mean 'backing-array :direction :input))
                    (cov-array (cov 'array :direction :input)))
        (is (approximately-equal (aref mean-array 0) 1.538073905103534d0 1d-8))
        (is (approximately-equal (aref mean-array 1) 1.250855452037581d0 1d-8))
        (is (approximately-equal (aref mean-array 2) 0.164087930640108d0 1d-8))
        (is (approximately-equal (aref mean-array 3) 1.197884364614975d0 1d-8))
        (is (approximately-equal (aref mean-array 4) 1.594066507320916d0 1d-8))
        (is (approximately-equal (aref mean-array 5) 4d0 1d-8))
        (is (approximately-equal (aref mean-array 6) -0.137214248794251d0 1d-8))
        (is (approximately-equal (aref mean-array 7) 0.311087103952433d0 1d-8))
        (is (approximately-equal (aref mean-array 8) 2.5d0 1d-8))

        (is (approximately-equal (aref cov-array 0 0) 0.810174560968881d0 1d-8))
        (is (approximately-equal (aref cov-array 1 0) -0.236608133262479d0 1d-8))
        (is (approximately-equal (aref cov-array 1 1) 1.324685944325091d0 1d-8))
        (is (approximately-equal (aref cov-array 2 0) -0.034295205194453d0 1d-8))
        (is (approximately-equal (aref cov-array 2 1) 0.331861424100573d0 1d-8))
        (is (approximately-equal (aref cov-array 2 2) 1.483170408489018d0 1d-8))
        (is (approximately-equal (aref cov-array 3 0) 0.809977311287877d0 1d-8))
        (is (approximately-equal (aref cov-array 3 1) -0.234699424459883d0 1d-8))
        (is (approximately-equal (aref cov-array 3 2) -0.025764715632116d0 1d-8))
        (is (approximately-equal (aref cov-array 3 3) 6.931496241383472d0 1d-8))
        (is (approximately-equal (aref cov-array 4 0) -0.232558772102410d0 1d-8))
        (is (approximately-equal (aref cov-array 4 1) 1.285501843992157d0 1d-8))
        (is (approximately-equal (aref cov-array 4 2) 0.156738032598882d0 1d-8))
        (is (approximately-equal (aref cov-array 4 3) -1.951420221968869d0 1d-8))
        (is (approximately-equal (aref cov-array 4 4) 7.184440221777025d0 1d-8))
        (is (approximately-equal (aref cov-array 5 0) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 5 1) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 5 2) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 5 3) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 5 4) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 5 5) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 6 0) -0.000084803445179d0 1d-8))
        (is (approximately-equal (aref cov-array 6 1) 0.000820610109377d0 1d-8))
        (is (approximately-equal (aref cov-array 6 2) 0.003667508612771d0 1d-8))
        (is (approximately-equal (aref cov-array 6 3) 2.338635379316212d0 1d-8))
        (is (approximately-equal (aref cov-array 6 4) -0.639778310921796d0 1d-8))
        (is (approximately-equal (aref cov-array 6 5) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 6 6) 1.897042153387893d0 1d-8))
        (is (approximately-equal (aref cov-array 7 0) -0.000084216557553d0 1d-8))
        (is (approximately-equal (aref cov-array 7 1) 0.000814931025021d0 1d-8))
        (is (approximately-equal (aref cov-array 7 2) 0.003642127386594d0 1d-8))
        (is (approximately-equal (aref cov-array 7 3) -0.637902511947001d0 1d-8))
        (is (approximately-equal (aref cov-array 7 4) 2.236374603679867d0 1d-8))
        (is (approximately-equal (aref cov-array 7 5) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 7 6) -0.089655258369123d0 1d-8))
        (is (approximately-equal (aref cov-array 7 7) 1.825448731995402d0 1d-8))
        (is (approximately-equal (aref cov-array 8 0) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 8 1) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 8 2) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 8 3) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 8 4) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 8 5) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 8 6) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 8 7) 0d0 1d-8))
        (is (approximately-equal (aref cov-array 8 8) 0d0 1d-8))))))


;; See variational_combined_dag_gp_test.m
(test n-ary-dag-gp-predict
  (let ((gp (make-parameter-controlled-1d-n-ary-dag-gp)))
    (destructuring-bind (pred-0 pred-2 pred-5)
        (predict gp '((0) (2) (5)))
      (flatten pred-0 '(0 1 2))
      (is (member '(1.5d0 0 0) (flattened pred-0) :test #'equalp))
      (is (approximately-equal (nth (position '(1.5d0 0 0) (flattened pred-0)
                                              :test #'equalp)
                                    (weights pred-0))
                               0.026512678388842d0
                               1d-8))
      (is (member '(1.5d0 1 0) (flattened pred-0) :test #'equalp))
      (is (approximately-equal (nth (position '(1.5d0 1 0) (flattened pred-0)
                                              :test #'equalp)
                                    (weights pred-0))
                               0.223420824660339d0
                               1d-8))
      (is (member '(1.5d0 2 0) (flattened pred-0) :test #'equalp))
      (is (approximately-equal (nth (position '(1.5d0 2 0) (flattened pred-0)
                                              :test #'equalp)
                                    (weights pred-0))
                               0.004004797823031d0
                               1d-8))
      (is (member '(1.5d0 0 1) (flattened pred-0) :test #'equalp))
      (is (approximately-equal (nth (position '(1.5d0 0 1) (flattened pred-0)
                                              :test #'equalp)
                                    (weights pred-0))
                               0.187705563966579d0
                               1d-8))
      (is (member '(1.5d0 1 1) (flattened pred-0) :test #'equalp))
      (is (approximately-equal (nth (position '(1.5d0 1 1) (flattened pred-0)
                                              :test #'equalp)
                                    (weights pred-0))
                               0.340882361071074d0
                               1d-8))
      (is (member '(1.5d0 2 1) (flattened pred-0) :test #'equalp))
      (is (approximately-equal (nth (position '(1.5d0 2 1) (flattened pred-0)
                                              :test #'equalp)
                                    (weights pred-0))
                               0.099939685260180d0
                               1d-8))
      (is (member '(1.5d0 0 2) (flattened pred-0) :test #'equalp))
      (is (approximately-equal (nth (position '(1.5d0 0 2) (flattened pred-0)
                                              :test #'equalp)
                                    (weights pred-0))
                               0.004250690858861d0
                               1d-8))
      (is (member '(1.5d0 1 2) (flattened pred-0) :test #'equalp))
      (is (approximately-equal (nth (position '(1.5d0 1 2) (flattened pred-0)
                                              :test #'equalp)
                                    (weights pred-0))
                               0.109232124882201d0
                               1d-8))
      (is (member '(1.5d0 2 2) (flattened pred-0) :test #'equalp))
      (is (approximately-equal (nth (position '(1.5d0 2 2) (flattened pred-0)
                                              :test #'equalp)
                                    (weights pred-0))
                               0.004051273088894d0
                               1d-8)))))


(test n-ary-dag-gp-combined-distributions
  (let ((gp (make-parameter-controlled-1d-n-ary-dag-gp))
        index-gp
        loc-0-index loc-2-index loc-5-index
        pred)
    (update-combined-distributions gp)
    (setf loc-0-index (position '(0) (constituent-locs gp) :test #'equalp)
          loc-2-index (position '(2) (constituent-locs gp) :test #'equalp)
          loc-5-index (position '(5) (constituent-locs gp) :test #'equalp))

    ;; GP 0
    (setf index-gp (nth 0 (constituent-gps gp))
          pred (nth loc-0-index (var-parent-distributions index-gp)))
    (is (member '(1.5d0) (flattened pred) :test #'equalp))
    (is (approximately-equal (nth (position '(1.5d0) (flattened pred)
                                            :test #'equalp)
                                  (weights pred))
                             1d0
                             1d-8))

    ;; GP 1
    (setf index-gp (nth 1 (constituent-gps gp))
          pred (nth loc-0-index (var-parent-distributions index-gp)))  
    (is (member '(0 1.5d0 0) (flattened pred) :test #'equalp))
    (is (approximately-equal (nth (position '(0 1.5d0 0) (flattened pred)
                                            :test #'equalp)
                                  (weights pred))
                             0.026512678388842d0
                             1d-8))
    (is (member '(1 1.5d0 0) (flattened pred) :test #'equalp))
    (is (approximately-equal (nth (position '(1 1.5d0 0) (flattened pred)
                                            :test #'equalp)
                                  (weights pred))
                             0.223420824660339d0
                             1d-8))
    (is (member '(2 1.5d0 0) (flattened pred) :test #'equalp))
    (is (approximately-equal (nth (position '(2 1.5d0 0) (flattened pred)
                                            :test #'equalp)
                                  (weights pred))
                             0.004004797823031d0
                             1d-8))
    (is (member '(0 1.5d0 1) (flattened pred) :test #'equalp))
    (is (approximately-equal (nth (position '(0 1.5d0 1) (flattened pred)
                                            :test #'equalp)
                                  (weights pred))
                             0.187705563966579d0
                             1d-8))
    (is (member '(1 1.5d0 1) (flattened pred) :test #'equalp))
    (is (approximately-equal (nth (position '(1 1.5d0 1) (flattened pred)
                                            :test #'equalp)
                                  (weights pred))
                             0.340882361071074d0
                             1d-8))
    (is (member '(2 1.5d0 1) (flattened pred) :test #'equalp))
    (is (approximately-equal (nth (position '(2 1.5d0 1) (flattened pred)
                                            :test #'equalp)
                                  (weights pred))
                             0.099939685260180d0
                             1d-8))
    (is (member '(0 1.5d0 2) (flattened pred) :test #'equalp))
    (is (approximately-equal (nth (position '(0 1.5d0 2) (flattened pred)
                                            :test #'equalp)
                                  (weights pred))
                             0.004250690858861d0
                             1d-8))
    (is (member '(1 1.5d0 2) (flattened pred) :test #'equalp))
    (is (approximately-equal (nth (position '(1 1.5d0 2) (flattened pred)
                                            :test #'equalp)
                                  (weights pred))
                             0.109232124882201d0
                             1d-8))
    (is (member '(2 1.5d0 2) (flattened pred) :test #'equalp))
    (is (approximately-equal (nth (position '(2 1.5d0 2) (flattened pred)
                                            :test #'equalp)
                                  (weights pred))
                             0.004051273088894d0
                             1d-8))
    
    ;; GP 2
    (setf index-gp (nth 2 (constituent-gps gp))
          pred (nth loc-0-index (var-parent-distributions index-gp)))
    (is (member '(0) (flattened pred) :test #'equalp))
    (is (approximately-equal (nth (position '(0) (flattened pred)
                                            :test #'equalp)
                                  (weights pred))
                             0.253938300872212d0
                             1d-8))
    (is (member '(1) (flattened pred) :test #'equalp))
    (is (approximately-equal (nth (position '(1) (flattened pred)
                                            :test #'equalp)
                                  (weights pred))
                             0.628527610297832d0
                             1d-8))
    (is (member '(2) (flattened pred) :test #'equalp))
    (is (approximately-equal (nth (position '(2) (flattened pred)
                                            :test #'equalp)
                                  (weights pred))
                             0.117534088829956d0
                             1d-8))))

