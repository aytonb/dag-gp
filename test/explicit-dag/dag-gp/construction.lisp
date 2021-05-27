(uiop:define-package #:dag-gp-test/explicit-dag/dag-gp/construction
    (:use #:cl
          #:fiveam
          #:mgl-mat
          #:dag-gp/kernel
          #:dag-gp/output
          #:dag-gp/explicit-dag/parent-dependent-base/base
          #:dag-gp/explicit-dag/linear-parent/gp
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/networked-base/all
          #:dag-gp/explicit-dag/dag-gp/all
          #:dag-gp/explicit-dag/dag-gp/likelihood)
  (:export #:make-parameter-controlled-1d-dag-gp
           #:make-parameter-controlled-1d-variational-dag-gp
           #:make-parameter-controlled-1d-n-ary-dag-gp)
  (:documentation "Tests that data containers are made and appropriately sized."))

(in-package #:dag-gp-test/explicit-dag/dag-gp/construction)


(defun make-empty-1d-dag-gp ()
  (make-instance 'dag-gp
                 :ref-kernel (make-instance 'rbf-kernel)
                 :output-dim 3
                 :outputs (list (make-instance 'gaussian-output)
                                (make-instance 'gaussian-output)
                                (make-instance 'gaussian-output))))

(defun make-empty-1d-variational-dag-gp ()
  (make-instance 'variational-dag-gp
                 :ref-kernel (make-instance 'rbf-kernel)
                 :output-dim 3
                 :n-latent 2
                 :outputs (list (make-instance 'gaussian-output)
                                (make-instance 'gaussian-output)
                                (make-instance 'gaussian-output))
                 :input-dim 1))

(defun make-empty-1d-n-ary-dag-gp ()
  (make-instance 'variational-dag-gp
                 :ref-kernel (make-instance 'rbf-kernel)
                 :output-dim 3
                 :n-latent 2
                 :outputs (list (make-instance 'gaussian-output)
                                (make-instance 'n-ary-output
                                               :n-params 3)
                                (make-instance 'n-ary-output
                                               :n-params 3))
                 :input-dim 1))


(defun make-uninitialized-1d-dag-gp ()
  (let ((gp (make-empty-1d-dag-gp)))

    (add-measurement gp '(0) 0 1d0)
    (add-measurement gp '(2) 0 2d0)
    (add-measurement gp '(2) 1 1.5d0)
    (add-measurement gp '(5) 1 4d0)
    (add-measurement gp '(5) 2 2.5d0)
    
    gp))

(defun make-uninitialized-1d-variational-dag-gp ()
  (let ((gp (make-empty-1d-variational-dag-gp)))

   
    (add-measurement gp '(2) 1 1.5d0)
    (add-measurement gp '(5) 1 4d0)
    (add-measurement gp '(5) 2 2.5d0)
    (add-measurement gp '(0) 0 1d0)
    (add-measurement gp '(2) 0 2d0)
    
    gp))

(defun make-uninitialized-1d-n-ary-dag-gp ()
  (let ((gp (make-empty-1d-n-ary-dag-gp)))

   
    (add-measurement gp '(2) 1 1)
    (add-measurement gp '(5) 1 0)
    (add-measurement gp '(5) 2 2)
    (add-measurement gp '(0) 0 1.5d0)
    (add-measurement gp '(2) 0 4d0)
    
    gp))


(defun make-parameter-controlled-1d-dag-gp ()
  (let ((gp (make-uninitialized-1d-dag-gp)))
    (initialize-gp gp :initialize-latent nil)

    
    (configure-for-factor gp 0 nil)
    (configure-for-factor gp 1 (list 0 2))
    (configure-for-factor gp 2 nil)

    ;; kernel params
    (setf (subseq (kern-params (kernel (nth 0 (constituent-gps gp)))) 0 2)
          (list 0.75d0 1.2d0)
          (subseq (kern-params (kernel (nth 1 (constituent-gps gp)))) 0 2)
          (list 1.25d0 0.7d0)
          (subseq (kern-params (kernel (nth 2 (constituent-gps gp)))) 0 2)
          (list 0.8d0 0.9d0))
    ;; output params
    (setf (subseq (output-params (nth 0 (outputs gp))) 0 1)
          (list -1.2d0)
          (subseq (output-params (nth 1 (outputs gp))) 0 1)
          (list -1.0d0)
          (subseq (output-params (nth 2 (outputs gp))) 0 1)
          (list -1.6d0))
    ;; parent params
    (setf (subseq (first (parent-params (nth 1 (constituent-gps gp)))) 0 1)
          (list 1d0)
          (subseq (second (parent-params (nth 1 (constituent-gps gp)))) 0 1)
          (list 1.5d0))

    gp))

(defun make-parameter-controlled-1d-variational-dag-gp ()
  (let ((gp (make-uninitialized-1d-variational-dag-gp)))
    (initialize-gp gp :initialize-latent nil)

    ;; u-locs
    (setf (u-locs (nth 0 (constituent-gps gp)))
          (make-array (list 2 1)
                      :element-type 'double-float
                      :initial-contents (list (list 1d0)
                                              (list 3d0)))
          (u-locs (nth 1 (constituent-gps gp)))
          (make-array (list 2 1)
                      :element-type 'double-float
                      :initial-contents (list (list 1d0)
                                              (list 3d0)))
          (u-locs (nth 2 (constituent-gps gp)))
          (make-array (list 2 1)
                      :element-type 'double-float
                      :initial-contents (list (list 1d0)
                                              (list 3d0))))
    
    (configure-for-factor gp 0 nil)
    (configure-for-factor gp 1 (list 0 2))
    (configure-for-factor gp 2 nil)

    ;; kernel params
    (setf (subseq (kern-params (kernel (nth 0 (constituent-gps gp)))) 0 2)
          (list 0.75d0 1.2d0)
          (subseq (kern-params (kernel (nth 1 (constituent-gps gp)))) 0 2)
          (list 1.25d0 0.7d0)
          (subseq (kern-params (kernel (nth 2 (constituent-gps gp)))) 0 2)
          (list 0.8d0 0.9d0))
    ;; output params
    (setf (subseq (output-params (nth 0 (outputs gp))) 0 1)
          (list -1.2d0)
          (subseq (output-params (nth 1 (outputs gp))) 0 1)
          (list -1.0d0)
          (subseq (output-params (nth 2 (outputs gp))) 0 1)
          (list -1.6d0))
    ;; parent params
    (setf (subseq (first (parent-params (nth 1 (constituent-gps gp)))) 0 1)
          (list 1d0)
          (subseq (second (parent-params (nth 1 (constituent-gps gp)))) 0 1)
          (list 1.5d0))
    ;; q mean params
    (setf (subseq (q-mean-params (nth 0 (constituent-gps gp))) 0 2)
          (list 0.3d0 0.5d0)
          (subseq (q-mean-params (nth 1 (constituent-gps gp))) 0 2)
          (list 1.3d0 0.6d0)
          (subseq (q-mean-params (nth 2 (constituent-gps gp))) 0 2)
          (list 0.9d0 0.5d0))
    ;; q cholesky params
    (setf (subseq (q-chol (nth 0 (constituent-gps gp))) 0 3)
          (list 0.7d0 0.2d0 0.8d0)
          (subseq (q-chol (nth 1 (constituent-gps gp))) 0 3)
          (list 1.0d0 0.2d0 1.2d0)
          (subseq (q-chol (nth 2 (constituent-gps gp))) 0 3)
          (list 0.4d0 0.1d0 0.7d0))

    gp))

(defun make-parameter-controlled-1d-n-ary-dag-gp ()
  (let ((gp (make-uninitialized-1d-n-ary-dag-gp)))
    (initialize-gp gp :initialize-latent nil)

    ;; u-locs
    (setf (u-locs (nth 0 (constituent-gps gp)))
          (make-array (list 2 1)
                      :element-type 'double-float
                      :initial-contents (list (list 1d0)
                                              (list 3d0)))
          (u-locs (nth 1 (constituent-gps gp)))
          (make-array (list 2 1)
                      :element-type 'double-float
                      :initial-contents (list (list 1d0)
                                              (list 3d0)))
          (u-locs (nth 2 (constituent-gps gp)))
          (make-array (list 2 1)
                      :element-type 'double-float
                      :initial-contents (list (list 1d0)
                                              (list 3d0))))
    
    (configure-for-factor gp 0 nil)
    (configure-for-factor gp 1 (list 0 2))
    (configure-for-factor gp 2 nil)

    ;; kernel params
    (setf (subseq (kern-params (kernel (nth 0 (constituent-gps gp)))) 0 2)
          (list 1.25d0 0.7d0)
          (subseq (kern-params (first (kernel (nth 1 (constituent-gps gp))))) 0 2)
          (list 0.75d0 1.2d0)
          (subseq (kern-params (second (kernel (nth 1 (constituent-gps gp))))) 0 2)
          (list 0.3d0 1.1d0)
          (subseq (kern-params (first (kernel (nth 2 (constituent-gps gp))))) 0 2)
          (list 0.8d0 0.9d0)
          (subseq (kern-params (second (kernel (nth 2 (constituent-gps gp))))) 0 2)
          (list 0.4d0 1.6d0))
    ;; output params
    (setf (subseq (output-params (nth 0 (outputs gp))) 0 1)
          (list -1.2d0))
    ;; parent params
    (setf (subseq (first (first (parent-params (nth 1 (constituent-gps gp))))) 0 1)
          (list 1d0)
          (subseq (second (first (parent-params (nth 1 (constituent-gps gp))))) 0 3)
          (list -1d0 -2d0 -3d0)
          (subseq (first (second (parent-params (nth 1 (constituent-gps gp))))) 0 1)
          (list 0.5d0)
          (subseq (second (second (parent-params (nth 1 (constituent-gps gp))))) 0 3)
          (list 3d0 0d0 2.3d0))
    ;; q mean params
    (setf (subseq (q-mean-params (nth 0 (constituent-gps gp))) 0 2)
          (list 0.3d0 0.5d0)
          (subseq (first (q-mean-params (nth 1 (constituent-gps gp)))) 0 2)
          (list 1.3d0 0.6d0)
          (subseq (second (q-mean-params (nth 1 (constituent-gps gp)))) 0 2)
          (list 0.8d0 0.5d0)
          (subseq (first (q-mean-params (nth 2 (constituent-gps gp)))) 0 2)
          (list 0.9d0 0.5d0)
          (subseq (second (q-mean-params (nth 2 (constituent-gps gp)))) 0 2)
          (list 2.3d0 1.3d0))
    ;; q cholesky params
    (setf (subseq (q-chol (nth 0 (constituent-gps gp))) 0 3)
          (list 0.7d0 0.2d0 0.8d0)
          (subseq (first (q-chol (nth 1 (constituent-gps gp)))) 0 3)
          (list 1d0 0.2d0 1.2d0)
          (subseq (second (q-chol (nth 1 (constituent-gps gp)))) 0 3)
          (list 0.6d0 1.4d0 0.3d0)
          (subseq (first (q-chol (nth 2 (constituent-gps gp)))) 0 3)
          (list 0.4d0 0.1d0 0.7d0)
          (subseq (second (q-chol (nth 2 (constituent-gps gp)))) 0 3)
          (list 0.8d0 0.3d0 1.1d0))

    gp))

  
(defun quick-test ()
  (let ((gp (make-parameter-controlled-1d-dag-gp)))

    ;(format t "~a~%" (predict gp '((0) (2) (5))))
    (format t "~a~%" (predict gp '((1) (3) (5))))
    ;(format t "~a~%" (predict gp '((0) (2) (5))))

    (update-combined-distributions gp)
    gp))

(defun variational-quick-test ()
  (let ((gp (make-parameter-controlled-1d-variational-dag-gp)))

    (format t "~a~%" (predict gp '((0) (2) (5))))

    (update-combined-distributions gp)
    gp))

(defun n-ary-quick-test ()
  (let ((gp (make-parameter-controlled-1d-n-ary-dag-gp))
        deriv-array)
    ;(initialize-gp gp)
    ;(configure-for-factor gp 0 nil)
    ;(configure-for-factor gp 1 '(0 2))
    (update-combined-distributions gp)

    (setf deriv-array (make-array (n-gp-params (second (constituent-gps gp)))
                                  :initial-element 0d0))
    (NLL-and-derivs (second (constituent-gps gp))
                    deriv-array)

    deriv-array))


(defun train-test ()
  (let ((gp (make-parameter-controlled-1d-dag-gp)))
    (train gp :progress-fn :summary)))

(defun variational-train-test ()
  (let ((gp (make-parameter-controlled-1d-variational-dag-gp)))
    
    ;(sb-sprof:start-profiling :mode :alloc)
    (train gp :progress-fn :summary)
    ;(sb-sprof:stop-profiling)
    ;(sb-sprof:report)
    ))


(defun indep-test ()
  (let ((gp (make-instance 'variational-dag-gp
                           :ref-kernel (make-instance 'rbf-kernel)
                           :input-dim 1
                           :output-dim 2
                           :n-latent 4
                           :outputs (list (make-instance 'gaussian-output)
                                          (make-instance 'gaussian-output)))))

    (add-measurement gp '(0) 0 0.39602430806649497d0)
    (add-measurement gp '(1) 0 0.576075194540262d0)
    (add-measurement gp '(2) 0 0.8605058336538315d0)
    (add-measurement gp '(3) 0 1.1020876546444685d0)
    (add-measurement gp '(4) 0 1.0145221693939772d0)
    (add-measurement gp '(5) 0 0.4857589771781318d0)
    (add-measurement gp '(6) 0 -0.1854250993417656d0)
    (add-measurement gp '(7) 0 -0.516328687260278d0)
    (add-measurement gp '(8) 0 -0.330946207755883d0)
    (add-measurement gp '(9) 0 0.09094119308837945d0)

    (add-measurement gp '(0) 1 -0.6803741564203488d0)
    (add-measurement gp '(1) 1 -0.6888347055911036d0)
    (add-measurement gp '(2) 1 -0.6458775339535814d0)
    (add-measurement gp '(3) 1 -0.49323207783527495d0)
    (add-measurement gp '(4) 1 -0.20593944405417663d0)
    (add-measurement gp '(5) 1 0.11645231454727809d0)
    (add-measurement gp '(6) 1 0.36736764487457313d0)
    (add-measurement gp '(7) 1 0.5607027407904557d0)
    (add-measurement gp '(8) 1 0.7423728961515472d0)
    (add-measurement gp '(9) 1 0.8555210552776938d0)

    (initialize-gp gp)
    (configure-for-factor gp 0 nil)
    (configure-for-factor gp 1 (list 0))

    (train gp :progress-fn :summary)))


(defun indep-test-2 ()
  (let ((gp (make-instance 'variational-dag-gp
                           :ref-kernel (make-instance 'rbf-kernel)
                           :input-dim 1
                           :output-dim 2
                           :n-latent 10
                           :outputs (list (make-instance 'gaussian-output)
                                          (make-instance 'gaussian-output))))
        prediction-1
        prediction-2
        (err-1 0) (err-2 0)
        truth)

    (setf truth #(-2.254633796275825d0 -1.4014369726625058d0 -0.6870250767420066d0
                  -0.2912761166124818d0 -0.12351964657074149d0))

    (add-measurement gp '(0) 0 0.001893593203854736d0)
    (add-measurement gp '(1) 0 -0.2625758279212894d0)
    (add-measurement gp '(2) 0 -0.7066093372868161d0)
    (add-measurement gp '(3) 0 -1.2630488538850977d0)
    (add-measurement gp '(4) 0 -1.6701655756385987d0)
    (add-measurement gp '(5) 0 -1.60981258338776d0)
    (add-measurement gp '(6) 0 -1.0360794791885934d0)
    (add-measurement gp '(7) 0 -0.253316868491572d0)
    (add-measurement gp '(8) 0 0.3959418280435591d0)
    (add-measurement gp '(9) 0 0.8036121366496164d0)
    (add-measurement gp '(10) 0 1.0460817494910364d0)
    (add-measurement gp '(11) 0 1.2511455085907084d0)
    (add-measurement gp '(12) 0 1.574644173202915d0)
    (add-measurement gp '(13) 0 2.127683527101249d0)
    (add-measurement gp '(14) 0 2.813981830366088d0)
    (add-measurement gp '(15) 0 3.3244059616635635d0)
    (add-measurement gp '(16) 0 3.3702079744846882d0)
    (add-measurement gp '(17) 0 2.9105113541021668d0)
    (add-measurement gp '(18) 0 2.1534092617511513d0)
    (add-measurement gp '(19) 0 1.3771221711555355d0)

    (add-measurement gp '(0) 1 3.0790514553784702d0)
    (add-measurement gp '(1) 1 3.56807830567054d0)
    (add-measurement gp '(2) 1 3.7704049574622776d0)
    (add-measurement gp '(3) 1 3.559119652668231d0)
    (add-measurement gp '(4) 1 2.8106190834393914d0)
    (add-measurement gp '(5) 1 1.6665653350571041d0)
    (add-measurement gp '(6) 1 0.5381393414880771d0)
    (add-measurement gp '(7) 1 -0.2176843901348512d0)
    (add-measurement gp '(8) 1 -0.5711261740581299d0)
    (add-measurement gp '(9) 1 -0.777859424922538d0)
    (add-measurement gp '(10) 1 -1.124563057838631d0)
    (add-measurement gp '(11) 1 -1.713127472106352d0)
    (add-measurement gp '(12) 1 -2.4018911409134835d0)
    (add-measurement gp '(13) 1 -2.8785984487390994d0)
    (add-measurement gp '(14) 1 -2.8447066378958525d0)
    ;; (add-measurement gp '(15) 1 -2.254633796275825d0)
    ;; (add-measurement gp '(16) 1 -1.4014369726625058d0)
    ;; (add-measurement gp '(17) 1 -0.6870250767420066d0)
    ;; (add-measurement gp '(18) 1 -0.2912761166124818d0)
    ;; (add-measurement gp '(19) 1 -0.12351964657074149d0)

    (initialize-gp gp)
    (configure-for-factor gp 0 nil)
    (configure-for-factor gp 1 (list 0))

    (train gp :progress-fn :summary)

    (setf prediction-1 (predict gp '((15) (16) (17) (18) (19))))


    (setf gp (make-instance 'variational-dag-gp
                            :ref-kernel (make-instance 'rbf-kernel)
                            :input-dim 1
                            :output-dim 2
                            :n-latent 10
                            :outputs (list (make-instance 'gaussian-output)
                                           (make-instance 'gaussian-output))))

    (add-measurement gp '(0) 0 0.001893593203854736d0)
    (add-measurement gp '(1) 0 -0.2625758279212894d0)
    (add-measurement gp '(2) 0 -0.7066093372868161d0)
    (add-measurement gp '(3) 0 -1.2630488538850977d0)
    (add-measurement gp '(4) 0 -1.6701655756385987d0)
    (add-measurement gp '(5) 0 -1.60981258338776d0)
    (add-measurement gp '(6) 0 -1.0360794791885934d0)
    (add-measurement gp '(7) 0 -0.253316868491572d0)
    (add-measurement gp '(8) 0 0.3959418280435591d0)
    (add-measurement gp '(9) 0 0.8036121366496164d0)
    (add-measurement gp '(10) 0 1.0460817494910364d0)
    (add-measurement gp '(11) 0 1.2511455085907084d0)
    (add-measurement gp '(12) 0 1.574644173202915d0)
    (add-measurement gp '(13) 0 2.127683527101249d0)
    (add-measurement gp '(14) 0 2.813981830366088d0)
    (add-measurement gp '(15) 0 3.3244059616635635d0)
    (add-measurement gp '(16) 0 3.3702079744846882d0)
    (add-measurement gp '(17) 0 2.9105113541021668d0)
    (add-measurement gp '(18) 0 2.1534092617511513d0)
    (add-measurement gp '(19) 0 1.3771221711555355d0)

    (add-measurement gp '(0) 1 3.0790514553784702d0)
    (add-measurement gp '(1) 1 3.56807830567054d0)
    (add-measurement gp '(2) 1 3.7704049574622776d0)
    (add-measurement gp '(3) 1 3.559119652668231d0)
    (add-measurement gp '(4) 1 2.8106190834393914d0)
    (add-measurement gp '(5) 1 1.6665653350571041d0)
    (add-measurement gp '(6) 1 0.5381393414880771d0)
    (add-measurement gp '(7) 1 -0.2176843901348512d0)
    (add-measurement gp '(8) 1 -0.5711261740581299d0)
    (add-measurement gp '(9) 1 -0.777859424922538d0)
    (add-measurement gp '(10) 1 -1.124563057838631d0)
    (add-measurement gp '(11) 1 -1.713127472106352d0)
    (add-measurement gp '(12) 1 -2.4018911409134835d0)
    (add-measurement gp '(13) 1 -2.8785984487390994d0)
    (add-measurement gp '(14) 1 -2.8447066378958525d0)

    (initialize-gp gp)
    (configure-for-factor gp 0 nil)
    (configure-for-factor gp 1 (list 0))

    (EM-on-structure gp :progress-fn :summary)
    (setf prediction-2 (predict gp '((15) (16) (17) (18) (19))))
    
    
    (format t "Prediction 1 = ~a~%" prediction-1)
    (format t "Prediction 2 = ~a~%" prediction-2)

    (loop for pred in prediction-1
          for i from 0
          do (with-facets ((mu ((first pred) 'backing-array :direction :input)))
               (incf err-1 (expt (- (aref mu 1) (aref truth i)) 2))))
    (setf err-1 (/ err-1 5))

    (loop for pred in prediction-2
          for i from 0
          do (with-facets ((mu ((first pred) 'backing-array :direction :input)))
               (incf err-2 (expt (- (aref mu 1) (aref truth i)) 2))))
    (setf err-2 (/ err-2 5))

    (format t "Err 1 = ~a~%" err-1)
    (format t "Err 2 = ~a~%" err-2)))

