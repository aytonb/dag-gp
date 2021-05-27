(uiop:define-package #:dag-gp-test/explicit-dag/linear-parent/computation
    (:use #:cl
          #:fiveam
          #:mgl-mat
          #:dag-gp/quadrature
          #:dag-gp/kernel
          #:dag-gp/output
          #:dag-gp/utils/distance
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/parent-dependent-base/base
          #:dag-gp/explicit-dag/parent-dependent-base/likelihood
          #:dag-gp/explicit-dag/linear-parent/all
          #:dag-gp/explicit-dag/linear-parent/gp
          #:dag-gp-test/explicit-dag/linear-parent/construction
          #:dag-gp-test/utils/all)
  (:export #:make-parameter-controlled-1d-linear-parent-gp
           #:make-parameter-controlled-1d-variational-linear-parent-gp)
  (:documentation "Tests that inference quantities are computed appropriately."))

(in-package #:dag-gp-test/explicit-dag/linear-parent/computation)


(def-suite explicit-dag/linear-parent/computation)
(in-suite explicit-dag/linear-parent/computation)


(defun make-parameter-controlled-1d-linear-parent-gp ()
  (let ((gp (make-uninitialized-1d-linear-parent-gp)))
    (initialize-measurements gp '((0) (2) (5)))
    (initialize-gp gp :initialize-latent nil)

    (setf (var-parent-distributions gp)
          (list (make-mat 9
                          :ctype :double
                          :initial-contents (list 1d0 0.3d0 0.6d0
                                                  2.2d0 0.5d0 0.7d0
                                                  1.7d0 0.4d0 0.8d0))
                (make-mat (list 9 9)
                          :ctype :double
                          :initial-contents
                          (list (list 0d0 0d0 0d0 0d0 0d0 0d0 0d0 0d0 0d0)
                                (list 0d0 1d0 0.5d0 0.4d0 0.3d0 0.1d0 0.2d0 0.2d0 0.3d0)
                                (list 0d0 0.5d0 1d0 0.1d0 0.14d0 0.2d0 0.2d0 0.3d0 0.2d0)
                                (list 0d0 0.4d0 0.1d0 0.7d0 0.25d0 0.1d0 0.05d0 0.2d0 0.3d0)
                                (list 0d0 0.3d0 0.14d0 0.25d0 0.8d0 0.3d0 0.03d0 0.1d0 0.1d0)
                                (list 0d0 0.1d0 0.2d0 0.1d0 0.3d0 0.4d0 0.13d0 0.1d0 0.24d0)
                                (list 0d0 0.2d0 0.2d0 0.05d0 0.03d0 0.13d0 1.2d0 0.6d0 0.16d0)
                                (list 0d0 0.2d0 0.3d0 0.2d0 0.1d0 0.1d0 0.6d0 0.9d0 0.3d0)
                                (list 0d0 0.3d0 0.2d0 0.3d0 0.1d0 0.24d0 0.16d0 0.3d0 0.4d0)))))
    
    (setf (subseq (kern-params (kernel gp)) 0 2)
          (list 1.25d0 0.7d0)
          (subseq (output-params (output gp)) 0 1)
          (list -1d0)
          (subseq (first (parent-params gp)) 0 1)
          (list 1d0)
          (subseq (second (parent-params gp)) 0 1)
          (list 1.5d0))
    gp))


(defun make-parameter-controlled-1d-variational-linear-parent-gp ()
  (let ((gp (make-uninitialized-1d-variational-linear-parent-gp)))
    (initialize-measurements gp '((0) (2) (5)))
    (setf (u-locs gp) (make-array (list 2 1)
                                  :element-type 'double-float
                                  :initial-contents (list (list 1d0)
                                                          (list 3d0))))
    (initialize-gp gp :initialize-latent nil)

    (setf (var-parent-distributions gp)
          (list (list (make-mat 3
                                :ctype :double
                                :initial-contents (list 1d0 0.3d0 0.6d0))
                      (make-mat (list 3 3)
                                :ctype :double
                                :initial-contents (list (list 0d0 0d0 0d0)
                                                        (list 0d0 1d0 0.5d0)
                                                        (list 0d0 0.5d0 1d0))))
                (list (make-mat 3
                                :ctype :double
                                :initial-contents (list 2.2d0 0.5d0 0.7d0))
                      (make-mat (list 3 3)
                                :ctype :double
                                :initial-contents (list (list 0.7d0 0.25d0 0.1d0)
                                                        (list 0.25d0 0.8d0 0.3d0)
                                                        (list 0.1d0 0.3d0 0.4d0))))
                (list (make-mat 3
                                :ctype :double
                                :initial-contents (list 1.7d0 0.4d0 0.8d0))
                      (make-mat (list 3 3)
                                :ctype :double
                                :initial-contents (list (list 1.2d0 0.6d0 0.5d0)
                                                        (list 0.6d0 0.9d0 0.3d0)
                                                        (list 0.5d0 0.3d0 0.7d0))))))
    (setf (subseq (kern-params (kernel gp)) 0 2)
          (list 1.25d0 0.7d0)
          (subseq (output-params (output gp)) 0 1)
          (list -1d0)
          (subseq (first (parent-params gp)) 0 1)
          (list 1d0)
          (subseq (second (parent-params gp)) 0 1)
          (list 1.5d0)
          (subseq (q-mean-params gp) 0 2)
          (list 1.3d0 0.6d0)
          (subseq (q-chol gp) 0 3)
          (list 1d0 0.2d0 1.2d0))
    gp))


(defun make-parameter-controlled-1d-variational-combined-output-linear-parent-gp ()
  (let ((gp (make-uninitialized-1d-variational-combined-output-linear-parent-gp))
        (quad-set-0 (make-instance 'quad-set))
        (quad-set-1 (make-instance 'quad-set))
        (quad-set-2 (make-instance 'quad-set)))
    (initialize-measurements gp '((0) (2) (5)))
    (setf (u-locs gp) (make-array (list 2 1)
                                  :element-type 'double-float
                                  :initial-contents (list (list 1d0)
                                                          (list 3d0))))
    (initialize-gp gp :initialize-latent nil)

    (flet ((prob-1 (vars) (declare (ignore vars)) '((0 0.4d0) (1 0.1d0) (2 0.5d0)))
           (prob-0-0 (vars)
             (list (list 0 (* 0.1d0 (first vars)))
                   (list 1 (* 0.2d0 (first vars)))
                   (list 2 (- 1 (* 0.3d0 (first vars))))))
           (prob-0-1 (vars)
             (list (list 0 (+ 0.1d0 (* 0.1d0 (first vars))))
                   (list 1 (+ 0.1d0 (* 0.2d0 (first vars))))
                   (list 2 (- 1 (+ 0.2d0 (* 0.3d0 (first vars)))))))
           (prob-2-0 (vars) (declare (ignore vars)) '((2 1d0)))
           (prob-2-1 (vars) (if (> (first vars) (second vars))
                                '((0 0.5d0) (1 0.5d0))
                                '((2 1d0)))))

      (add-quad-variable quad-set-0 1 #'prob-1 nil)
      (add-quad-variable quad-set-0 0 #'prob-0-0 '(1))
      (add-quad-variable quad-set-0 2 #'prob-2-0 '(0 1))
      (flatten quad-set-0 '(2 0 1))
      (add-quad-variable quad-set-1 1 #'prob-1 nil)
      (add-quad-variable quad-set-1 0 #'prob-0-1 '(1))
      (add-quad-variable quad-set-1 2 #'prob-2-1 '(0 1))
      (flatten quad-set-1 '(2 0 1))
      (add-quad-variable quad-set-2 1 #'prob-1 nil)
      (add-quad-variable quad-set-2 0 #'prob-0-1 '(1))
      (add-quad-variable quad-set-2 2 #'prob-2-1 '(0 1))
      (flatten quad-set-2 '(2 0 1))

      (setf (var-parent-distributions gp)
            (list quad-set-0 quad-set-1 quad-set-2))

      (setf (subseq (kern-params (first (kernel gp))) 0 2)
            (list 1.25d0 0.7d0)
            (subseq (kern-params (second (kernel gp))) 0 2)
            (list 0.3d0 1.2d0)
            (subseq (first (first (parent-params gp))) 0 3)
            (list 1d0 2d0 3d0)
            (subseq (second (first (parent-params gp))) 0 3)
            (list -1d0 -2d0 -3d0)
            (subseq (first (second (parent-params gp))) 0 3)
            (list 0d0 1d0 1.5d0)
            (subseq (second (second (parent-params gp))) 0 3)
            (list 3d0 0d0 2.3d0)
            (subseq (first (q-mean-params gp)) 0 2)
            (list 1.3d0 0.6d0)
            (subseq (second (q-mean-params gp)) 0 2)
            (list 0.8d0 0.5d0)
            (subseq (first (q-chol gp)) 0 3)
            (list 1d0 0.2d0 1.2d0)
            (subseq (second (q-chol gp)) 0 3)
            (list 0.6d0 1.4d0 0.3d0))

      gp)))


;; See linear_parent_test.m
(test NLL-and-derivs
  (let* ((gp (make-parameter-controlled-1d-linear-parent-gp))
         (deriv-array (make-array (n-gp-params gp) :element-type 'double-float))
         NLL)
    (setf NLL (NLL-and-derivs gp deriv-array))

    (is (approximately-equal NLL 5.144395496406011d0 1d-8))
    (is (approximately-equal (aref deriv-array 0) -1.478661615296085d0 1d-8))
    (is (approximately-equal (aref deriv-array 1) -0.150071278042643d0 1d-8))
    (is (approximately-equal (aref deriv-array 2) -0.148232814627982d0 1d-8))
    (is (approximately-equal (aref deriv-array 3) 2.896918830544349d0 1d-8))
    (is (approximately-equal (aref deriv-array 4) 2.858197038575005d0 1d-8))))


;; See variational_linear_parent_test.m
(test variational-NLL-and-derivs
  (let* ((gp (make-parameter-controlled-1d-variational-linear-parent-gp))
         (deriv-array (make-array (n-gp-params gp) :element-type 'double-float))
         NLL)
    (setf NLL (NLL-and-derivs gp deriv-array :latent nil))
    (is (approximately-equal NLL 24.954797810692298d0 1d-8))
    (is (approximately-equal (aref deriv-array 0) 9.646046955655638d0 1d-8))
    (is (approximately-equal (aref deriv-array 1) -2.163835139099699d0 1d-8))
    (is (approximately-equal (aref deriv-array 2) -23.274497133632050d0 1d-8))
    (is (approximately-equal (aref deriv-array 3) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 4) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 5) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 6) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 7) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 8) 10.751475570936964d0 1d-8))
    (is (approximately-equal (aref deriv-array 9) 12.097554434464307d0 1d-8))

    (setf NLL (NLL-and-derivs gp deriv-array :latent t))
    (is (approximately-equal NLL 24.954797810692298d0 1d-8))
    (is (approximately-equal (aref deriv-array 0) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 1) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 2) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 3) 2.506156953762778d0 1d-8))
    (is (approximately-equal (aref deriv-array 4) 0.449710865749293d0 1d-8))
    (is (approximately-equal (aref deriv-array 5) 1.233551288509050d0 1d-8))
    (is (approximately-equal (aref deriv-array 6) 0.835532592625003d0 1d-8))
    (is (approximately-equal (aref deriv-array 7) 0.532657818352579d0 1d-8))
    (is (approximately-equal (aref deriv-array 8) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 9) 0d0 1d-8))))


;; See variational_combined_linear_parent_test.m
(test variational-combined-NLL-and-derivs
  (let* ((gp (make-parameter-controlled-1d-variational-combined-output-linear-parent-gp))
         (deriv-array (make-array (n-gp-params gp) :element-type 'double-float))
         NLL)
    (setf NLL (NLL-and-derivs gp deriv-array :latent nil))
    (is (approximately-equal NLL 14.415420130012601d0 1d-8))
    (is (approximately-equal (aref deriv-array 0) 0.633303391266318d0 1d-8))
    (is (approximately-equal (aref deriv-array 1) -0.135712895319180d0 1d-8))
    (is (approximately-equal (aref deriv-array 2) 0.058738433152710d0 1d-8))
    (is (approximately-equal (aref deriv-array 3) -0.083650660615283d0 1d-8))
    (is (approximately-equal (aref deriv-array 4) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 5) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 6) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 7) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 8) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 9) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 10) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 11) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 12) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 13) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 14) 0.045417442737164d0 1d-8))
    (is (approximately-equal (aref deriv-array 15) 0.036534785798730d0 1d-8))
    (is (approximately-equal (aref deriv-array 16) -0.076856612395052d0 1d-8))
    (is (approximately-equal (aref deriv-array 17) -0.147867321445463d0 1d-8))
    (is (approximately-equal (aref deriv-array 18) 0.060713320606997d0 1d-8))
    (is (approximately-equal (aref deriv-array 19) 0.092249616979308d0 1d-8))
    (is (approximately-equal (aref deriv-array 20) 0.432622917513602d0 1d-8))
    (is (approximately-equal (aref deriv-array 21) 0.705807933709369d0 1d-8))
    (is (approximately-equal (aref deriv-array 22) 0.916547112630078d0 1d-8))
    (is (approximately-equal (aref deriv-array 23) 0.614026888235313d0 1d-8))
    (is (approximately-equal (aref deriv-array 24) 0.100065830717570d0 1d-8))
    (is (approximately-equal (aref deriv-array 25) 1.340885244900166d0 1d-8))
    
    (setf NLL (NLL-and-derivs gp deriv-array :latent t))
    (is (approximately-equal NLL 14.415420130012601d0 1d-8))
    (is (approximately-equal (aref deriv-array 0) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 1) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 2) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 3) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 4) 0.410187302166711d0 1d-8))
    (is (approximately-equal (aref deriv-array 5) 0.068711064033404d0 1d-8))
    (is (approximately-equal (aref deriv-array 6) 1.485406161249288d0 1d-8))
    (is (approximately-equal (aref deriv-array 7) 0.619591294602293d0 1d-8))
    (is (approximately-equal (aref deriv-array 8) -0.659181941998642d0 1d-8))
    (is (approximately-equal (aref deriv-array 9) 0.041984020714339d0 1d-8))
    (is (approximately-equal (aref deriv-array 10) -0.453677701477015d0 1d-8))
    (is (approximately-equal (aref deriv-array 11) -1.443560172968802d0 1d-8))
    (is (approximately-equal (aref deriv-array 12) 1.064760302772132d0 1d-8))
    (is (approximately-equal (aref deriv-array 13) -3.075756712449586d0 1d-8))
    (is (approximately-equal (aref deriv-array 14) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 15) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 16) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 17) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 18) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 19) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 20) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 21) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 22) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 23) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 24) 0d0 1d-8))
    (is (approximately-equal (aref deriv-array 25) 0d0 1d-8))))
