(uiop:define-package #:dag-gp/explicit-dag/linear-parent/likelihood
    (:use #:cl
          #:mgl-mat
          #:dag-gp/kernel
          #:dag-gp/output
          #:dag-gp/lapack
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/linear-parent/gp)
  (:export #:negative-log-likelihood
           #:NLL-and-derivs
           #:dLL/dKff))

(in-package #:dag-gp/explicit-dag/linear-parent/likelihood)


(defmethod negative-log-likelihood ((gp linear-parent-gp))
  (when (closed-downwards-p gp)
    (return-from negative-log-likelihood
      (closed-downwards-negative-log-likelihood gp)))

  (let ((NLL 0d0)
        (n-parents (list-length (parent-gps gp)))
        var-parent-dim
        a-mat
        parent-mat
        y-cov
        temp-mat)

    (setf (obs-mat gp) (make-array (n-obs gp) :element-type 'double-float)
          var-parent-dim (* (1+ n-parents) (n-obs gp))
          y-cov (make-mat (list (n-obs gp) (n-obs gp))
                          :ctype :double)
          temp-mat (make-mat (list (n-obs gp) var-parent-dim)
                             :ctype :double))
    
    (destructuring-bind (var-parent-mean var-parent-cov)
        (var-parent-distributions gp)
      (with-facets ((vp-mean (var-parent-mean 'backing-array
                                              :direction :input)))
        ;; Set obs-mat = y - p_ij y_j
        (loop for i below (n-obs gp) do
          (setf (aref (obs-mat gp) i) (aref vp-mean i))
          (loop for p below n-parents
                for parent-param in (parent-params gp)
                for offset = (* (1+ p) (n-obs gp))
                do (incf (aref (obs-mat gp) i)
                         (- (* (aref parent-param 0)
                               (aref vp-mean (+ offset i))))))))
      
      ;; Make the y covariance
      ;; (setf a-mat (make-array (list (n-obs gp) (n-obs gp))
      ;;                         :element-type 'double-float
      ;;                         :initial-element 0d0))
      ;; (dotimes (i (n-obs gp))
      ;;   (setf (aref a-mat i i) 1d0))
      ;; (setf a-mat (list (array-to-mat a-mat :ctype :double)))
      ;; (loop for p below n-parents
      ;;       for parent-param in (parent-params gp)
      ;;       do (setf parent-mat (make-array (list (n-obs gp) (n-obs gp))
      ;;                                       :element-type 'double-float
      ;;                                       :initial-element 0d0))
      ;;          (dotimes (i (n-obs gp))
      ;;            (setf (aref parent-mat i i) (- (aref parent-param 0))))
      ;;          (setf a-mat (nconc a-mat (list (array-to-mat parent-mat
      ;;                                                       :ctype :double)))))
      ;; ;; This call to stack consumes 31% of computation time???
      ;; (setf a-mat (stack 1 a-mat)
      ;;       (a-mat gp) a-mat)

      (setf a-mat (make-array (list (n-obs gp) (* (1+ n-parents) (n-obs gp)))
                              :element-type 'double-float
                              :initial-element 0d0))
      (dotimes (i (n-obs gp))
        (setf (aref a-mat i i) 1d0))
      (loop for p below n-parents
            for parent-params in (parent-params gp)
            for param = (- (aref parent-params 0))
            for offset = (* (1+ p) (n-obs gp))
            do (dotimes (i (n-obs gp))
                 (setf (aref a-mat i (+ offset i)) param)))
      (setf a-mat (array-to-mat a-mat)
            (a-mat gp) a-mat)
      
      ;; y-cov <- a var-parent-cov aT
      (gemm! 1d0 a-mat var-parent-cov 0d0 temp-mat
             :m (n-obs gp) :n var-parent-dim :k var-parent-dim
             :lda var-parent-dim :ldb var-parent-dim :ldc var-parent-dim)
      (gemm! 1d0 temp-mat a-mat 0d0 y-cov
             :transpose-b? t :m (n-obs gp) :n (n-obs gp) :k var-parent-dim
             :lda var-parent-dim :ldb var-parent-dim :ldc (n-obs gp))
      
      (setf (obs-mat gp) (array-to-mat (obs-mat gp) :ctype :double)
            (obs-mat-copy gp) (copy-mat (obs-mat gp)))
      ;; Cholesky factorize the covariance
      (potrf! (Kff gp) :uplo #\L :n (n-obs gp) :lda (n-obs gp))
      ;; y <- Kff^-1 y
      (potrs! (Kff gp) (obs-mat-copy gp)
              :uplo #\L :n (n-obs gp) :nrhs 1 :transpose-b? t
              :lda (n-obs gp) :ldb (n-obs gp))
      (setf NLL (dot (obs-mat gp) (obs-mat-copy gp) :n (n-obs gp)))
      
      ;; Log det part of expression
      (with-facets ((Kff ((Kff gp) 'array :direction :input)))
        (dotimes (i (n-obs gp))
          (incf NLL (* 2 (log (aref Kff i i))))))
      
      ;; Expectation with respect to y = -1/2 tr(y-cov Kff^-1)
      ;; y-cov <- y-cov Kff^-1
      (potrs! (Kff gp) y-cov :uplo #\L :n (n-obs gp) :nrhs (n-obs gp) 
                             :lda (n-obs gp) :ldb (n-obs gp) :transpose-b? t)
      (setf (y-cov gp) y-cov)
      (with-facets ((cov-array (y-cov 'array :direction :input)))
        (dotimes (i (n-obs gp))
          (incf NLL (aref cov-array i i))))
      
      (* NLL 0.5d0))))


(defgeneric closed-downwards-negative-log-likelihood (gp)
  (:method ((gp linear-parent-gp))
    (let ((NLL 0d0)
          (n-parents (list-length (parent-gps gp))))

      (setf (obs-mat gp) (copy-mat (first (var-parent-distributions gp))))

      (loop for p below n-parents
            for parent-param in (parent-params gp)
            for parent-obs in (rest (var-parent-distributions gp))
            do (axpy! (- (aref parent-param 0)) parent-obs (obs-mat gp)
                      :n (n-true-obs gp)))
           
      (setf (obs-mat-copy gp) (copy-mat (obs-mat gp)))
      ;; Cholesky factorize the covariance
      (potrf! (Kff gp) :uplo #\L :n (n-true-obs gp) :lda (n-true-obs gp))
      ;; y <- Kff^-1 y
      (potrs! (Kff gp) (obs-mat-copy gp)
              :uplo #\L :n (n-true-obs gp) :nrhs 1 :transpose-b? t
              :lda (n-true-obs gp) :ldb (n-true-obs gp))
      (setf NLL (dot (obs-mat gp) (obs-mat-copy gp) :n (n-true-obs gp)))
        
      ;; Log det part of expression
      (with-facets ((Kff ((Kff gp) 'array :direction :input)))
        (dotimes (i (n-true-obs gp))
          (incf NLL (* 2 (log (aref Kff i i))))))
        
        (* NLL 0.5d0))))


(defmethod negative-log-likelihood ((gp variational-linear-parent-gp))
  (with-facets ((qf-mean-array ((qf-mean gp) 'backing-array :direction :input)))
    (let ((qf-mean-list (coerce qf-mean-array 'list))
          (qf-cov-list (coerce (Kff gp) 'list))
          NLL)

      ;; TODO: Move this sum into the LL function
      (setf NLL (- (apply #'+ (LL (output gp) qf-mean-list qf-cov-list
                                  (var-parent-distributions gp)
                                  (list-length (parent-gps gp))
                                  (parent-outputs gp)
                                  (parent-params gp)))))

      (+ NLL (post-prior-kl gp)))))


(defmethod negative-log-likelihood
    ((gp variational-combined-output-linear-parent-gp))
  (let ((qf-mean-list nil)
        (qf-cov-list nil)
        NLL)

    (setf NLL (- (apply #'+ (LL (output gp)
                                (reshaped-qf-mean gp)
                                (reshaped-Kff gp)
                                (var-parent-distributions gp)
                                (list-length (parent-gps gp))
                                (parent-outputs gp)
                                (parent-params gp)))))
    
    (+ NLL (post-prior-kl gp))))


;; TODO: Make a specialized 'exact Gaussian' output to handle this and noise and things
(defgeneric dLL/dparent (gp param)
  (:documentation "Computes the derivative of log likelihood with respect to parents for exact gaussian processes.")
  (:method ((gp linear-parent-gp) param)
    (when (closed-downwards-p gp)
      (return-from dLL/dparent (closed-downwards-dLL/dparent gp param)))

    (let (dLL
          (n-parents (list-length (parent-gps gp)))
          var-parent-dim
          a-deriv
          offset
          (dy (make-array (n-obs gp) :element-type 'double-float))
          temp-mat
          d-cov)
      (setf var-parent-dim (* (1+ n-parents) (n-obs gp))
            offset (* (1+ param) (n-obs gp)))

      (if (a-deriv-mats-updated-p gp)

          (setf a-deriv (nth param (a-deriv-mats gp)))

          (progn
            ;; Make a derivative matrix
            (setf a-deriv (make-array (list (n-obs gp) var-parent-dim)
                                      :initial-element 0d0))
            (dotimes (i (n-obs gp))
              (setf (aref a-deriv i (+ offset i)) -1d0))
            (setf a-deriv (array-to-mat a-deriv :ctype :double)
                  (a-deriv-mats gp) (nconc (a-deriv-mats gp)
                                           (list a-deriv)))))

      ;; Make y subject to the param derivative
      (destructuring-bind (var-parent-mean var-parent-cov)
          (var-parent-distributions gp)
        (with-facets ((vp-mean (var-parent-mean 'backing-array
                                                :direction :input)))
          (dotimes (i (n-obs gp))
            (setf (aref dy i)
                  (- (aref vp-mean (+ offset i))))))

        ;; y part of the derivative
        (setf dy (array-to-mat dy :ctype :double)
              dLL (- (dot dy (obs-mat-copy gp))))

        ;; Covariance part of the derivative
        (setf temp-mat (make-mat (list (n-obs gp) var-parent-dim)
                                 :ctype :double)
              d-cov (make-mat (list (n-obs gp) (n-obs gp))
                              :ctype :double))
        (gemm! 1d0 a-deriv var-parent-cov 0d0 temp-mat
               :m (n-obs gp) :n var-parent-dim :k var-parent-dim
               :lda var-parent-dim :ldb var-parent-dim :ldc var-parent-dim)
        (gemm! 1d0 temp-mat (a-mat gp) 0d0 d-cov
               :transpose-b? t :m (n-obs gp) :n (n-obs gp) :k var-parent-dim
               :lda var-parent-dim :ldb var-parent-dim :ldc (n-obs gp))
        (setf d-cov (m+ d-cov (transpose d-cov)))
        (potrs! (Kff gp) d-cov :uplo #\L :n (n-obs gp) :nrhs (n-obs gp) 
                               :lda (n-obs gp) :ldb (n-obs gp) :transpose-b? t)
        (with-facets ((cov-array (d-cov 'array :direction :input)))
          (dotimes (i (n-obs gp))
            (incf dLL (* -0.5d0 (aref cov-array i i)))))

        dLL))))


(defgeneric closed-downwards-dLL/dparent (gp param)
  (:documentation "Computes the derivative of log likelihood with respect to parents for exact gaussian processes.")
  (:method ((gp linear-parent-gp) param)
    (let ((dy (nth (1+ param) (var-parent-distributions gp))))
      (dot dy (obs-mat-copy gp)))))


(defmethod NLL-and-derivs ((gp linear-parent-gp) deriv-array &key (latent nil))
  (declare (ignore latent))
  (let ((NLL (call-next-method))
        (n-parent-params (apply #'+ (n-parent-params gp))))

    ;; Parent param derivatives
    (loop for param below n-parent-params do
      (setf (aref deriv-array (+ (- (n-gp-params gp)
                                    n-parent-params)
                                 param))
            (- (dLL/dparent gp param))))
    (setf (a-deriv-mats-updated-p gp) t)

    NLL))


(defmethod NLL-and-derivs ((gp variational-linear-parent-gp) deriv-array
                           &key (latent nil))
  (let ((NLL (call-next-method))
        (n-total-parent-params (apply #'+ (n-parent-params gp)))
        (offset 0))

    (with-facets ((qf-mean-array ((qf-mean gp) 'backing-array
                                  :direction :input)))
      (let ((qf-mean-list (coerce qf-mean-array 'list))
            (qf-cov-list (coerce (Kff gp) 'list)))

        ;; Output derivatives
        (loop for param below (n-output-params (output gp)) do
          (setf (aref deriv-array (+ (n-kern-params (kernel gp)) param))
                (if latent
                    0d0
                    (- (dLL/doutputparam (output gp) param qf-mean-list
                                         qf-cov-list
                                         (var-parent-distributions gp)
                                         (list-length (parent-gps gp))
                                         (parent-outputs gp)
                                         (parent-params gp))))))
        
        ;; Parent param derivatives
        (loop for parent-index from 0
              for n-parent-params in (n-parent-params gp)
              do (loop for param below n-parent-params do
                (setf (aref deriv-array (+ (- (n-gp-params gp)
                                              n-total-parent-params)
                                           offset
                                           param))
                      (if latent
                          0d0
                          (- (dLL/dparentparam (output gp) 0 parent-index param
                                               qf-mean-list qf-cov-list
                                               (var-parent-distributions gp)
                                               (list-length (parent-gps gp))
                                               (parent-outputs gp)
                                               (parent-params gp))))))
              (incf offset n-parent-params))
        
        ;; (loop for param below n-total-parent-params do
        ;;   (setf (aref deriv-array (+ (- (n-gp-params gp)
        ;;                                 n-total-parent-params)
        ;;                              param))
        ;;         (if latent
        ;;             0d0
        ;;             (- (dLL/dparentparam (output gp) param qf-mean-list
        ;;                                  qf-cov-list
        ;;                                  (var-parent-distributions gp)
        ;;                                  (list-length (parent-gps gp))
        ;;                                  (parent-outputs gp)
        ;;                                  (parent-params gp))))))
        ))

    ;;(format t "~a~%" NLL)
    NLL))


(defmethod NLL-and-derivs ((gp variational-combined-output-linear-parent-gp)
                           deriv-array &key (latent nil))
  (let ((NLL (call-next-method))
        (n-total-kern-params (loop for kernel in (kernel gp)
                                   sum (n-kern-params kernel)))
        (n-total-parent-params (* (n-combined gp)
                                  (apply #'+ (n-parent-params gp))))
        (offset 0))

    ;; Output derivatives
    (loop for param below (n-output-params (output gp)) do
      (setf (aref deriv-array (+ n-total-kern-params param))
            (if latent
                0d0
                (- (dLL/doutputparam (output gp) param
                                     (reshaped-qf-mean gp)
                                     (reshaped-Kff gp)
                                     (var-parent-distributions gp)
                                     (list-length (parent-gps gp))
                                     (parent-outputs gp)
                                     (parent-params gp))))))
        
    ;; Parent param derivatives
    (loop for gp-index below (n-combined gp) do
      (loop for parent-index from 0
            for n-parent-params in (n-parent-params gp)
            do (loop for param below n-parent-params do
              (setf (aref deriv-array (+ (- (n-gp-params gp)
                                            n-total-parent-params)
                                         offset
                                         param))
                    (if latent
                        0d0
                        (- (dLL/dparentparam (output gp) gp-index parent-index param
                                             (reshaped-qf-mean gp)
                                             (reshaped-Kff gp)
                                             (var-parent-distributions gp)
                                             (list-length (parent-gps gp))
                                             (parent-outputs gp)
                                             (parent-params gp))))))
               (incf offset n-parent-params)))
    
    NLL))
