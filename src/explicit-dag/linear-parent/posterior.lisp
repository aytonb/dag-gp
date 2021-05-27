(uiop:define-package #:dag-gp/explicit-dag/linear-parent/posterior
    (:use #:cl
          #:mgl-mat
          #:dag-gp/output
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/linear-parent/gp)
  (:export #:dLL/dpost))

(in-package #:dag-gp/explicit-dag/linear-parent/posterior)


(defmethod dLL/dpost ((gp linear-parent-gp))
  (call-next-method)

  ;; No trace term when closed downwards
  (when (closed-downwards-p gp)
    (return-from dLL/dpost))
  
  (let ((product (make-mat (list (n-obs gp) (n-obs gp))
                           :ctype :double))
        (dLL/dqf-var (dLL/dqf-var gp)))

    ;; d tr(Kff^-1 y-cov)/dKff = - (Kff^-1 y-cov Kff^-1)
    (gemm! 0.5d0 (Kff-inv gp) (y-cov gp) 0d0 product
           :m (n-obs gp) :n (n-obs gp) :k (n-obs gp)
           :lda (n-obs gp) :ldb (n-obs gp) :ldc (n-obs gp))
    (with-facets ((prod-array (product 'array :direction :input)))
      (dotimes (row (n-obs gp))
        (dotimes (col (n-obs gp))
          (incf (aref dLL/dqf-var row col) (aref prod-array row col)))))))


(defmethod dLL/dpost ((gp variational-linear-parent-gp))
  (with-facets ((qf-mean-array ((qf-mean gp) 'backing-array :direction :input)))
    (let* ((qf-mean-list (coerce qf-mean-array 'list))
           (Kff-list (coerce (Kff gp) 'list)))
      (destructuring-bind (dLL/dqf-mu dLL/dqf-var)
          (dLL/dmu+dvar (output gp) qf-mean-list Kff-list
                        (var-parent-distributions gp)
                        (list-length (parent-gps gp))
                        (parent-outputs gp)
                        (parent-params gp))
        (setf (dLL/dqf-mu gp) dLL/dqf-mu
              (dLL/dqf-var gp) dLL/dqf-var)))))


(defmethod dLL/dpost ((gp variational-combined-output-linear-parent-gp))
  (let ((qf-means nil)
        (qf-mean-list nil)
        (Kff-list nil))
    ;; Transform from ((ind0loc0 ind0loc1 ...) (ind1loc0 ind1loc1 ...)) to
    ;; ((loc0ind0 loc0ind1) (loc1ind0 loc1ind1) ...)
    (loop for qf-mean in (qf-mean gp) do
      (setf qf-means (nconc qf-means (list (mat-to-array qf-mean)))))

    (loop for i below (array-dimension (first qf-means) 0)
          collect (loop for qf-mean in qf-means
                        collect (aref qf-mean i))
            into qf-mean-list-temp
          collect (loop for Kff in (Kff gp)
                        collect (aref Kff i))
            into Kff-list-temp
          finally (setf qf-mean-list qf-mean-list-temp
                        Kff-list Kff-list-temp))
    
    (destructuring-bind (dLL/dqf-mu dLL/dqf-var)
        (dLL/dmu+dvar (output gp) qf-mean-list Kff-list
                      (var-parent-distributions gp)
                      (list-length (parent-gps gp))
                      (parent-outputs gp)
                      (parent-params gp))

      ;; Undo the transform
      (loop for index below (n-combined gp)
            collect (loop for dLL/dqf-mu-loc in dLL/dqf-mu
                          collect (nth index dLL/dqf-mu-loc))
              into dLL/dqf-mu-temp
            collect (loop for dLL/dqf-var-loc in dLL/dqf-var
                          collect (nth index dLL/dqf-var-loc))
              into dLL/dqf-var-temp
            finally (setf (dLL/dqf-mu gp) dLL/dqf-mu-temp
                          (dLL/dqf-var gp) dLL/dqf-var-temp)))))


