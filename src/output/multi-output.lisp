(uiop:define-package #:dag-gp/output/multi-output
    (:use #:cl
          #:dag-gp/quadrature
          #:dag-gp/output/output)
  (:export #:multi-output
           #:n-gps
           #:LL-single
           #:quadrature-set
           #:n-gps))

(in-package #:dag-gp/output/multi-output)


(defclass multi-output (output)
  (
   ;; (quadrature-points
   ;;  :initform (list -2.350604973674492222834d0
   ;;                  -1.335849074013696949715d0
   ;;                  -0.4360774119276165086792d0
   ;;                  0.436077411927616508679d0
   ;;                  1.335849074013696949715d0
   ;;                  2.350604973674492222834d0)
   ;;  :Accessor quad-points)
   ;; (quadrature-weights
   ;;  :initform (list 0.0025557844020562465d0
   ;;                  0.08861574604191454d0
   ;;                  0.40882846955602925d0
   ;;                  0.40882846955602925d0
   ;;                  0.08861574604191454d0
   ;;                  0.0025557844020562465d0))
   (quadrature-set
    :initform nil
    :accessor quadrature-set)
   (n-gps
    :initarg :n-gps
    :accessor n-gps))
  (:documentation "An output depending on multiple latent processes."))


(defmethod initialize-instance :after ((output multi-output) &key)
  ;; (let ((new-quad-points nil)
  ;;       temp-quad-points
  ;;       (new-quad-weights nil)
  ;;       temp-quad-weights)
  ;;   (dotimes (i (n-gps output))
  ;;     (setf temp-quad-points nil
  ;;           temp-quad-weights nil)
  ;;     (loop for point in (quad-points output)
  ;;           for weight in (quad-weights output)
  ;;           do (if new-quad-points
  ;;                  (loop for new-point in new-quad-points
  ;;                        for new-weight in new-quad-weights
  ;;                        do (setf temp-quad-points
  ;;                                 (nconc temp-quad-points
  ;;                                        (list (append new-point
  ;;                                                      (list point))))
  ;;                                 temp-quad-weights
  ;;                                 (nconc temp-quad-weights
  ;;                                        (list (* weight new-weight)))))
  ;;                  (progn
  ;;                    (setf temp-quad-points
  ;;                          (nconc temp-quad-points (list (list point)))
  ;;                          temp-quad-weights
  ;;                          (nconc temp-quad-weights (list weight))))))
  ;;     (setf new-quad-points temp-quad-points
  ;;           new-quad-weights temp-quad-weights))
  ;;   (setf (quad-points output) new-quad-points
  ;;         (quad-weights output) new-quad-weights))
  (setf (quadrature-set output) (make-instance 'quad-set))
  (dotimes (i (n-gps output))
    (add-quad-variable (quadrature-set output) i #'gauss-hermite-generator))
  (flatten (quadrature-set output)))


(defmethod LL-single ((output multi-output) mean var y-and-parent-dist
                      n-parents parent-outputs parent-params)
  (loop for weight in (weights (quadrature-set output))
        for quad-point in (flattened (quadrature-set output))
        for int-point = (loop for x in quad-point
                              for mi in mean
                              for vi in var
                              collect (+ mi (* (sqrt (* 2 vi)) x)))
        sum (* weight (LL-single-known-f output int-point y-and-parent-dist
                                         parent-outputs parent-params))))


;; Acts on means and vars for (index0 index1 ...)
(defmethod dLL/dmu+dvar ((output multi-output) mean var y-and-parent-dists n-parents
                         parent-outputs parent-params)
    ;; For quadrature based likelihoods, we compute the derivative with respect to
    ;; quadrature points, then use this to generate mu and var derivatives.
    (let ((dLL/df (dLL/df output mean var y-and-parent-dists n-parents
                          parent-outputs parent-params)))

      ;; dLL/df is (loc0: (quad0: (df0 df1) quad1: (df0 df1)) ...)
      ;; To derive by mu, add the first els for each location, second els for each
      ;; location, etc.
      (flet ((add-location (x) (apply #'mapcar #'+ x))
             (parameterized-dLL/dvar-single (var-i dLL/df-i)
               (dLL/dvar-single output var-i dLL/df-i)))

        (list (mapcar #'add-location dLL/df) ;; dLL/dmu
              ;; dLL/dvar
              (mapcar #'add-location
                      (mapcar #'parameterized-dLL/dvar-single var dLL/df))))))


(defmethod dLL/dvar-single ((output multi-output) var dLL/df-single)
  ;; dLL/dvar = dLL/df df/dvar = dLL/df x/sqrt(2 var)
  ;(format t "dLL/df-single = ~a~%" dLL/df-single)
  (flet ((var-deriv (quad-point dLL/df)
           (loop for x in quad-point
                 for dLL/df-i in dLL/df
                 for var-i in var
                 collect (/ (* x dLL/df-i)
                            (sqrt (* 2 var-i))))))
    (let ((int-points (mapcar #'var-deriv
                              (flattened (quadrature-set output))
                              dLL/df-single)))
      int-points)))


(defmethod dLL/df-single-known-f ((output multi-output) f y-and-parent-dist
                                   parent-outputs parent-params)
  (loop for index below (n-gps output)
        collect (loop for y-and-parent-value in (flattened y-and-parent-dist)
                      for weight in (weights y-and-parent-dist)
                      sum (* weight (dlog-lik/df output index f y-and-parent-value
                                                 parent-outputs parent-params)))))


(defmethod dLL/df-single ((output multi-output) mean var y-and-parent-dists
                          n-parents parent-outputs parent-params)
  (flet ((parameterized-dLL/df-single-known-f (f weight)
           (loop for dLL/df in (dLL/df-single-known-f output f y-and-parent-dists
                                                      parent-outputs parent-params)
                 collect (* weight dLL/df)))
         (make-int-points (quad-point)
           (loop for x in quad-point
                 for mi in mean
                 for vi in var
                 collect (+ mi (* (sqrt (* 2 vi)) x)))))
    (let ((int-points (mapcar #'make-int-points
                              (flattened (quadrature-set output)))))
      (mapcar #'parameterized-dLL/df-single-known-f
              int-points
              (weights (quadrature-set output))))))


(defmethod dLL-single/dparentparam-known-f ((output multi-output) index parent param
                                            f y-and-parent-dist parent-outputs
                                            parent-params)
  (loop for y-and-parent-value in (flattened y-and-parent-dist)
        for weight in (weights y-and-parent-dist)
        sum (* weight
               (dlog-lik/df output index f y-and-parent-value
                            parent-outputs parent-params)
               (dchild/dparent-param (nth parent parent-outputs)
                                     param
                                     (nth (1+ parent) y-and-parent-value)
                                     (nth parent (nth index parent-params))))))


(defmethod dLL-single/dparentparam ((output multi-output) index parent param mean
                                     var y-and-parent-dists n-parents parent-outputs
                                     parent-params)
  (loop for weight in (weights (quadrature-set output))
        for quad-point in (flattened (quadrature-set output))
        for int-point = (loop for x in quad-point
                              for mi in mean
                              for vi in var
                              collect (+ mi (* (sqrt (* 2 vi)) x)))
        sum (* weight (dLL-single/dparentparam-known-f
                       output index parent param int-point y-and-parent-dists
                       parent-outputs parent-params))))
