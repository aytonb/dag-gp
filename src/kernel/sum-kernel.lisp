(uiop:define-package #:dag-gp/kernel/sum-kernel
    (:use #:cl
          #:dag-gp/kernel/kernel)
  (:import-from #:array-operations)
  (:export #:sum-kernel))

(in-package #:dag-gp/kernel/sum-kernel)


(defclass sum-kernel (kernel)
  ((constituents
    :accessor constituents
    :documentation "The kernels to be summed.")
   (param-indices
    :accessor param-indices
    :documentation "A list of starting indices for the parameters of the constituent kernels."))
  (:documentation "A kernel constructed as a sum of its constituents."))


(defmethod initialize-instance :before ((kernel sum-kernel) &key constituents)
  ;; Determine the number of parameters an param indices
  (unless constituents
    (error "Constituents must be specified for a sum kernel."))
  (setf (constituents kernel) constituents)
  (let ((n-params 0)
        (param-indices nil))
    (dolist (constituent constituents)
      (setf param-indices
            (nconc param-indices (list n-params)))
      (incf n-params (n-kern-params constituent)))
    (setf (n-kern-params kernel) n-params
          (param-indices kernel) param-indices)))


;; In a sum kernel, each constituent may operate on a different distance
;; index, so we have to override evaluate-matrix directly.
(defmethod evaluate-matrix ((kernel sum-kernel) sq-dists abs-dists
                            parent-sq-dists parent-abs-dists
                            &key (add-jitter t))
  (let ((mats (loop for constituent in (constituents kernel)
                    for i from 0
                    collect (evaluate-matrix constituent sq-dists abs-dists
                                              parent-sq-dists parent-abs-dists
                                              :add-jitter (and add-jitter
                                                               (equal i 0))))))
    (apply #'aops:each #'+ mats)))


(defmethod dk/dparam-matrix ((kernel sum-kernel) param sq-dists abs-dists
                             parent-sq-dists parent-abs-dists)
  (let ((constituent-index
          (loop for c-index from 0
                for param-index in (param-indices kernel)
                do (when (> param-index param)
                     (return (1- c-index)))
                finally (return (1- c-index)))))
    (dk/dparam-matrix (nth constituent-index (constituents kernel))
                      (- param (nth constituent-index (param-indices kernel)))
                      sq-dists
                      abs-dists
                      parent-sq-dists
                      parent-abs-dists)))


(defmethod initialize-kernel-params ((kernel sum-kernel))
  (loop for constituent in (constituents kernel)
        for param-index in (param-indices kernel)
        do (adjust-array (kern-params constituent)
                         (n-kern-params constituent)
                         :displaced-to (kern-params kernel)
                         :displaced-index-offset param-index)
           (initialize-kernel-params constituent)))


(defmethod copy-kernel ((kernel sum-kernel) &key (operate-on-parent :copy))
  (make-instance 'sum-kernel
                 :constituents (loop for constituent in (constituents kernel)
                                     collect (copy-kernel constituent
                                                          :operate-on-parent
                                                          operate-on-parent))))
