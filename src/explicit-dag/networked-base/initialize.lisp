(uiop:define-package #:dag-gp/explicit-dag/networked-base/initialize
    (:use #:cl
          #:dag-gp/output
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/networked-base/base)
  (:export #:initialize-gp))

(in-package #:dag-gp/explicit-dag/networked-base/initialize)


(defmethod initialize-gp ((gp networked-base) &key (initialize-latent t)
                                                (use-all-locs t))
  (declare (ignore initialize-latent use-all-locs))
  (loop for constituent-gp in (constituent-gps gp)
        for var from 0
        do (setf (factors gp) (acons var nil (factors gp)))))

(defmethod initialize-gp ((gp variational-networked-base)
                          &key (initialize-latent t) (use-all-locs t))
  (declare (ignore use-all-locs))
  (loop for constituent-gp in (constituent-gps gp)
        for var from 0
        do (when initialize-latent
               (initialize-latent-locations constituent-gp
                                            :custom-obs-locs (constituent-locs gp)))
           (setf (factors gp) (acons var nil (factors gp))))

  (setf (all-gaussian-p gp) t)
  (loop for output in (outputs gp) do
    (unless (typep output 'gaussian-output)
      (setf (all-gaussian-p gp) nil)
      (return))))
