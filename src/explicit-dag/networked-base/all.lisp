(uiop:define-package #:dag-gp/explicit-dag/networked-base/all
    (:use #:cl
          #:dag-gp/explicit-dag/networked-base/base
          #:dag-gp/explicit-dag/networked-base/measurement
          #:dag-gp/explicit-dag/networked-base/initialize
          #:dag-gp/explicit-dag/networked-base/predict)
  (:export #:networked-base
           #:variational-networked-base
           
           #:constituent-gps
           #:output-dim
           #:outputs
           #:constituent-locs
           #:constituent-obs
           #:constituent-loc-obs
           #:ref-dist-fns
           #:ref-parent-param
           #:factors
           #:n-latent-per-constituent
           #:networked-input-dim
           #:all-gaussian-p
           #:factor-ordering
           #:factor-ordering-up-to-date
           
           #:make-kernel-from-reference
           #:make-constituent-gps
           #:configure-for-factor

           #:add-measurement

           #:predict
           #:solve-factor-ordering))

(in-package #:dag-gp/explicit-dag/networked-base/all)
