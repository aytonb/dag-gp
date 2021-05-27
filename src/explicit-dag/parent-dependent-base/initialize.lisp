(uiop:define-package #:dag-gp/explicit-dag/parent-dependent-base/initialize
    (:use #:cl
          #:dag-gp/kernel
          #:dag-gp/output
          #:dag-gp/explicit-dag/parent-dependent-base/base
          #:dag-gp/explicit-dag/parent-dependent-base/latent
          #:dag-gp/explicit-dag/parent-dependent-base/distance-matrix)
  (:export #:initialize-gp))

(in-package #:dag-gp/explicit-dag/parent-dependent-base/initialize)


(defgeneric initialize-gp (gp &key initialize-latent use-all-locs)
  (:documentation "Contains all the logic to set up the GP. Not in initialize-instance so the GP can be reset.")

  (:method ((gp parent-dependent-gp) &key (initialize-latent t) (use-all-locs t))
    (declare (ignore initialize-latent))
    
    ;; Count the number of parameters
    (setf (n-gp-params gp) (count-params gp))

    ;; Create the parameter vector
    (setf (param-vec gp) (make-array (n-gp-params gp)
                                     :element-type 'double-float
                                     :adjustable t))

    ;; Initialize the kernel parameters, assuming they come first in the parameter
    ;; vector
    (adjust-array (kern-params (kernel gp)) (n-kern-params (kernel gp))
                  :displaced-to (param-vec gp)
                  :displaced-index-offset 0)
    (initialize-kernel-params (kernel gp))

    ;; Initialize the output parameters
    (when (> (n-output-params (output gp)) 0)
      (adjust-array (output-params (output gp)) (n-output-params (output gp))
                    :displaced-to (param-vec gp)
                    :displaced-index-offset (n-kern-params (kernel gp)))
      (initialize-output-params (output gp)))

    ;; Initialize the specialized parameters
    (initialize-specialized-parameters gp)

    (make-observed-distance-matrices gp :use-all-locs use-all-locs))

  (:method ((gp variational-parent-dependent-gp) &key (initialize-latent t)
                                                   (use-all-locs t))
    (declare (ignore use-all-locs))
    (call-next-method)

    (when initialize-latent
      (initialize-latent-locations gp))
    
    (make-unobserved-distance-matrices gp)
    (initialize-q-means gp)
    (initialize-q-cholesky gp))

  (:method ((gp variational-combined-output-parent-dependent-gp)
            &key (initialize-latent t) (use-all-locs t))
    (declare (ignore use-all-locs))
    (let ((offset 0))
      ;; Count the number of parameters
      (setf (n-gp-params gp) (count-params gp))
      
      ;; Create the parameter vector
      (setf (param-vec gp) (make-array (n-gp-params gp)
                                       :element-type 'double-float
                                       :adjustable t))
      
      ;; Initialize the kernel parameters, assuming they come first in the parameter
      ;; vector
      (loop for kern in (kernel gp) do 
        (adjust-array (kern-params kern) (n-kern-params kern)
                      :displaced-to (param-vec gp)
                      :displaced-index-offset offset)
        (initialize-kernel-params kern)
        (incf offset (n-kern-params kern)))

      ;; Initialize the output parameters
      (when (> (n-output-params (output gp)) 0)
        (adjust-array (output-params (output gp)) (n-output-params (output gp))
                      :displaced-to (param-vec gp)
                      :displaced-index-offset offset)
        (initialize-output-params (output gp)))

      ;; Initialize the specialized parameters
      (initialize-specialized-parameters gp)
      
      (make-observed-distance-matrices gp)
      (when initialize-latent
        (initialize-latent-locations gp))
      (make-unobserved-distance-matrices gp)
      (initialize-q-means gp)
      (initialize-q-cholesky gp))))
