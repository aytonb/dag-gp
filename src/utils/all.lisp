(uiop:define-package #:dag-gp/utils/all
    (:use #:cl
          #:dag-gp/utils/distance
          #:dag-gp/utils/normalize)
  (:export #:squared-distance
           #:euclidean-distance
           #:normalize-data))

(in-package #:dag-gp/utils/all)
