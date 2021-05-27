(uiop:define-package #:dag-gp/quadrature/quad-set
    (:use #:cl)
  (:export #:quad-set
           #:add-quad-variable
           #:nested
           #:flattened
           #:weights
           #:accumulate-variables
           #:flatten
           #:marginalize
           #:run-on-quadrature
           #:copy-quad-set
           #:gauss-hermite-generator))

(in-package #:dag-gp/quadrature/quad-set)


(defclass quad-set ()
  ((nested-vals
    :initform '(1d0)
    :accessor nested)
   (flattened-vals
    :initform nil
    :accessor flattened)
   (weights
    :initform nil
    :accessor weights)
   (ordering
    :initform nil
    :accessor quad-ordering)))


(defgeneric add-quad-variable (quad-set var generator-fn &optional ordering)
  (:documentation "Adds a variable to the quadrature. generator-fn acts on a list of variables and returns a list of (value weights) for the quadrature.")
  (:method ((quad-set quad-set) var generator-fn &optional (ordering :default))
    (let* ((n-vars (if (equal ordering :default)
                       (list-length (quad-ordering quad-set))
                       (list-length ordering)))
           inverse-ordering
           (accumulated (make-list n-vars)))

      ;; Solve the inverse ordering, level -> position on accumulated
      (if (equal ordering :default)
          (setf inverse-ordering (loop for i below n-vars
                                       collect i))
          (setf inverse-ordering (loop for i in (quad-ordering quad-set)
                                       for pos = (position i ordering)
                                       collect (if pos
                                                   pos
                                                   nil))))

      (flet ((add-fn (accumulated orig-weight)
               (let ((new-alist nil))
                 (loop for (var weight) in (funcall generator-fn accumulated) do
                   (setf new-alist (acons var (list (* weight orig-weight))
                                          new-alist)))
                 new-alist)))
        
        (accumulate-variables-and-modify quad-set nil (nested quad-set) nil
                                         inverse-ordering accumulated 0 #'add-fn)
        (setf (quad-ordering quad-set)
              (nconc (quad-ordering quad-set) (list var)))))))


;; Probably the most ratchet function I've ever written.
(defun accumulate-variables-and-modify (quad-set old-var nested old-nested
                                        desired-ordering accumulated level final-fn)
  "Goes through a nested a-list, accumulates variables, and runs final-fn on the final result."
  (let (orig-weight)
    
    (if (listp (first nested))

        ;; Keep accumulating if nested contains lists
        (loop for (var . new-nested) in nested do
          (when (nth level desired-ordering)
            (setf (nth (nth level desired-ordering)
                       accumulated)
                  var))
          (accumulate-variables-and-modify quad-set var new-nested nested
                                           desired-ordering accumulated
                                           (1+ level) final-fn))

        ;; Run the function otherwise.
        (if (equal level 0)
            (progn
              (setf orig-weight (first nested))
              (setf (nested quad-set)
                    (funcall final-fn accumulated orig-weight)))
            (progn
              (setf orig-weight (first nested))
              (setf (cdr (assoc old-var old-nested :test #'equalp))
                    (funcall final-fn accumulated orig-weight)))))))


(defgeneric marginalize (quad-set variable)
  (:documentation "Marginalize out a variable from the quadrature set.")
  (:method ((quad-set quad-set) variable)
    (let ((initial-level (position variable (quad-ordering quad-set))))
      (marginalize-at-level quad-set nil (nested quad-set) nil 0 initial-level)
      (combine-after-marginalization quad-set nil (nested quad-set) nil
                                     0 initial-level)
      (setf (quad-ordering quad-set) (remove variable (quad-ordering quad-set))))))


(defun marginalize-at-level (quad-set old-var nested old-nested level desired-level)
  (if (not (equal level desired-level))

      (loop for (var . new-nested) in nested do
        (marginalize-at-level quad-set var new-nested nested (1+ level) desired-level))

      (if (equal level 0)
          (let ((marginalized nil))     
            (loop for (var . new-nested) in nested
                  do (setf marginalized (nconc marginalized new-nested)))
            (setf (nested quad-set) marginalized))
          
          (let ((marginalized nil))
            (loop for (var . new-nested) in nested
                  do (setf marginalized (nconc marginalized new-nested)))
            (setf (cdr (assoc old-var old-nested :test #'equalp)) marginalized)))))


(defun combine-after-marginalization (quad-set old-var nested old-nested
                                      level desired-level)
  (if (< level desired-level)

      (loop for (var . new-nested) in nested do
        (combine-after-marginalization quad-set var new-nested nested
                                       (1+ level) desired-level))

      (if (not (listp (first nested)))

          ;; If not a list of lists, then a list of numbers, so combine
          (if (equal level 0)
              (setf (nested quad-set)
                    (list (loop for weight in nested
                                sum weight)))
              (setf (cdr (assoc old-var old-nested :test #'equalp))
                    (list (loop for weight in nested
                                sum weight))))
          
          ;; Otherwise, combine equal lists
          (let ((marginalized nil))
            (loop for (var . new-nested) in nested
                  do (if (member var marginalized :key #'car :test #'equalp)
                         (setf (cdr (assoc var marginalized :test #'equalp))
                               (nconc (cdr (assoc var marginalized
                                                  :test #'equalp))
                                      new-nested))
                         (setf marginalized (acons var new-nested marginalized))))
            (if (equal level 0)
                (setf (nested quad-set) marginalized)
                (setf (cdr (assoc old-var old-nested :test #'equalp))
                      marginalized))

            ;; Now go through margialized and repeat the process
            (loop for (var . new-nested) in marginalized do
              (combine-after-marginalization quad-set var new-nested marginalized
                                             (1+ level) desired-level))))))


(defgeneric flatten (quad-set &optional ordering)
  (:documentation "Makes distributions and weights for quadrature sets in a list.")
  (:method ((quad-set quad-set) &optional (ordering :default))
    (let* ((n-vars (if (equal ordering :default)
                       (list-length (quad-ordering quad-set))
                       (list-length ordering)))
           inverse-ordering
           (accumulated (make-list n-vars)))

      ;; Solve the inverse ordering, level -> position on accumulated
      (if (equal ordering :default)
          (setf inverse-ordering (loop for i below n-vars
                                       collect i))
          (setf inverse-ordering (loop for i in (quad-ordering quad-set)
                                       for pos = (position i ordering)
                                       collect (if pos
                                                   pos
                                                   nil))))
      
      (setf (flattened quad-set) nil
            (weights quad-set) nil)

      (flet ((flatten-fn (accumulated weight)
               (setf (flattened quad-set) (nconc (flattened quad-set)
                                                 (list (copy-tree accumulated)))
                     (weights quad-set) (nconc (weights quad-set)
                                               (list weight)))))
        (accumulate-variables (nested quad-set) inverse-ordering accumulated
                              0 #'flatten-fn)))))


(defun accumulate-variables (nested desired-ordering accumulated level final-fn)
  "Goes through a nested a-list, accumulates variables, and runs final-fn on the final result."
  (let (orig-weight)
    (if (listp (first nested))

        ;; Keep accumulating if nested contains lists
        (loop for (var . new-nested) in nested do
          (when (nth level desired-ordering)
            (setf (nth (nth level desired-ordering)
                       accumulated)
                  var))
          (accumulate-variables new-nested desired-ordering accumulated (1+ level)
                                final-fn))

        ;; Run the function otherwise.
        (progn
          (setf orig-weight (first nested))
          (funcall final-fn accumulated orig-weight)))))


(defgeneric run-on-quadrature (quad-set ordering final-fn)
  (:documentation "Runs final-fn on (vars weight) with vars given in the order specified by ordering.")
  (:method ((quad-set quad-set) ordering final-fn)
    (let* ((n-vars (list-length (quad-ordering quad-set)))
           (inverse-ordering (loop for i in (quad-ordering quad-set)
                                   collect (position i ordering)))
           (accumulated (make-list n-vars)))
      (accumulate-ordered-variables (nested quad-set)
                                    inverse-ordering
                                    accumulated
                                    0
                                    final-fn))))


(defun accumulate-ordered-variables (nested desired-ordering
                                     accumulated level final-fn)
  "Goes through a nested a-list, accumulates variables in an order, and runs final-fn on the final result."
  (let (orig-weight)
    (if (listp (first nested))

        ;; Keep accumulating if nested contains lists
        (loop for (var . new-nested) in nested do
          ;; Position given by desired-ordering[var]
          (setf (nth (nth level
                          desired-ordering)
                     accumulated)
                var)
          (accumulate-ordered-variables new-nested desired-ordering
                                        accumulated (1+ level) final-fn))

        ;; Run the function otherwise.
        (progn
          (setf orig-weight (first nested))
          (funcall final-fn accumulated orig-weight)))))


(defgeneric copy-quad-set (quad-set)
  (:documentation "Makes a copy of the quadrature set.")
  (:method ((quad-set quad-set))
    (let ((new-quad (make-instance 'quad-set)))
      (setf (nested new-quad) (copy-tree (nested quad-set))
            (quad-ordering new-quad) (copy-tree (quad-ordering quad-set)))
      new-quad)))


(defun gauss-hermite-generator (variables)
  (declare (ignore variables))
  (list (list -2.350604973674492222834d0 0.0025557844020562465d0)
        (list -1.335849074013696949715d0 0.08861574604191454d0)
        (list -0.4360774119276165086792d0 0.40882846955602925d0)
        (list 0.4360774119276165086792d0 0.40882846955602925d0)
        (list 1.335849074013696949715d0 0.08861574604191454d0)
        (list 2.350604973674492222834d0 0.0025557844020562465d0)))



(defun binary-test-generator (variables)
  (list (list 0 (expt 0.5d0 (1+ (list-length variables))))
        (list 1 (- 1 (expt 0.5d0 (1+ (list-length variables)))))))


(defun discrete-test-generator (variables)
  (declare (ignore variables))
  (list (list 2 0.3d0)
        (list 3 0.7d0)))


(defun test-add ()
  (let ((quad (make-instance 'quad-set)))
    (add-quad-variable quad 1 #'discrete-test-generator)
    (add-quad-variable quad 0 #'binary-test-generator)
    quad))


(defun test-add-2 ()
  (let ((quad (make-instance 'quad-set)))
    (add-quad-variable quad 1 #'discrete-test-generator)
    (add-quad-variable quad 3 #'discrete-test-generator)
    (add-quad-variable quad 0 #'binary-test-generator)
    (add-quad-variable quad 2 #'discrete-test-generator)
    quad))
