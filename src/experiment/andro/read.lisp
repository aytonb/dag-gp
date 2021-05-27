(uiop:define-package #:dag-gp/experiment/andro/read
    (:use #:cl)
  (:export #:read-andro))

(in-package #:dag-gp/experiment/andro/read)


(defun read-andro ()
  (let ((data (make-array '(54 7)))
        line)
    (with-open-file (s (asdf:system-relative-pathname
                        :gaussian-process "./src/experiment/andro/andro.arff")
                       :if-does-not-exist :error)
      (dotimes (i 40)
        (read-line s))
      (dotimes (i 48)
        (setf line (uiop:split-string (read-line s) :separator '(#\,)))
        (setf (aref data i 0) (coerce i 'double-float))
        (dotimes (v 6)
          (setf (aref data i (1+ v))
                (coerce (with-input-from-string (var (nth v line))
                          (read var))
                        'double-float))))
      (setf line (uiop:split-string (read-line s) :separator '(#\,)))
      (dotimes (i 6)
        (setf (aref data (+ 48 i) 0) (coerce (+ 48 i) 'double-float))
        (dotimes (v 6)
          (setf (aref data (+ 48 i) (1+ v))
                (coerce (with-input-from-string
                            (var (nth (+ v (* 6 i)) line))
                          (read var))
                        'double-float))))
      data)))
