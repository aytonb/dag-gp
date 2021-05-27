(uiop:define-package #:dag-gp/experiment/exchange/read
    (:use #:cl)
  (:export #:read-exchange))

(in-package #:dag-gp/experiment/exchange/read)


(defun read-exchange ()
  (let ((data (make-array '(251 14)))
        line)
    (with-open-file (s (asdf:system-relative-pathname
                        :gaussian-process "./src/experiment/exchange/exchange.dat")
                       :if-does-not-exist :error)
      (dotimes (i 2)
        (read-line s))
      (dotimes (i 251)
        (setf line (uiop:split-string (read-line s) :separator '(#\,)))
        (dotimes (v 14)
          (setf (aref data i v)
                (cond
                  ((equal v 0)
                   ;;(1+ i)
                   (- (parse-integer (nth v line))
                     2454102)
                   )
                  (t
                   (if (string-equal (nth (+ v 2) line) "")
                       nil
                       (coerce (with-input-from-string (var (nth (+ v 2) line))
                                 (read var))
                               'double-float)))))))
      data)))
