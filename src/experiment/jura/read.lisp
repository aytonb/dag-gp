(uiop:define-package #:dag-gp/experiment/jura/read
    (:use #:cl
          #:cl-ppcre)
  (:export #:read-jura-sample
           #:read-jura-validation))

(in-package #:dag-gp/experiment/jura/read)


;; Rock Types: 1: Argovian, 2: Kimmeridgian, 3: Sequanian, 4: Portlandian, 5: Quaternary.
;; Land uses: 1: Forest, 2: Pasture, 3: Meadow , 4: Tillage
(defun read-jura-sample ()
  (let ((data (make-array '(259 11)))
        line)
    (with-open-file (s (asdf:system-relative-pathname
                        :gaussian-process "./src/experiment/jura/jura_sample.dat")
                       :if-does-not-exist :error)
      (dotimes (i 13)
        (read-line s))
      (dotimes (i 259)
        (setf line (split "\\s+" (subseq (read-line s) 4)))
        (dotimes (v 11)
          (setf (aref data i v)
                (if (member v '(2 3))
                    (parse-integer (nth v line))
                    (coerce (with-input-from-string (var (nth v line))
                              (read var))
                            'double-float)))))
      data)))


(defun read-jura-validation ()
  (let ((data (make-array '(100 11)))
        line)
    (with-open-file (s (asdf:system-relative-pathname
                        :gaussian-process "./src/experiment/jura/jura_validation.dat")
                       :if-does-not-exist :error)
      (read-line s)
      (dotimes (i 100)
        (setf line (split "\\s+" (read-line s)))
        (dotimes (v 11)
          (setf (aref data i v)
                (cond
                  ((equal v 2)
                   (parse-integer (nth v line)))
                  ((equal v 3)
                   (cond
                     ((string-equal (nth v line) "Argovian") 1)
                     ((string-equal (nth v line) "Kimmeridgian") 2)
                     ((string-equal (nth v line) "Sequanian") 3)
                     ((string-equal (nth v line) "Portlandian") 4)
                     ((string-equal (nth v line) "Quaternary") 5)))
                  (t (coerce (with-input-from-string (var (nth v line))
                               (read var))
                             'double-float))))))
      data)))
