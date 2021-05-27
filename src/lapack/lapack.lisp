(uiop:define-package #:dag-gp/lapack/lapack
    (:use #:cl
          #:cffi
          #:mgl-mat)
  (:import-from #:alexandria
                #:with-unique-names)
  (:export #:define-lapack-function))

(in-package #:dag-gp/lapack/lapack)


;; (define-foreign-library liblapack
;;   (t (:default "liblapack")))


;; Name of lapack function for a specific ctype
(defun lapack-function-name (name ctype)
  (read-from-string (format nil "lapack-~A~A"
                            (mgl-mat::ctype-blas-prefix ctype)
                            name)))

;; Get lapack function name for an input ctype
(defmacro lapack-function-name* (name ctype)
  `(ecase ,ctype
     ,@(loop for ctype in *supported-ctypes*
             collect `((,ctype)
                       ',(lapack-function-name name ctype)))))


;; Call the lapack function of the given name
(defmacro call-lapack-function (name (&rest params))
  (let* ((mgl-mat::*mat-param-type* '(:pointer :float))
         (mat-params (remove-if-not #'mgl-mat::mat-param-p params)))
    (with-unique-names (ctype)
      `(let ((,ctype (mgl-mat::common-mat-ctype ,@(mapcar #'mgl-mat::param-name mat-params))))
         (funcall (lapack-function-name* ,name ,ctype)
                  ,@(mapcar (lambda (param)
                              (let ((name (mgl-mat::param-name param)))
                                (if (and (not (mgl-mat::mat-param-p param))
                                         (equal (mgl-mat::param-type param)
                                                '(:pointer :float)))
                                    `(coerce-to-ctype ,name :ctype ,ctype)
                                    name)))
                            params))))))


;; Returns the name of the lapack function, i.e. [fname] -> d[fname]_
(defun lapack-foreign-function-name (name ctype)
  (format nil "~A~A_" (mgl-mat::ctype-blas-prefix ctype)
          (string-downcase (symbol-name name))))


(defun lapack-funcall-form (name ctype params return-type args)
  (let ((cname (lapack-foreign-function-name name ctype)))
    `(foreign-funcall
      ,cname
      ,@(loop for param in params
              for arg in args
              append (list (mgl-mat::convert-param-types
                            (mgl-mat::param-type param)
                            ctype)
                           (convert-param arg param)))
      ,(mgl-mat::convert-param-types return-type ctype))))


;; Convert parameter arguments, as necessary
(defun convert-param-ptr (arg param)
  (if (equal (mgl-mat::param-type param) (list :pointer :char))
      `(char-code ,arg)
      arg))

(defun convert-param (arg param)
  (if (equal (mgl-mat::param-type param) :char)
      `(char-code ,arg)
        arg))


;; Makes the function body up to the foreign funcall 
(defun lapack-call-form* (params args fn)
  (if (endp params)
      (funcall fn (reverse args))
      (let* ((param (first params))
             (name (mgl-mat::param-name param))
             (ctype (mgl-mat::param-type param))
             (direction (mgl-mat::param-direction param)))
        (if (mgl-mat::mat-param-p param)
            (let ((arg (gensym (symbol-name name))))
              `(with-facets ((,arg (,name 'foreign-array
                                          :direction ,direction)))
                 (let ((,arg (mgl-mat::offset-pointer ,arg)))
                   ,(lapack-call-form* (rest params) (cons arg args) fn))))
            (if (and (listp ctype)
                     (eq (first ctype) :pointer))
                (let ((pointer-ctype (second ctype))
                      (arg (gensym (symbol-name name))))
                  `(with-foreign-object (,arg ,pointer-ctype)
                     ,@(when (member direction '(:input :io))
                         `((setf (mem-ref ,arg ,pointer-ctype) ,(convert-param-ptr name param))))
                     ,(lapack-call-form* (rest params) (cons arg args) fn)
                     ,@(when (member direction '(:io :output))
                         `((mem-ref ,arg ,pointer-ctype)))))
                (lapack-call-form* (rest params) (cons name args) fn))))))


;; Returns the body of the function 'lapack-name'
(defun lapack-call-form (lapack-name ctype params return-type)
  (lapack-call-form* params ()
                     (lambda (args)
                       (lapack-funcall-form lapack-name ctype
                                            params return-type args))))


;; Make the functions 'lapack-[fname]', 'lapack-d[fname]', 'lapack-s[fname]'
(defmacro define-lapack-function ((name &key (ctypes '(:float :double)))
                                  (return-type (&rest params)))
  (let* ((mgl-mat::*mat-param-type* '(:pointer :float))
         (in-params (remove-if #'mgl-mat::non-mat-output-param-p params))
         (lisp-parameters (mapcar #'mgl-mat::param-name in-params)))
    `(progn
       ,@(loop for ctype in ctypes
               collect `(defun ,(lapack-function-name name ctype)
                          ,lisp-parameters
                          ,(let ((params (mgl-mat::convert-param-types params ctype)))
                             (lapack-call-form name ctype params return-type))))
       (defun ,(lapack-function-name name nil) (,@lisp-parameters)
         (call-lapack-function ,name ,in-params)))))

;; (defparameter a (foreign-alloc :double :initial-element 4d0))
;; (defparameter ch (foreign-alloc :char :initial-element (char-code #\L)))
;; (defparameter n (foreign-alloc :int :initial-element 1))
;; (defparameter lda (foreign-alloc :int :initial-element 1))
;; (defparameter info (foreign-alloc :int :initial-element 0))
;; (foreign-funcall "dpotrf_" :pointer ch :pointer n :pointer a :pointer lda :pointer info :int)
