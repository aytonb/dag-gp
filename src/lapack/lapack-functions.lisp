(uiop:define-package #:dag-gp/lapack/lapack-functions
    (:use #:cl
          #:mgl-mat
          #:cffi
          #:dag-gp/lapack/lapack)
  (:export #:potrf!
           #:zero-potrf!
           #:potri!
           #:symmetric-potri!
           #:potrs!
           #:gemv!
           #:ger!
           #:symm!
           #:fixed-dot))

(in-package #:dag-gp/lapack/lapack-functions)


(define-lapack-function (potrf)
    (:int ((uplo (:pointer :char)) (n (:pointer :int)) (x :mat :io)
           (lda (:pointer :int)) (info (:pointer :int)))))


;; mgl-mat is row major, while lapack is column major. So if we have a lower
;; diagonal lisp matrix, we actually have an upper diagonal C matrix and vice
;; versa. It's valid then if we just switch the uplo, and everything is as
;; expected.
(defun potrf! (x &key (n (mat-size x)) (uplo #\L) (lda (mat-size x)))
  (let ((new-uplo (ecase uplo
                    (#\L #\U)
                    (#\U #\L))))
    (lapack-potrf new-uplo n x lda 0)))


;; A version of potrf that explicitly zeros out other elements. Useful for
;; remultiplication
;; This shouldn't depend explicitly on double if I want to export this
(defun zero-potrf! (x &key (n (mat-size x)) (uplo #\L) (lda (mat-size x)))
  (let* ((new-uplo (ecase uplo
                    (#\L #\U)
                    (#\U #\L))))
    (lapack-potrf new-uplo n x lda 0)
    (with-facets ((foreign (x 'foreign-array :direction :io)))
      (let ((foreign-ptr (mgl-mat::offset-pointer foreign)))
        (if (equal uplo #\L)
            (dotimes (row (1- n))
              (loop for col from (1+ row) below n
                    do (setf (mem-aref foreign-ptr :double (+ (* n row) col)) 0d0)))
            (dotimes (col (1- n))
              (loop for row from (1+ col) below n
                    do (setf (mem-aref foreign-ptr :double (+ (* n row) col)) 0d0))))))))



(define-lapack-function (potri)
    (:int ((uplo (:pointer :char)) (n (:pointer :int)) (x :mat :io)
           (lda (:pointer :int)) (info (:pointer :int)))))


(defun potri! (x &key (n (mat-dimension x 0)) (uplo #\L) (lda (mat-dimension x 1)))
  (let ((new-uplo (ecase uplo
                    (#\L #\U)
                    (#\U #\L))))
    (lapack-potri new-uplo n x lda 0)))


;; Version that explicitly makes the inverse symmetric
(defun symmetric-potri! (x &key (n (mat-dimension x 0)) (uplo #\L)
                             (lda (mat-dimension x 1)))
  (potri! x :n n :uplo uplo :lda lda)
  (with-facets ((x-array (x 'array :direction :io)))
    (if (equal uplo #\L)
        (dotimes (c n)
          (dotimes (r c)
            (setf (aref x-array r c) (aref x-array c r))))
        (dotimes (r n)
          (dotimes (c r)
            (setf (aref x-array r c) (aref x-array c r)))))))


(define-lapack-function (potrs)
    (:int ((uplo (:pointer :char)) (n (:pointer :int)) (nrhs (:pointer :int))
           (x :mat :input) (lda (:pointer :int)) (b :mat :io)
           (ldb (:pointer :int)) (info (:pointer :int)))))


(defun potrs! (x b &key (n (mat-dimension x 0)) (uplo #\L)
                     (nrhs (mat-dimension b 1)) (lda (mat-dimension x 0))
                     (ldb (mat-dimension b 0)) (transpose-b? nil))
  (let ((new-uplo (ecase uplo
                    (#\L #\U)
                    (#\U #\L))))
    (if transpose-b?
        (lapack-potrs new-uplo n nrhs x lda b ldb 0)
        (progn
          (setf b (transpose b))
          (lapack-potrs new-uplo n nrhs x lda b ldb 0)
          (setf b (transpose b))))))


(define-lapack-function (gemv)
    (:void ((trans (:pointer :char)) (n (:pointer :int)) (m (:pointer :int))
            (alpha (:pointer :double)) (a :mat :input) (lda (:pointer :int))
            (x :mat :input) (incx (:pointer :int)) (beta (:pointer :double))
            (y :mat :io) (incy (:pointer :int)))))


(defun gemv! (alpha a x beta y &key (transpose-a? nil) (m (mat-dimension a 0))
                                 (n (mat-dimension a 1)) (lda (mat-dimension a 1))
                                 (incx 1) (incy 1))
  ;; We are row major, so if we are not transposing a, then the input to BLAS is
  ;; that a should be transposed, and the number of rows is the number of cols
  ;; and vice versa.
  (let ((trans (if transpose-a?
                   #\N
                   #\T)))
    (lapack-gemv trans n m alpha a lda x incx beta y incy)))


(define-lapack-function (ger)
    (:void ((n (:pointer :int)) (m (:pointer :int)) (alpha (:pointer :double))
            (y :mat :input) (incy (:pointer :int)) (x :mat :input)
            (incx (:pointer :int)) (a :mat :io) (lda (:pointer :int)))))


(defun ger! (alpha x y a &key (m (mat-dimension x 0)) (n (mat-dimension y 0))
                           (incx 1) (incy 1) (lda (mat-dimension a 1)))
  ;; Since we are row major, we get the correct result by reversing x and y
  (lapack-ger n m alpha y incy x incx a lda))


(define-lapack-function (symm)
    (:void ((side (:pointer :char)) (uplo (:pointer :char)) (n (:pointer :int))
            (m (:pointer :int)) (alpha (:pointer :double)) (a :mat :input)
            (lda (:pointer :int)) (b :mat :input) (ldb (:pointer :int))
            (beta (:pointer :double)) (c :mat :io) (ldc (:pointer :int)))))


(defun symm! (alpha a b beta c &key (side #\L) (uplo #\L) (m (mat-dimension b 0))
                                 (n (mat-dimension b 1)) (lda (mat-dimension a 1))
                                 (ldb (mat-dimension b 1)) (ldc (mat-dimension c 1)))
  ;; Reversing both side and uplo should have the desired effect.
  ;; To get A*B with B m x n, we foreign solve B^T*A^T with B^T n x m.
  (let ((new-side (ecase side
                    (#\L #\R)
                    (#\R #\L)))
        (new-uplo (ecase uplo
                    (#\U #\L)
                    (#\L #\U))))
    (lapack-symm new-side new-uplo n m alpha a lda b ldb beta c ldc)))


;; mgl-mat's function dot does not have correct assertions, hence this function
(defun fixed-dot (x y &key (n (mat-size x)) (incx 1) (incy 1))
  (assert (<= (abs (1+ (* (1- n) incx))) (mat-size x)))
  (assert (<= (abs (1+ (* (1- n) incy))) (mat-size y)))
  (if (use-cuda-p x y)
      (mgl-mat::cublas-dot n x incx y incy)
      (mgl-mat::blas-dot n x incx y incy)))


;(defparameter x (make-mat (list 1 1) :ctype :double :initial-element 2d0))
;(defparameter b (make-mat 1 :ctype :double :initial-element 1.5d0))

