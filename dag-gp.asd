;;;; Gaussian Process System Definition

(defsystem #:dag-gp
  :description "Common Lisp implementation of DAG-Gaussian Processes."
  :pathname "src"
  :class :package-inferred-system
  :depends-on (#:dag-gp/dag-gp)
  :in-order-to ((test-op (test-op dag-gp-test))))
