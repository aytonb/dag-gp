(defsystem #:dag-gp-test
  :pathname "test"
  :class :package-inferred-system
  :depends-on (#:dag-gp-test/dag-gp-test)
  :perform (test-op (op c) (symbol-call :fiveam :run-all-tests)))
