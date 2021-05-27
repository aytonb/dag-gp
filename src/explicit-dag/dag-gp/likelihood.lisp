(uiop:define-package #:dag-gp/explicit-dag/dag-gp/likelihood
    (:use #:cl
          #:alexandria
          #:mgl-mat
          #:dag-search
          #:dag-gp/explicit-dag/parent-dependent-base/all
          #:dag-gp/explicit-dag/linear-parent/all
          #:dag-gp/explicit-dag/networked-base/all
          #:dag-gp/explicit-dag/dag-gp/gp
          #:dag-gp/explicit-dag/dag-gp/predict

          #:dag-gp/explicit-dag/parent-dependent-base/base
          #:dag-gp/explicit-dag/parent-dependent-base/covariance
          #:dag-gp/explicit-dag/parent-dependent-base/latent
          #:dag-gp/explicit-dag/parent-dependent-base/posterior)
  (:export #:train
           #:EM-on-structure))

(in-package #:dag-gp/explicit-dag/dag-gp/likelihood)


(defmethod train ((gp dag-gp-base) &key (progress-fn nil) (allowable-fn nil)
                                     (max-set-fn nil) (search :abc)
                                     (reach-dag nil) (max-restarts 5)
                                     (relative-tol 0.02))
  (let ((score-record 0d0)
        outer-converged
        (method-swap nil)
        prev-dag
        ;(all-vars (loop for gp-index below (output-dim gp) collect gp-index))
        EM-on-structure-output
        inner-converged
        (all-factor-params nil)
        (total-liks 0))

    (when (dag-closed-downwards-p gp)
      (setf all-factor-params
            (train-closed-downwards-structure gp
                                              :progress-fn progress-fn
                                              :relative-tol relative-tol
                                              :factor-params all-factor-params
                                              :max-set-fn max-set-fn)))
      
    (loop do
      (setf prev-dag (copy-tree (factors gp)))
            
      (unless (dag-closed-downwards-p gp)
        (setf EM-on-structure-output
              (EM-on-structure gp
                               :progress-fn progress-fn
                               :max-iters 10
                               :reach-dag reach-dag
                               :relative-tol relative-tol
                               :factor-params all-factor-params)
              inner-converged (first EM-on-structure-output)
              all-factor-params (second EM-on-structure-output)))
      
        (destructuring-bind (best-dag best-score factor-params n-liks)
            (ecase search
              (:abc
               (abc-best-dag gp :progress-fn progress-fn
                                :prev-dag prev-dag
                                :factor-params all-factor-params
                                :allowable-fn allowable-fn
                                :max-set-fn max-set-fn
                                :relative-tol relative-tol))
              (:astar
               (astar-best-dag gp :progress-fn progress-fn
                                  :factor-params all-factor-params
                                  :allowable-fn allowable-fn
                                  ;;:max-set-fn max-set-fn
                                  :relative-tol relative-tol))
              (:tabu
               (tabu-best-dag gp :progress-fn progress-fn
                                 :reach-dag reach-dag
                                 :prev-dag prev-dag
                                 :factor-params all-factor-params
                                 :allowable-fn allowable-fn
                                 :max-restarts max-restarts
                                 :relative-tol relative-tol))
              (:swap
               (if method-swap
                   (abc-best-dag gp :progress-fn progress-fn
                                    :prev-dag prev-dag
                                    :factor-params all-factor-params
                                    :allowable-fn allowable-fn
                                    :max-set-fn max-set-fn
                                    :relative-tol relative-tol)
                   (tabu-best-dag gp :progress-fn progress-fn
                                     :reach-dag reach-dag
                                     :prev-dag prev-dag
                                     :factor-params all-factor-params
                                     :max-restarts max-restarts
                                     :relative-tol relative-tol))))
          
          (when (or (equal progress-fn :brief)
                    (equal progress-fn :summary))
            (format t "New best DAG = ~a: ~a~%" best-dag best-score))

          (incf total-liks n-liks)
          
          (if (> (/ (abs (- best-score score-record))
                    (abs best-score))
                 relative-tol)
              (setf outer-converged nil)
              (setf outer-converged t))
          
          ;; Store new record
          (setf score-record best-score)
          
          ;; Set the GP to follow the best DAG
          (loop for (index . parents) in best-dag do
            (configure-for-factor gp index parents)
            (update-parameter-vector (nth index (constituent-gps gp))
                                     (gethash parents
                                              (cdr (assoc index all-factor-params)))))
          
          (when reach-dag
            (return nil))

          ;; When closed downwards, have to remake the distance matrices so that
          ;; all locations are used
          (when (dag-closed-downwards-p gp)
            (loop for constituent-gp in (constituent-gps gp)
                  do (make-observed-distance-matrices constituent-gp
                                                      :use-all-locs t))
            (return nil))

          ;; If on swap mode and would terminate before swap, start running abc 
          (when (and (equal search :swap)
                     outer-converged
                     (not method-swap))
            (setf outer-converged nil
                  method-swap t)
            ;; (setf prev-dag (copy-tree (factors gp)))
            ;; (destructuring-bind (swap-best-dag swap-best-score swap-factor-params)
            ;;     (abc-best-dag gp :progress-fn progress-fn
            ;;                      :prev-dag prev-dag
            ;;                      :allowable-fn allowable-fn
            ;;                      :max-set-fn max-set-fn)

            ;;   (when (or (equal progress-fn :brief)
            ;;             (equal progress-fn :summary))
            ;;     (format t "New best DAG = ~a: ~a~%" swap-best-dag swap-best-score))
          
            ;;   (if (> (/ (abs (- swap-best-score score-record))
            ;;             (abs swap-best-score))
            ;;          relative-tol)
            ;;       (setf outer-converged nil)
            ;;       (setf outer-converged t))
          
            ;;   ;; Store new record
            ;;   (setf score-record swap-best-score)
          
            ;;   ;; Set the GP to follow the best DAG
            ;;   (loop for (index . parents) in swap-best-dag do
            ;;     (configure-for-factor gp index parents)
            ;;     (update-parameter-vector (nth index (constituent-gps gp))
            ;;                              (gethash parents
            ;;                                       (cdr (assoc index swap-factor-params))))))
            )
          
          (when (and inner-converged
                     outer-converged)
            (return nil))
          ;; ))
          ))
    total-liks))


(defgeneric EM-on-structure (gp &key progress-fn max-iters reach-dag
                                  relative-tol factor-params)
  (:documentation "Applies EM to a specified structure, iterating until convergence.")
  (:method ((gp dag-gp-base) &key (progress-fn nil) (max-iters nil) (reach-dag nil)
                               (relative-tol 0.05) (factor-params nil))
    ;; Convergence parameters
    (let ((sub-progress-fn (if (equal progress-fn :verbose)
                               :verbose
                               nil))
          (LL-record nil)
          new-LL
          converged)
      ;; Set likelihood record to be 0 to start
      (loop for gp-index below (output-dim gp) do
        (setf LL-record (acons gp-index 0d0 LL-record)))

      ;; If factor-params does exist, parameters for the chosen factors necesarily
      ;; exist, so use them
      (if factor-params
          (loop for i below (output-dim gp) do
            (update-parameter-vector (nth i (constituent-gps gp))
                                     (gethash (cdr (assoc i (factors gp)))
                                              (cdr (assoc i factor-params)))))

          ;; If factor-params does not exist, make it empty
          (loop for i below (output-dim gp) do
            (setf factor-params (acons i (make-hash-table :test #'equalp)
                                       factor-params))))
      
      (loop for iter from 0 do
        (update-combined-distributions gp)
        (setf converged t)

        (when reach-dag
          (return nil))
        
        (when (or (equal progress-fn :verbose)
                  (equal progress-fn :summary))
          (format t "~%Iter: ~a~%" iter))
      
        (loop for constituent-gp in (constituent-gps gp)
              for gp-index from 0
              do (when (or (equal progress-fn :summary))      
                   (format t "Output ~a: ~a -> "
                           gp-index
                           (cdr (assoc gp-index LL-record))))
                 
                 ;; Retrain
                 (setf new-LL (train constituent-gp
                                     :progress-fn sub-progress-fn
                                     :relative-tol relative-tol))
                 
                 ;; Compute relative convergence
                 (when (> (/ (abs (- new-LL (cdr (assoc gp-index LL-record))))
                             (abs new-LL))
                          relative-tol)
                   (setf converged nil))
                 ;; Store new values
                 (setf (cdr (assoc gp-index LL-record)) new-LL)
               
                 (when (equal progress-fn :summary)
                   (format t "~a~%" new-LL)))

        ;; Break loop if converged
        (when converged
          (return nil))
        (when (and max-iters
                   (equal iter (1- max-iters)))
          (return nil)))

      ;; ;; Construct an a-list of converged log likelihoods
      ;; (loop for gp-index below (list-length (constituent-gps gp)) do
      ;;   (setf final-LL (acons gp-index (make-hash-table :test #'equalp) final-LL)
      ;;         (gethash (cdr (assoc gp-index (factors gp)))
      ;;                  (cdr (assoc gp-index final-LL)))
      ;;         (cdr (assoc gp-index LL-record))))

      ;; Update factor-params
      (loop for (var . parents) in (factors gp) do
        (setf (gethash parents (cdr (assoc var factor-params)))
              (copy-array (param-vec (nth var (constituent-gps gp))))))

      (list converged factor-params))))


(defgeneric train-closed-downwards-structure (gp &key progress-fn
                                                   relative-tol factor-params
                                                   max-set-fn)
  (:documentation "Trains a specified structure, without iteration.")
  (:method ((gp dag-gp-base) &key (progress-fn nil)
                               (relative-tol 0.05) (factor-params nil)
                               (max-set-fn nil))
    ;; Convergence parameters
    (let ((sub-progress-fn (if (equal progress-fn :verbose)
                               :verbose
                               nil))
          new-LL
          parent-local-obs
          parent-obs rest-parent-obs
          parent-loc rest-parent-locs
          prediction)
      
      ;; If factor-params does exist, parameters for the chosen factors necesarily
      ;; exist, so use them
      ;; (if factor-params
      ;;     (loop for i below (output-dim gp) do
      ;;       (update-parameter-vector (nth i (constituent-gps gp))
      ;;                                (gethash (cdr (assoc i (factors gp)))
      ;;                                         (cdr (assoc i factor-params))))))

      ;; If factor-params does not exist, make it empty
      (loop for i below (output-dim gp) do
        (setf factor-params (acons i (make-hash-table :test #'equalp)
                                   factor-params)))
      
      ;; Don't call update-combined-distributions. Update one by one.
      ;;(update-combined-distributions gp)
      (unless (factor-ordering-up-to-date gp)
        (solve-factor-ordering gp))
      
      (loop for gp-index in (factor-ordering gp)
            for constituent-gp = (nth gp-index (constituent-gps gp))

            ;; Regenerate distance matrices, using only local locs
            do (make-observed-distance-matrices constituent-gp :use-all-locs nil)
            
               ;; ;; Perform a train and impute process only when directed
               ;; (when (member gp-index (impute-indices gp))
                 
               ;; Make the var-parent-distributions using only n-true-obs for this
               ;; constituent
               (setf (var-parent-distributions constituent-gp)
                     (list* (make-mat (n-true-obs constituent-gp)
                                      :ctype :double
                                      :initial-contents (true-obs constituent-gp))
                            (loop for parent in (cdr (assoc gp-index (factors gp)))
                                  for parent-gp = (nth parent (constituent-gps gp))
                                  do (setf parent-local-obs
                                           (make-array (n-true-obs constituent-gp)
                                                       :element-type 'double-float)
                                           parent-obs (first (true-obs parent-gp))
                                           rest-parent-obs (rest (true-obs parent-gp))
                                           parent-loc (first (true-obs-locs parent-gp))
                                           rest-parent-locs (rest (true-obs-locs parent-gp)))
                                     ;; Here we exploit the fact that true-obs-locs
                                     ;; are sorted, so we can search for locations in
                                     ;;sequence
                                     (loop for loc in (true-obs-locs constituent-gp)
                                           for i from 0
                                           ;; Find the local loc in the parent
                                           ;; observed locs
                                           do (loop while (not (equalp loc parent-loc)) do
                                             (setf parent-obs (first rest-parent-obs)
                                                   rest-parent-obs (rest rest-parent-obs)
                                                   parent-loc (first rest-parent-locs)
                                                   rest-parent-locs (rest rest-parent-locs)))
                                              ;; Add the parent observation to the
                                              ;; array
                                              (setf (aref parent-local-obs i) parent-obs))
                                  collect (array-to-mat parent-local-obs
                                                        :ctype :double))))
               
               ;; Perform a train and impute process only when directed
               (when (member gp-index (impute-indices gp))
                 
                 (when (or (equal progress-fn :summary))      
                   (format t "Imputing output ~a: " gp-index))
                 
                 ;; Retrain
                 (setf new-LL (train constituent-gp
                                     :progress-fn sub-progress-fn
                                     :relative-tol relative-tol))
                 
                 (when (equal progress-fn :summary)
                   (format t "~a~%" new-LL))
                 
                 ;; Impute across whole domain if necessary, using only local locs
                 (destructuring-bind (f-mean f-cov)
                     (make-predictive-posteriors constituent-gp
                                                 (constituent-locs gp)
                                                 :closed-downwards-p t
                                                 :use-all-locs nil)
                   (declare (ignore f-cov))
                   
                   ;; Set the unseen observations based on the mean
                   (with-facets ((f-mean-array (f-mean 'backing-array
                                                       :direction :input)))
                     (loop for i from 0
                           for loc in (obs-locs constituent-gp)
                           for obs in (obs constituent-gp)
                           for con-obs = (gethash loc (constituent-obs gp))
                           for con-loc-obs = (gethash loc (constituent-loc-obs gp))
                           ;; When the observation is listed as :dist, replace it
                           do (when (equal obs :dist)
                                ;; Start with predicted f
                                (setf prediction (aref f-mean-array i))
                                ;; Add the parent y values
                                (loop for parent-index from 0
                                      for parent in (cdr (assoc gp-index (factors gp)))
                                      for parent-gp = (nth parent (constituent-gps gp))
                                      do (incf prediction
                                               (* (aref (nth parent-index
                                                             (parent-params
                                                              constituent-gp))
                                                        0)
                                                  (nth i (obs parent-gp)))))
                                ;; Finally, replace the observation
                                (setf (nth i (obs constituent-gp)) prediction)
                                
                                ;; Update constituent-obs and constituent-loc-obs
                                (setf (gethash loc (constituent-obs gp))
                                      (sort (list* gp-index con-obs) #'<))
                                (setf (gethash loc (constituent-loc-obs gp))
                                      (acons gp-index prediction con-loc-obs))))
                     
                     ;; Update true count, locations, and observations, which now equal
                     ;; the overall values.
                     (setf (n-true-obs constituent-gp) (n-obs constituent-gp)
                           (true-obs-locs constituent-gp)
                           (copy-list (obs-locs constituent-gp))
                           (true-obs constituent-gp)
                           (copy-list (obs constituent-gp)))))))
      
      ;; Update factor-params
      (loop for (var . parents) in (factors gp) do
        (setf (gethash parents (cdr (assoc var factor-params)))
              (copy-array (param-vec (nth var (constituent-gps gp))))))
      
      ;; Set combined-dists, so configure-for-factor works as intended
      (setf (combined-dists gp) t)

      factor-params)))


(defgeneric train-all-factors (gp &key progress-fn factor-params relative-tol allowable-fn)
  (:method ((gp dag-gp-base) &key (progress-fn nil) (factor-params nil)
                               (relative-tol 0.02) (allowable-fn nil))
    (let ((sub-progress-fn (if (equal progress-fn :verbose)
                               :verbose
                               nil))
          (liks nil)
          possible-parents
          (outputs (loop for gp-index below (list-length (outputs gp))
                         collect gp-index)))
      (loop for gp-index in outputs
            for constituent-gp in (constituent-gps gp)
            do (setf liks (acons gp-index (make-hash-table :test #'equalp)
                                 liks)
                     possible-parents (sort (powerset (remove gp-index outputs))
                                            #'<
                                            :key #'list-length))
               (loop for parents in possible-parents do
                 (when (or (not allowable-fn)
                           (funcall allowable-fn gp-index parents))
                   (configure-for-factor gp gp-index parents
                                         :use-all-locs
                                         (not (dag-closed-downwards-p gp)))
                   (when (equal progress-fn :summary)
                     (format t "Training ~a <- ~a: " gp-index parents))
                   ;; When the factors exist, use-them as a hot-start
                   (if (gethash parents (cdr (assoc gp-index factor-params)))
                       (update-parameter-vector (nth gp-index (constituent-gps gp))
                                                (gethash parents
                                                         (cdr (assoc gp-index
                                                                     factor-params))))
                       ;; When they don't, look for the smallest superset available
                       (let ((superset nil)
                             superset-params
                             new-params
                             (count 0))
                         (loop for par being the hash-keys of (cdr (assoc gp-index
                                                                          factor-params))
                                 using (hash-value params)
                               do (when (and (subsetp parents par)
                                             (or (not superset)
                                                 (< (list-length par)
                                                    (list-length superset))))
                                    (setf superset par
                                          superset-params params)))
                         (when superset
                           (setf new-params
                                 (make-array (n-gp-params (nth gp-index
                                                               (constituent-gps gp)))
                                             :element-type 'double-float))
                           (dotimes (i (- (n-gp-params (nth gp-index
                                                            (constituent-gps gp)))
                                          (list-length parents)))
                             (setf (aref new-params i) (aref superset-params i))
                             (incf count))
                           (loop for par in superset
                                 for param-ind from count 
                                 do (when (member par parents)
                                      (setf (aref new-params count)
                                            (aref superset-params param-ind))
                                      (incf count)))
                           (update-parameter-vector (nth gp-index (constituent-gps gp))
                                                    new-params))))
                   
                   
                   (setf (gethash parents (cdr (assoc gp-index liks)))
                         (train constituent-gp
                                :progress-fn sub-progress-fn
                                :relative-tol relative-tol)
                         (gethash parents (cdr (assoc gp-index factor-params)))
                         (copy-array (param-vec constituent-gp)))
                   (when (equal progress-fn :summary)
                     (format t "~a~%" (gethash parents
                                               (cdr (assoc gp-index
                                                           liks))))))))
      liks)))


(defun exhaustive-best-dag (factor-liks assigned-vars remaining-vars
                            partial-dag partial-score)
  (unless remaining-vars
    (format t "DAG ~a has score ~a~%" partial-dag partial-score)
    (return-from exhaustive-best-dag (list partial-dag partial-score)))
  (let ((best-dag nil)
        (best-score most-negative-double-float))
    (loop for var in remaining-vars
          for new-assigned = (sort (append assigned-vars (list var)) #'<)
          for new-remaining = (remove var remaining-vars)
          do (loop for possible-parents in (powerset assigned-vars)
                   for new-dag = (acons var possible-parents partial-dag)
                   for new-score = (+ partial-score
                                      (gethash possible-parents
                                               (cdr (assoc var factor-liks)))
                                      (- (* 2 (list-length possible-parents))))
                   do (destructuring-bind (candidate-dag candidate-score)
                          (exhaustive-best-dag factor-liks new-assigned
                                               new-remaining new-dag new-score)
                        (when (> candidate-score best-score)
                          (setf best-score candidate-score
                                best-dag candidate-dag)))))
    (list best-dag best-score)))


(defun powerset (lst)
  (if lst
      (mapcan (lambda (el) (list (cons (car lst) el) el))
              (powerset (cdr lst)))
      '(())))


(defun dag-equalp (dag-1 dag-2 all-vars)
  (loop for var in all-vars do
    (unless (equalp (cdr (assoc var dag-1))
                    (cdr (assoc var dag-2)))
      (return-from dag-equalp nil)))
  t)


(defun astar-best-dag (gp &key (progress-fn nil) (factor-params nil)
                            (relative-tol 0.02) (allowable-fn nil))
  (let ((initial-node (make-initial-search-node (output-dim gp)))
        ;;(score (make-instance 'gp-aic-score :n-vars (output-dim gp)))
        (score (make-instance 'gp-bic-score
                              :n-vars (output-dim gp)
                              :n-data-points (hash-table-count (constituent-obs gp))))
        (k-groups (loop for i below (output-dim gp) collect (list i)))
        (liks (train-all-factors gp
                                 :progress-fn progress-fn
                                 :factor-params factor-params
                                 :relative-tol relative-tol
                                 :allowable-fn allowable-fn))
        out
        out-node)
    (setf out (astar-search initial-node score liks k-groups allowable-fn)
          out-node (first out))
    (when (or (equal progress-fn :brief)
                (equal progress-fn :summary))
        (format t "Evaluated ~a likelihoods~%"
                (loop for (var . hash) in (third out)
                      sum (hash-table-count hash))))
    (list (edges out-node)
          (- (g out-node))
          factor-params
          (loop for (var . hash) in (third out)
                sum (hash-table-count hash)))))


(defun abc-best-dag (gp &key (progress-fn nil) (prev-dag nil)
                          (allowable-fn nil) (max-set-fn nil) (factor-params nil)
                          (relative-tol 0.02))
  (let ((sub-progress-fn (if (equal progress-fn :verbose)
                             :verbose
                             nil))
        (initial-node (make-initial-search-node (output-dim gp)))
        ;; (score (make-instance 'gp-aic-score :n-vars (output-dim gp)))
        (score (make-instance 'gp-bic-score
                              :n-vars (output-dim gp)
                              :n-data-points (hash-table-count (constituent-obs gp))))
        (k-groups (loop for i below (output-dim gp) collect (list i)))
        out
        out-node)
    ;; (loop for i below (output-dim gp) do
    ;;   (setf factor-params (acons i (make-hash-table :test #'equalp) factor-params)))
    
    (flet ((compute-fn (gp-index parents)
             (let (lik)
               (configure-for-factor gp gp-index parents
                                     :use-all-locs
                                     (not (dag-closed-downwards-p gp)))
               (when (equal progress-fn :summary)
                 (format t "Training ~a <- ~a: " gp-index parents))
               ;; When the factors exist, use-them as a hot-start
               (if (gethash parents (cdr (assoc gp-index factor-params)))
                   (update-parameter-vector (nth gp-index (constituent-gps gp))
                                            (gethash parents
                                                     (cdr (assoc gp-index
                                                                 factor-params))))
                   ;; When they don't, look for the smallest superset available
                   (let ((superset nil)
                         superset-params
                         new-params
                         (count 0))
                     (loop for par being the hash-keys of (cdr (assoc gp-index
                                                                      factor-params))
                             using (hash-value params)
                           do (when (and (subsetp parents par)
                                         (or (not superset)
                                             (< (list-length par)
                                                (list-length superset))))
                                (setf superset par
                                      superset-params params)))
                     (when superset
                       (setf new-params
                             (make-array (n-gp-params (nth gp-index
                                                           (constituent-gps gp)))
                                         :element-type 'double-float))
                       (dotimes (i (- (n-gp-params (nth gp-index
                                                        (constituent-gps gp)))
                                      (list-length parents)))
                         (setf (aref new-params i) (aref superset-params i))
                         (incf count))
                       (loop for par in superset
                             for param-ind from count 
                             do (when (member par parents)
                                  (setf (aref new-params count)
                                        (aref superset-params param-ind))
                                  (incf count)))
                       (update-parameter-vector (nth gp-index (constituent-gps gp))
                                                new-params))))
               
               (setf lik (train (nth gp-index (constituent-gps gp))
                                :progress-fn sub-progress-fn
                                :relative-tol relative-tol)
                     (gethash parents (cdr (assoc gp-index factor-params)))
                     (copy-array (param-vec (nth gp-index (constituent-gps gp)))))
               (when (equal progress-fn :summary)
                 (format t "~a~%" lik))
               lik)))
      (setf out (abc-search initial-node score #'compute-fn k-groups
                            prev-dag allowable-fn max-set-fn)
            out-node (first out))
      (when (or (equal progress-fn :brief)
                (equal progress-fn :summary))
        (format t "Evaluated ~a likelihoods~%"
                (loop for (var . hash) in (third out)
                      sum (hash-table-count hash))))
      (list (edges out-node)
            (- (g out-node))
            factor-params
            (loop for (var . hash) in (third out)
                  sum (hash-table-count hash))))))


(defun tabu-best-dag (gp &key (progress-fn nil) (max-restarts 5) (reach-dag nil)
                           (prev-dag nil) (allowable-fn nil) (factor-params nil)
                           (relative-tol 0.02))
  (let ((sub-progress-fn (if (equal progress-fn :verbose)
                             :verbose
                             nil))
        ;; (score (make-instance 'gp-aic-score :n-vars (output-dim gp)))
        (score (make-instance 'gp-bic-score
                              :n-vars (output-dim gp)
                              :n-data-points (hash-table-count (constituent-obs gp)))))
    ;; (loop for i below (output-dim gp) do
    ;;   (setf factor-params (acons i (make-hash-table :test #'equalp) factor-params)))
    (flet ((compute-fn (gp-index parents)
             (let (lik)
               (configure-for-factor gp gp-index parents
                                     :use-all-locs
                                     (not (dag-closed-downwards-p gp)))
               (when (equal progress-fn :summary)
                 (format t "Training ~a <- ~a: " gp-index parents))
               ;; When the factors exist, use-them as a hot-start
               (if (gethash parents (cdr (assoc gp-index factor-params)))
                   (update-parameter-vector (nth gp-index (constituent-gps gp))
                                            (gethash parents
                                                     (cdr (assoc gp-index
                                                                 factor-params))))
                   ;; When they don't, look for the smallest superset available
                   (let ((superset nil)
                         superset-params
                         new-params
                         (count 0))
                     (loop for par being the hash-keys of (cdr (assoc gp-index
                                                                      factor-params))
                             using (hash-value params)
                           do (when (and (subsetp parents par)
                                         (or (not superset)
                                             (< (list-length par)
                                                (list-length superset))))
                                (setf superset par
                                      superset-params params)))
                     (when superset
                       (setf new-params
                             (make-array (n-gp-params (nth gp-index
                                                           (constituent-gps gp)))
                                         :element-type 'double-float))
                       (dotimes (i (- (n-gp-params (nth gp-index
                                                        (constituent-gps gp)))
                                      (list-length parents)))
                         (setf (aref new-params i) (aref superset-params i))
                         (incf count))
                       (loop for par in superset
                             for param-ind from count 
                             do (when (member par parents)
                                  (setf (aref new-params count)
                                        (aref superset-params param-ind))
                                  (incf count)))
                       (update-parameter-vector (nth gp-index (constituent-gps gp))
                                                new-params))))
               
               (setf lik (train (nth gp-index (constituent-gps gp))
                                :progress-fn sub-progress-fn
                                :relative-tol relative-tol)
                     (gethash parents (cdr (assoc gp-index factor-params)))
                     (copy-array (param-vec (nth gp-index (constituent-gps gp)))))
               (when (equal progress-fn :summary)
                 (format t "~a~%" lik))
               lik)))
      (destructuring-bind (best-dag best-score scores)
          (tabu-search score #'compute-fn
                       :reach-dag reach-dag
                       :candidate-dag prev-dag
                       :allowable-fn allowable-fn
                       :max-restarts max-restarts)
        (when (or (equal progress-fn :brief)
                  (equal progress-fn :summary))
          (format t "Evaluated ~a likelihoods~%"
                  (loop for (var . hash) in scores
                        sum (hash-table-count hash))))
        (list best-dag
              (- best-score)
              factor-params
              (loop for (var . hash) in scores
                    sum (hash-table-count hash)))))))
