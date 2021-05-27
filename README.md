# DAG-GP

## Purpose

This package implements the DAG-GP model, as described in the paper "DAG-GPs: Learning Directed Acyclic Graph Structure For Multi-Output Gaussian Processes". A DAG-GP enforces a DAG model between the output dimensions of a multi-output Gaussian process, to control which outputs are correlated. Training a DAG-GP consists of both optimizing parameters and performing structure learning.

This package is implemented in Common Lisp. It has only been tested in SBCL.
 
 
## License

This project is released under the BSD 3-Clause License. This is research level code, provided in an "as is" state.
 

## Dependencies

The following dependencies that are **not in quicklisp** are required to run this package:

- lbfgs-wrapper(https://anonymous.4open.science/r/lbfgs-wrapper-B71D) (BSD 3-clause license)
- dag-search(https://anonymous.4open.science/r/dag-search-ED68) (BSD 3-clause license)
- [liblbfgs](https://github.com/chokkan/liblbfgs) (MIT license)
- [mgl-mat](https://github.com/melisgl/mgl-mat) (MIT license)

In addition, the following dependencies **available in quicklisp** are required:

- [cffi](https://github.com/cffi/cffi) (MIT license)
- [alexandria](https://gitlab.common-lisp.net/alexandria/alexandria) (public domain)
- [array-operations](https://github.com/bendudson/array-operations) (MIT license)
- [lla](https://github.com/tpapp/lla) (Boost license 1.0)
- [fiveam](https://github.com/lispci/fiveam) (Custom permissive license)
- [cl-ppcre](https://github.com/edicl/cl-ppcre) (BSD 2-clause license)

It is also assumed that a Lapack implementation is available in standard install locations.

### Dependencies for Experiment Baselines

Files are included to recreate our experiments on baselines. These are written in Python 3. They do not need to be loaded to use this package, but if you wish to run the baselines you will need the following dependencies **available in pip**:

- [numpy](https://github.com/numpy/numpy) (BSD 3-clause license)
- [matplotlib](https://github.com/matplotlib/matplotlib) (Custom permissive license)
- [pandas](https://github.com/pandas-dev/pandas) (BSD 3-clause license)
- [GPy](https://github.com/SheffieldML/GPy) (BSD 3-clause license)
- [gpar](https://github.com/wesselb/gpar) (MIT License)


## Getting Started

This package can be loaded with quicklisp or asdf. You can test your installation use `(asdf:test-system :dag-gp)`. 


Simple setup and usage looks like this:

```
(use-package :dag-gp)

;; Make a DAG-GP with 3 outputs and a RBF kernel.
(defparameter dag-gp (make-instance 'dag-gp
                                    :output-dim 3
                                    :ref-kernel (make-instance 'rbf-kernel)
                                    :outputs (list (make-instance 'gaussian-output)
                                                   (make-instance 'gaussian-output)
                                                   (make-instance 'gaussian-output))))
                                                   
;; Add data, inputs are always lists (x1 x2 ...)
(add-measurement dag-gp '(0) 0 1d0)   ;; y0(0) = 1
(add-measurement dag-gp '(2) 0 2d0)   ;; y0(2) = 2
(add-measurement dag-gp '(2) 1 1.5d0) ;; y1(2) = 1.5
(add-measurement dag-gp '(5) 1 4d0)   ;; y1(5) = 4
(add-measurement dag-gp '(5) 2 2.5d0) ;; y2(5) = 2.5

;; Initialize prior to training
(initialize-gp dag-gp)

;; Specify an initial structure for use in EM
(configure-for-factor dag-gp 0 nil)    ;; No parents of y0
(configure-for-factor dag-gp 1 nil)    ;; No parents of y1
(configure-for-factor dag-gp 2 '(0 1)) ;; y0 and y1 are parents of y2

;; Train using A*BC structural search
(train dag-gp :progress-fn :summary
              :search :abc
              :relative-tol 0.005)
              
;; Predict at locations (5) and (6)
(defparameter prediction (predict dag-gp '((5) (6))))
```

`predict` returns a predictive mean and covariance matrix of outputs. These are `mat` objects, as specified by mgl-mat. Data is ordered in terms of output, ie

```
mean = [ E[y0(5)] E[y0(6)] E[y1(5)] E[y1(6)] E[y2(5)] E[y2(6)] ]
```

and equivalently for the predictive covariance.


## Advanced Options

### DAG-GPs

The full constructor for the `dag-gp` class is given below.

```
(make-instance 'dag-gp
               :output-dim
               :outputs
               :ref-kernel
               :constituent-gps
               :ref-parent-param
               :ref-distance-fns
               :closed-downwards-p
               :impute-indices)
```

- `:output-dim` (required): Number of output dimensions.
- `:outputs` (required): Should be a list of `gaussian-output` objects. Non-Gaussian outputs may be constructed for use in `variational-dag-gp` objects, but these are not yet released.
- `:ref-kernel` (default `nil`): A kernel to apply to all latent processes.
- `:constituent-gps` (default `nil`): Provided if the user wished to supply a list of latent GPs explicitly, rather than have them constructed by the `dag-gp` object. Useful if the latent processes should have different kernels or other unique settings. Each constituent GP should be a `linear-parent-gp` object, listed in order of outputs, ie `(latent-gp-0 latent-gp-1 ...)`.
- `:ref-parent-param` (default `nil`): If specified, sets the default parent weight (lambda_m,n in the paper) to use at the start of optimization. When not specified, selected randomly.
- `:ref-distance-fns` (default `'(#'squared-distance)`): Inter-input distances are computed and cached for each functions in this list, to then be used when evaluating covariance matrices. `squared-distance` is sufficient for all the basic kernels in this package, which computes the squared euclidean distance between inputs. More complex, custom kernels can make use of any other distances added to this list. Each element of the list must be a function `(fn x1 x2) => dist`. See the kernels section for more details.
- `:closed-downwards-p` (default `nil`): If non-nil, specifies that the data set is closed downwards for all allowable DAG structures. In this case, training is performed without using structural EM, and is much faster as a result, at the expense of not allowing any possible DAG.
- `:impute-indices` (default `nil`): When `closed-downwards-p` is non-nil, the user may supply a list of output indices to impute (replace missing data with mean predictions) to ensure the closed downwards property holds. Imputing is done with the process depending on the parents that are specified by the DAG configuration when `train` is called. **Beware**, this is **not** a technically rigorous thing to do, because imputed predictions are made independently of observations of descendants, while structural EM conditions on those descendant observations. In general, this will result in worse predictions. This is most useful when a dataset is *almost* closed downwards, to replace odd missing data points that are well described by nearby observations, rather than imputing large unobserved sections of data.

The user must either specify `:ref-kernel` and `:outputs`, or `:constituent-gps`. 


### Kernels

This package currently supports a small set of kernels. Kernels are specified in /src/kernel/ and follow a standard API, so adding new kernels should be straightforward. 

As previously mentioned, kernels may be constructed to make use of custom distances. For example, assume there are 3 input dimensions $x_i = (x_{i,1}, x_{i,2}, x_{i,3})$, and we wish to specify a sum of an RBF kernel over the first two dimensions and a RBF kernel over the third dimension:

$k(x_i,x_j) = \sigma_a^2 \exp \left( \frac{(x_{i,1} - x_{j,1})^2 + (x_{i,2} - x_{j,2})^2}{2 l_a^2} \right) + \sigma_b^2 \exp \left( \frac{(x_{i,3} - x_{j,3})^2}{2 l_b^2} \right)$

We can do this by specifying custom distance functions

```
(defun sq-dist-a (xi xj)
  (+ (expt (- (nth 0 xi) (nth 0 xj)) 2)
     (expt (- (nth 1 xi) (nth 1 xj)) 2)))
     
(defun sq-dist-b (xi xj)
  (expt (- (nth 2 xi) (nth 2 xj)) 2))
```

which are fed into the dag-gp with the keyword `:ref-distance-fns (list #'sq-dist-a #'sq-dist-b)`. The kernel would then be constructed as

```
(make-instance 'sum-kernel
               :constituents (list (make-instance 'rbf-kernel
                                                  :dist-index 0)
                                   (make-instance 'rbf-kernel
                                                  :dist-index 1)))
```

Here, `:dist-index i` tells the RBF kernel to use index i of the reference distance functions to evaluate its distances.

The full constructor of the `kernel` superclass is specified below. Specific kernel subclasses should be constructed directly, and may have unique options.

```
(make-instance 'kernel :dist-index
                       :default-kern-params)
```

- `dist-index` (default `0`): Specifies which distance function in the DAG-GP to use to calculate distances.
- `default-kern-params` (default `nil`): If specified, must be an array of double-floats of length `(n-kern-params kernel)`. Sets the kernel parameters to these values at the start of optimization. When not specified, selected randomly. Note that these values are the log-transformed parameters.


### Training

The `train` method has the options specified below.

```
(train dag-gp
       :progress-fn
       :allowable-fn
       :max-set-fn
       :search
       :max-restarts
       :relative-tol)
```

- `:progress-fn` (default `nil`): One of `nil`, `:summary`, `:verbose`. Gives different levels of detail on training. `nil` is silent, `:summary` prints data about each latent GP's likelihood and convergence of structural search, `:verbose` prints full tables of hyperparameters at each stage of latent GP training.
- `:allowable-fn` (default `nil`): Used to restrict the space of possible DAG structures. If specified, must be a function `(allowable-fn output-index parent-indices)`. If this function returns non-nil, the parents given by `parent-indices` are permitted. If not, the parents are not a permissible set.
- `:max-set-fn` (default `nil`): Used when restricting the space of possible DAG structures. If specified, must be a function `(max-set-fn output-index) => parent-indices`. The parent set returned by this function is evaluated at the start of structural EM to provide bounds on all allowable parent sets, so should be the smallest set that is a superset of all parent sets for which `allowable-fn` returns non-nil. If not specified, the set of all other outputs in the DAG-GP is used.
- `:search` (default `:abc`): One of `:abc`, `:astar`, or `:tabu`. Specified that A\*BC, A\*, or tabu search should be used.
- `:max-restarts` (default `0`): How many random restarts to perform when using tabu search.
- `:relative-tol` (default `0.02`): The relative tolerance used for convergence of structural EM.


## Recreating Experiments

Files to perform experiments from the paper are specified in the subfolders of /src/experiments/. Files are named according to the method used. Instructions are provided at the top of each file, with our tested output given below. The experiment functions are **not exported**; to run the Common Lisp experiments you must be `(in-package :package-name)`.
