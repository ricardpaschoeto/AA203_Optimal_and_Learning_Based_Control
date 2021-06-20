# hj_reachability: Hamilton-Jacobi reachability analysis in [JAX]
This package implements numerical solvers for Hamilton-Jacobi (HJ) Partial Differential Equations (PDEs) which, in the context of optimal control, may be used to represent the continuous-time formulation of dynamic programming. Specifically, the focus of this package is on reachability analysis for zero-sum differential games modeled as Hamilton-Jacobi-Isaacs (HJI) PDEs, wherein an optimal controller and (optional) disturbance interact, and the set of reachable states at any time is represented as the zero sublevel set of a value function realized as the viscosity solution of the corresponding PDE.

This package is inspired by a number of related projects, including:

- [A Toolbox of Level Set Methods (`toolboxls`, MATLAB)](https://www.cs.ubc.ca/~mitchell/ToolboxLS/)
- [An Optimal Control Toolbox for Hamilton-Jacobi Reachability Analysis (`helperOC`, MATLAB)](https://github.com/HJReachability/helperOC)
- [Berkeley Efficient API in C++ for Level Set methods (`beacls`, C++/CUDA C++)](https://hjreachability.github.io/beacls/)
- [Optimizing Dynamic Programming-Based Algorithms (`optimized_dp`, python)](https://github.com/SFU-MARS/optimized_dp)

## Installation
To accommodate different [JAX] versions (i.e., CPU-only vs. JAX with GPU support), this package does not list JAX as a dependency in `requirements.txt`. Therefore, to use this package you must first install JAX (with your preferred accelerator support); see the [JAX installation instructions](https://github.com/google/jax#installation) for details.

Then, install this package using pip:
```
pip install --upgrade git+https://github.com/StanfordASL/hj_reachability
```

## TODOs
Aside from the specific TODOs scattered throughout the codebase, a few general TODOs:
- Single-line docstrings (at a bare minimum) for everything. Test coverage, book/paper references, and proper documentation to come... eventually.
- Register on [PyPI](https://pypi.org/)? Might be prudent to change the package name (e.g., to `hjax`) in that case.
- Look into using `jax.pmap`/`jax.lax.ppermute` for multi-device parallelism; see, e.g., [jax demo notebooks](https://github.com/google/jax/tree/master/cloud_tpu_colabs).
- Incorporate neural-network-based PDE solvers; see, e.g., [Bansal, S. and Tomlin, C. "DeepReach: A Deep Learning Approach to High-Dimensional Reachability." (2020)](https://arxiv.org/abs/2011.02082).

[JAX]: https://github.com/google/jax
