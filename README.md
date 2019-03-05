# `KernelGoodnessOfFit.jl`

This package provides implementations goodness-of-fit tests based on Stein discrepancy and Reproducing Kernel Hilbert Spaces (RKHSs). At the moment, the goodness-of-fit tests supported are
- `KSD`: goodness-of-fit test based on Kernelized Stein Discrepancy (KSD) with bootstrapping [1]
- `FSSDrand` and `FSSDopt`: goodness-of-fit test based on Finite-Set Stein Discrepancy (FSSD), both with randomized and optimized test-parameters [2]

Due to the nature of the tests and implementation, we allow user-specified distributions and kernels.

# Usage
## Univariate Gaussian using `FSSDopt`
```julia
using Distributions # provides a large number of standard distributions
using KernelGoodnessOfFit

p = Normal(0.0, 1.0) # true distribution
xs = reshape(rand(p, 100), 1, 100);  # expects samples in shape (d, n) for d-dimensional data

q = Normal(1.0, 1.0) # proposal distribution

FSSDopt(xs, q)
```

This returns something similar to:

```
Finite-Set Stein Discrepancy optimized (FSSD-opt)
-------------------------------------------------
Population details:
    parameter of interest:   Finite-Set Stein Discrepancy (FSSD)
    value under h_0:         0
    point estimate:          3.0167943827814967

Test summary:
    outcome with 95% confidence: reject h_0
    p-value:                     0.0077

Details:
    kernel:           GaussianRBF(0.43228083254028116)
    test locations:   [2.10232 -0.323255 … -1.39591 0.716078]
    num. simulate H₀: 3000
```

## Different kernels
The package include the following (universal) kernels:
- `GaussianRBF(γ)` (default)
- `InverseMultiQuadratic(c, b)` (c > 0 and b < 0, which is enforced by optimization process)
- `Matern25Kernel(ν=2.5, ρ)` (the test requires once-differentiability, therefore ν = 2.5 is fixed)

```julia
using Distributions
using KernelGoodnessOfFit

p = Normal(0.0, 1.0)
xs = reshape(rand(p, 100), 1, 100);

q = Normal(1.0, 1.0)

FSSDopt(xs, q, KernelGoodnessOfFit.GaussianRBF(1.0))
FSSDopt(xs, q, KernelGoodnessOfFit.InverseMultiQuadratic(1.0, -0.5))
```

### Custom kernels
```julia
using Distributions, KernelGoodnessOfFit

import KernelGoodnessOfFit: Kernel, kernel, get_params, set_params! # allows extending `kernel` method

p = Normal(0.0, 1.0)
xs = reshape(rand(p, 100), 1, 100);

q = Normal(1.0, 1.0)

# define kernel struct; `mutable` to allow updating parameters in optimization process
mutable struct CustomGaussianRBF <: Kernel
    γ
    custom_constant
end
CustomGaussianRBF(γ) = CustomGaussianRBF(γ, 5.0) # default constructor

# defines how evaluation of kernel
kernel(k::CustomGaussianRBF, x::AbstractVector, y::AbstractVector) = k.custom_constant * exp(- 0.5 * k.γ^(-2) * sum((x - y).^2))

# following is required for optimization wrt. γ
# NOT for fixed γ, e.g. `FSSDrand` or `FSSDopt(...; test_locations_only=true)`
get_params(k::CustomGaussianRBF) = [k.γ]  # array of parameters to optimize => `custom_constant` stays fixed
set_params!(k::CustomGaussianRBF, γ) = begin k.γ = γ end

FSSDopt(xs, q, CustomGaussianRBF(1.0, 5.0))
```

Output:
```
Finite-Set Stein Discrepancy optimized (FSSD-opt)
-------------------------------------------------
Population details:
    parameter of interest:   Finite-Set Stein Discrepancy (FSSD)
    value under h_0:         0
    point estimate:          3.8215122818979457

Test summary:
    outcome with 95% confidence: reject h_0
    p-value:                     <1e-99

Details:
    kernel:           CustomGaussianRBF(2.770003377834852, 5.0)
    test locations:   [127.592 105.799 … -24.3751 85.3997]
    num. simulate H₀: 3000
```

# References
[1] Liu, Q., Lee, J. D., & Jordan, M. I., A kernelized stein discrepancy for goodness-of-fit tests and model evaluation, CoRR, (),  (2016). 
[2] Jitkrittum, W., Xu, W., Szabo, Z., Fukumizu, K., & Gretton, A., A Linear-Time Kernel Goodness-Of-Fit Test, CoRR, (),  (2017). 
