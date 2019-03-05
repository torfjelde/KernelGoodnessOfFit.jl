# `KernelGoodnessOfFit.jl`

This package provides implementations goodness-of-fit tests based on Stein discrepancy and Reproducing Kernel Hilbert Spaces (RKHSs). At the moment, the goodness-of-fit tests supported are
- `KSD`: goodness-of-fit test based on Kernelized Stein Discrepancy (KSD) with bootstrapping [1]
- `FSSDrand` and `FSSDopt`: goodness-of-fit test based on Finite-Set Stein Discrepancy (FSSD), both with randomized and optimized test-parameters [2]

Due to the nature of the tests and implementation, we allow both user-specified distributions and kernels.

# Usage
## Univariate Gaussian
### `FSSDopt`
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

### `FSSDrand`
```julia
using Distributions # provides a large number of standard distributions
using KernelGoodnessOfFit

p = Normal(0.0, 1.0) # true distribution
xs = reshape(rand(p, 100), 1, 100);  # expects samples in shape (d, n) for d-dimensional data

q = Normal(1.0, 1.0) # proposal distribution

FSSDrand(xs, q)
```

This returns something similar to:

```
Finite-Set Stein Discrepancy randomized (FSSD-rand)
---------------------------------------------------
Population details:
    parameter of interest:   Finite-Set Stein Discrepancy (FSSD)
    value under h_0:         0
    point estimate:          45.570034393743555

Test summary:
    outcome with 95% confidence: reject h_0
    p-value:                     <1e-99

Details:
    kernel:           GaussianRBF(1.1219378550323837)
    test locations:   [0.277374 -0.704376 … 0.329766 -0.739727]
    num. simulate H₀: 3000
```

## Custom distribution
The `KSD` and `FSSD` both only require the `∇log(p(x))`. Therefore, to perform the tests on some arbitrary distribution, we only need to provide an implementation of `gradlogpdf` from `Distributions.jl`.

```julia
using Distributions # provide implementations of `gradlogpdf` for most standard distributions
using KernelGoodnessOfFit

import Distributions: gradlogpdf # allow extensions

μ = 0.0
σ² = 1.0

# true distribution is isotropic Gaussian
p = MultivariateNormal([μ, 0], [σ² 0; 0 σ²])  # true

n = 400
xs = rand(p, n)  # draw samples from `p`

# Gaussian Mixture Model (GMM) with two components as our model distribution
q = MixtureModel(MvNormal[ 
    MvNormal([μ, -2.0], [5 * σ² 0; 0 σ²]),
    MvNormal([μ, 2.0], [3 * σ² 0; 0 3 * σ²]),
    ]
)

# `Distributions.jl` does not provide `gradlogpdf` for mixture models; auto-differentiation using `ForwardDiff.jl` to the rescue!
using ForwardDiff
gradlogpdf(d::MixtureModel, x::AbstractArray) = ForwardDiff.gradient(z -> log(pdf(d, z)), x)

res = FSSDopt(xs, q; J = 2, β_H₁ = 0.001) # using 2 test locations
res
```

This example lends itself nicely to interpretable test locations. Run the following code to visualize the results.

```julia
using Plots


V = res.V

f = scatter(xs[1, :], xs[2, :], markeralpha = 0.5, markerstrokewidth = 0.1, label = "\$x_i \\sim p\$")

for i = 1:size(V, 2)
    v = V[:, i]
    scatter!([v[1]], [v[2]], markerstrokewidth = 0.5, label = "\$v_$i\$", color = "red")
end

x = range(-5, 5, length=100)
y = x

contour!(x, y, (x, y) -> pdf(q, [x; y]), nlevels = 10, width = 0.2)
f
```

## Different kernels
The package include the following (universal) kernels:
- `GaussianRBF(γ)` (default)
- `InverseMultiQuadratic(c, b)` (`c > 0` and `b < 0`, enforced by optimization process even w/o specifying bounds)
- `Matern25Kernel(ν=2.5, ρ)` (the test requires once-differentiability, therefore `ν = 2.5` is fixed)

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
    custom_constant  # arbitrary constant to multiply with
end
CustomGaussianRBF(γ) = CustomGaussianRBF(γ, 5.0) # default constructor

# defines how evaluation of kernel
kernel(k::CustomGaussianRBF, x::AbstractVector, y::AbstractVector) = k.custom_constant * exp(- 0.5 * k.γ^(-2) * sum((x - y).^2))

# following is required for optimization wrt. γ
# NOT required for fixed γ, e.g. `FSSDrand` or `FSSDopt(...; test_locations_only=true)`
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
