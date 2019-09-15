using LinearAlgebra
import SpecialFunctions: gamma, besselk

import Distributions: params

# General
abstract type Kernel end

function kernel(k::Kernel, x::AbstractVector, y::AbstractVector) end
kernel(k::Kernel, x::Number, y::Number) = kernel(k, [x], [y])

"""
    params(k::Kernel)

Returns a tuple of parameters for the kernel `k`.
"""
params(k::Kernel)

"""
    update(k::Kernel, args...)

Returns a kernel of same type with the new parameters `args`.
"""
update(k::Kernel, args...)

# partial evaluations
kernel(k::Kernel, x::AbstractVector) = begin
    y -> kernel(k, x, y)
end

kernel(k::Kernel) = begin
    (x, y) -> kernel(k, x, y)
end

### Gaussian Radial Basis Function ###
struct GaussianRBF <: Kernel
    gamma
end

params(k::GaussianRBF, gamma) = (gamma, )
update(k::GaussianRBF, gamma) = GaussianRBF(gamma)
update(k::GaussianRBF, gamma::AbstractArray) = update(k, first(gamma))

kernel(k::GaussianRBF, x::AbstractVector, y::AbstractVector) = begin
    # added factor of 0.5 to get same result as code attached to paper
    exp(- 0.5 * k.gamma^(-2) * sum((x - y).^2))
end

kernel(k::GaussianRBF, x::AbstractVector, y::AbstractVector, γ::Number) = begin
    # added factor of 0.5 to get same result as code attached to paper
    exp(- 0.5 * γ^(-2) * sum((x - y).^2))
end

### Exponential kernel
struct ExponentialKernel <: Kernel end
kernel(k::ExponentialKernel, x::AbstractVector, y::AbstractVector) = begin
    exp(dot(x, y))
end
update(k::ExponentialKernel) = k

# Works with FSSDrand and FSSDopt for SOME values, and only using `finite-differences` to optimize.
"""
    Matern25Kernel(ρ)

Matern25Kernel is a Matern kernel with ν = 2.5, and is thus 1st order differentiable, which is what we require.

# Notes
Optimization wrt. ρ is potentially unstable, and so some runs might return an error, while others wont.
"""
struct Matern25Kernel{T} <: Kernel
    ρ::T
end

@inline function kernel(κ::Matern25Kernel, x::AbstractVector, y::AbstractVector)
    ν = 2.5
    
    # squared euclidean distance
    d = norm(x - y)
    # d = d < eps(T) ? eps(T) : d  # If d is zero, besselk will return NaN
    d = d < eps(Float64) ? eps(Float64) : d
    tmp = √(2ν)*d /κ.ρ
    return (2^(1.0 - ν)) * (tmp^ν) * besselk(ν, tmp) / gamma(ν)
end
params(k::Matern25Kernel) = (k.ρ, )
update(k::Matern25Kernel, ρ) = Matern25Kernel(ρ)

### InverseMultiQuadratic (IMQ) kernel
struct InverseMultiQuadratic <: Kernel
    c  # > 0
    b  # < 0
end

function kernel(k::InverseMultiQuadratic, x::AbstractVector, y::AbstractVector)
    (k.c^2 + norm(x - y)^2)^k.b
end
params(k::InverseMultiQuadratic) = (k.c, k.b)
update(k::InverseMultiQuadratic, c, b) = InverseMultiQuadratic(c, b)

### Derivatives
# These functions can be implemented manually for particular kernels were analytic expressions are easily attainble,
# The below allow the use to implement any arbitrary kernel, and then we can optimize over it.

k_dx(k::Kernel, x::AbstractArray, y::AbstractArray) = ForwardDiff.gradient(z -> kernel(k, z, y), x);
k_dy(k::Kernel, x::AbstractArray, y::AbstractArray) = ForwardDiff.gradient(z -> kernel(k, x, z), y);
k_dxdy(k::Kernel, x::AbstractArray, y::AbstractArray) = ForwardDiff.jacobian(z -> k_dx(k, x, z), y);

k_dx(k::Kernel, x::Number, y::Number) = ForwardDiff.derivative(z -> kernel(k, z, y), x);
k_dy(k::Kernel, x::Number, y::Number) = ForwardDiff.derivative(z -> kernel(k, x, z), y);
k_dxdy(k::Kernel, x::Number, y::Number) = ForwardDiff.derivative(z -> k_dx(k, x, z), y);
