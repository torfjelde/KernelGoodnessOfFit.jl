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

### Derivatives
# These functions can be implemented manually for particular kernels were analytic expressions are easily attainble,
# The below allow the use to implement any arbitrary kernel, and then we can optimize over it.

k_dx(k::Kernel, x::AbstractArray, y::AbstractArray) = ForwardDiff.gradient(z -> kernel(k, z, y), x);
k_dy(k::Kernel, x::AbstractArray, y::AbstractArray) = ForwardDiff.gradient(z -> kernel(k, x, z), y);
k_dxdy(k::Kernel, x::AbstractArray, y::AbstractArray) = ForwardDiff.jacobian(z -> k_dx(k, x, z), y);

k_dx(k::Kernel, x::Real, y::Real) = ForwardDiff.derivative(z -> kernel(k, z, y), x);
k_dy(k::Kernel, x::Real, y::Real) = ForwardDiff.derivative(z -> kernel(k, x, z), y);
k_dxdy(k::Kernel, x::Real, y::Real) = ForwardDiff.derivative(z -> k_dx(k, x, z), y);


### Gaussian Radial Basis Function ###
struct GaussianRBF{T} <: Kernel
    gamma::T
end

@inline params(k::GaussianRBF, gamma) = (gamma, )
@inline update(k::GaussianRBF, gamma) = GaussianRBF(gamma)
@inline update(k::GaussianRBF, gamma::AbstractArray) = update(k, first(gamma))

@inline function kernel(k::GaussianRBF, x::AbstractVector, y::AbstractVector)
    # added factor of 0.5 to get same result as code attached to paper
    return exp(- 0.5 * k.gamma^(-2) * sum((x - y).^2))
end

@inline function k_dx(k::GaussianRBF, x::AbstractArray, y::AbstractArray)
    #   ∂₁ᵏ(exp(- 0.5 γ⁻² ∑ᵢ (xⁱ - yⁱ)^2))
    # = exp(- 0.5 γ⁻² ∑ᵢ (xⁱ - yⁱ)^2) ∂₁ᵏ (- 0.5 γ⁻² ∑ᵢ (xⁱ - yⁱ)^2)
    # = - 0.5 γ⁻² exp(- 0.5 γ⁻² ∑ᵢ (xⁱ - yⁱ)^2) ∂₁ᵏ (∑ᵢ (xⁱ - yⁱ)^2)
    # = - 0.5 γ⁻² * k(x, y) * 2 * (xᵏ - yᵏ)
    # = - γ⁻² * k(x, y) * (xᵏ - yᵏ)
    return - k.gamma^(-2) * (x - y) .* kernel(k, x, y)
end

@inline k_dy(k::GaussianRBF, x::AbstractArray, y::AbstractArray) = k_dx(k, y, x)
@inline function k_dxdy(k::GaussianRBF, x::AbstractArray, y::AbstractArray)
    #   ∂₂ⁱ (∂₁ʲ k(x, y))
    # = ∂₂ⁱ (- γ⁻² * k(x, y) * (xʲ - yʲ))
    # = - γ⁻² [ (∂₂ⁱ k(x, y)) * (xʲ - yʲ) + k(x, y) * ∂₂ⁱ (xʲ - yʲ) ]
    # FIXME: there's a sign-error here somewhere (though code is correct because I've checked with ForwardDiff:) )
    # = - γ⁻² [ - γ⁻² k(x, y) * (xⁱ - yⁱ) * (xʲ - yʲ) + k(x, y) * δⁱʲ]
    # = - γ⁻² k(x, y) [ - γ⁻²  (xⁱ - yⁱ) * (xʲ - yʲ) + δⁱʲ]
    # = γ⁻² k(x, y) [ γ⁻²  (xⁱ - yⁱ) * (xʲ - yʲ) - δⁱʲ]
    Δ = (x - y)
    γ⁻² = k.gamma^(-2)
    return - (γ⁻² .* (Δ * Δ') - Diagonal(ones(length(x)))) .* γ⁻² .* kernel(k, x, y)
end

### Exponential kernel
struct ExponentialKernel <: Kernel end
@inline function kernel(k::ExponentialKernel, x::AbstractVector, y::AbstractVector)
    return exp(dot(x, y))
end
@inline update(k::ExponentialKernel) = k

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
@inline params(k::Matern25Kernel) = (k.ρ, )
@inline update(k::Matern25Kernel, ρ) = Matern25Kernel(ρ)

### InverseMultiQuadratic (IMQ) kernel
struct InverseMultiQuadratic <: Kernel
    c  # > 0
    b  # < 0
end

@inline function kernel(k::InverseMultiQuadratic, x::AbstractVector, y::AbstractVector)
    (k.c^2 + norm(x - y)^2)^k.b
end
@inline params(k::InverseMultiQuadratic) = (k.c, k.b)
@inline update(k::InverseMultiQuadratic, c, b) = InverseMultiQuadratic(c, b)
