using LinearAlgebra
import SpecialFunctions: gamma, besselk

# General
abstract type Kernel end

function kernel(k::Kernel, x::AbstractVector, y::AbstractVector) end
kernel(k::Kernel, x::Number, y::Number) = kernel(k, [x], [y])

function get_params(k::Kernel) end
function set_params!(k::Kernel, params) end

# partial evaluations
kernel(k::Kernel, x::AbstractVector) = begin
    y -> kernel(k, x, y)
end

kernel(k::Kernel) = begin
    (x, y) -> kernel(k, x, y)
end

### Gaussian Radial Basis Function ###
mutable struct GaussianRBF <: Kernel
    gamma
end

get_params(k::GaussianRBF, gamma) = [gamma]
set_params!(k::GaussianRBF, gamma::Number) = begin
    k.gamma = gamma
end
set_params!(k::GaussianRBF, gamma::AbstractArray) = set_params!(k, first(gamma))

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
set_params!(k::ExponentialKernel, anything) = ()

### Matern Kernel
# ### Matern Kernel
# # Works with FSSDrand and FSSDopt for SOME values, and only using `finite-differences` to optimize.
# mutable struct MaternKernel <: Kernel
#     ν
#     ρ
# end

# @inline function kernel(κ::MaternKernel, x::AbstractVector, y::AbstractVector)
#     # squared euclidean distance
#     d = norm(x - y)
#     # d = d < eps(T) ? eps(T) : d  # If d is zero, besselk will return NaN
#     d = d < eps(Float64) ? eps(Float64) : d
#     tmp = √(2κ.ν)*d /κ.ρ
#     return (2^(1.0 - κ.ν)) * (tmp^κ.ν) * besselk(κ.ν, tmp) / gamma(κ.ν)
# end
# get_params(k::MaternKernel) = [k.ν, k.ρ]
# set_params!(k::MaternKernel, ν, ρ) = begin
#     k.ν = ν
#     k.ρ = ρ
# end

# Works with FSSDrand and FSSDopt for SOME values, and only using `finite-differences` to optimize.
"""
    Matern25Kernel(ρ)

Matern25Kernel is a Matern kernel with ν = 2.5, and is thus 1st order differentiable, which is what we require.

# Notes
Optimization wrt. ρ is potentially unstable, and so some runs might return an error, while others wont.
"""
mutable struct Matern25Kernel <: Kernel
    ρ
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
get_params(k::Matern25Kernel) = [k.ρ]
set_params!(k::Matern25Kernel, ρ) = begin
    k.ρ = ρ
end

### Derivatives

# define the necessary derivatives of kernel
function k_dx(::Kernel, x::AbstractArray, y::AbstractArray) end
function k_dy(::Kernel, x::AbstractArray, y::AbstractArray) end
function k_dxdy(::Kernel, x::AbstractArray, y::AbstractArray) end

# TODO: make derivatives using `SymPy` instead, as we can simply the expressions
# for numerical stability!
# => NOPE. Can't do this for arbitrary dimensions...

"Overloads the derivative operations for `kernel`."
macro make_derivatives(kernel, autodiff::Symbol = :forward)
    # call `eval` to overload methods in scope
    if autodiff == :forward
        eval(quote
             @inline k_dx(k::$kernel, x::AbstractArray, y::AbstractArray) = ForwardDiff.gradient(z -> kernel(k, z, y), x);
             @inline k_dy(k::$kernel, x::AbstractArray, y::AbstractArray) = ForwardDiff.gradient(z -> kernel(k, x, z), y);
             @inline k_dxdy(k::$kernel, x::AbstractArray, y::AbstractArray) = ForwardDiff.jacobian(z -> k_dx(k, x, z), y);

             @inline k_dx(k::$kernel, x::Number, y::Number) = ForwardDiff.derivative(z -> kernel(k, z, y), x);
             @inline k_dy(k::$kernel, x::Number, y::Number) = ForwardDiff.derivative(z -> kernel(k, x, z), y);
             @inline k_dxdy(k::$kernel, x::Number, y::Number) = ForwardDiff.derivative(z -> k_dx(k, x, z), y);
             end)
    elseif autodiff == :backward
        eval(quote
             @inline k_dx(k::$kernel, x::AbstractArray, y::AbstractArray) = ReverseDiff.gradient(z -> kernel(k, z, y), x);
             @inline k_dy(k::$kernel, x::AbstractArray, y::AbstractArray) = ReverseDiff.gradient(z -> kernel(k, x, z), y);
             @inline k_dxdy(k::$kernel, x::AbstractArray, y::AbstractArray) = ReverseDiff.jacobian(z -> k_dx(k, x, z), y);

             # TODO: `ReverseDiff` does not support taking derivatives, as reverse-mode differentiation used useful for N -> 1 case not 1 -> 1 case
             end)
    else
        throw(ValueError("unsupported autodiff method $autodiff"))
    end
end

# does so for the relevant kernels
@make_derivatives GaussianRBF
@make_derivatives ExponentialKernel
# @make_derivatives MaternKernel
@make_derivatives Matern25Kernel
