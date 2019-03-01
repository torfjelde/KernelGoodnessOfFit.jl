# General
abstract type Kernel end

function kernel(k::Kernel, x::AbstractVector, y::AbstractVector) end

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
             k_dx(k::$kernel, x::AbstractArray, y::AbstractArray) = ForwardDiff.gradient(z -> kernel(k, z, y), x);
             k_dy(k::$kernel, x::AbstractArray, y::AbstractArray) = ForwardDiff.gradient(z -> kernel(k, x, z), y);
             k_dxdy(k::$kernel, x::AbstractArray, y::AbstractArray) = ForwardDiff.jacobian(z -> k_dx(k, x, z), y);
             end)
    elseif autodiff == :backward
        eval(quote
             k_dx(k::$kernel, x::AbstractArray, y::AbstractArray) = ReverseDiff.gradient(z -> kernel(k, z, y), x);
             k_dy(k::$kernel, x::AbstractArray, y::AbstractArray) = ReverseDiff.gradient(z -> kernel(k, x, z), y);
             k_dxdy(k::$kernel, x::AbstractArray, y::AbstractArray) = ReverseDiff.jacobian(z -> k_dx(k, x, z), y);
             end)
    else
        throw(ValueError("unsupported autodiff method $autodiff"))
    end
end

# does so for the relevant kernels
@make_derivatives GaussianRBF
