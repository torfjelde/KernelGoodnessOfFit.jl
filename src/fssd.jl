using LinearAlgebra
using Statistics

using Distributions
using ForwardDiff, ReverseDiff
using Optim

import HypothesisTests: pvalue, testname, show_params, default_tail, population_param_of_interest


struct FSSDTest <: KernelGoodnessOfFitTest
    kernel::Kernel

    # test locations
    num_test::Int
    initial_V

    # optimization
    optimize::Bool
    num_steps::Int

    # size of test
    α::Float64

    # simulation under H₀
    num_simulate::Int
end

FSSDTest(kernel::Kernel, V::AbstractArray; α = 0.01, num_simulate = 3000) = begin
    FSSDTest(kernel, size(V, 2), V, true, 50, α, num_simulate)
end


struct FSSDResult <: KernelGoodnessOfFitResult
    stat::Float64
    
    p_val # p-value of the test
    α
    result::Symbol # :reject or :accept

    V  # test locations
end

abstract type FSSD <: KernelGoodnessOfFitTest end

# test stuff
population_param_of_interest(t::T where T <: FSSD) = ("Finite-Set Stein Discrepancy (FSSD)", 0, t.stat)
default_tail(t::T where T <: FSSD) = :tail
show_params(io::IO, t::FSSD, ident) = begin
    println(io, ident, "kernel:           ", t.k)
    println(io, ident, "test locations:   ", t.V)
    println(io, ident, "num. simulate H₀: ", t.num_simulate)
end

###############
### FSSDopt ###
###############
mutable struct FSSDopt{K, A} <: FSSD where {K<:Kernel, T<:Real, A<:AbstractArray{T, 2}}
    stat::Float64             # FSSD estimate
    p_val::Float64            # p-value
    k::K                      # kernel
    V::A                      # test locations
    num_simulate::Int         # number of simulations used to approximation null-dist
    train_test_ratio::Float64 # ratio to use as TRAINING data
    # bounds::AbstractArray     # bounds for the the variables to optimize over (e.g. test locations V, kernel params)
end


function FSSDopt(
    x::AbstractArray{T1, 2},
    q::Distribution,
    k::Kernel,
    V::AbstractArray{T2, 2};
    nsim = 3000, train_test_ratio = 0.5, kwargs...
) where {T1, T2}
    n = size(x, 2)
    d, J = size(V)

    split_idx = Integer(floor(n * train_test_ratio))
    train = @view x[:, 1:split_idx]
    test = @view x[:, split_idx + 1:end]

    # update kernel parameters inplace
    kernel_params, V, opt_res = optimize_power(k, V, train, q; kwargs...)

    # update kernel
    k_new = update(k, kernel_params...)

    res = perform(k, V, test, q; num_simulate = nsim)
    FSSDopt(res.stat, res.p_val, k, V, nsim, train_test_ratio)
end

FSSDopt(x::AbstractArray, q, k::Kernel; J::Int = 5, kwargs...) = begin
    # initialization
    p = fit_mle(MvNormal, x)
    V = rand(p, J)
    FSSDopt(x, q, k, V; kwargs...)
end

FSSDopt(x::AbstractArray, q; J::Int = 5, kwargs...) = begin
    # initialization
    p = fit_mle(MvNormal, x)
    V = rand(p, J)

    # default to `GaussianRBF` kernel with variance similar to MLE fit
    k = GaussianRBF(maximum(cov(p)))

    FSSDopt(x, q, k, V; kwargs...)
end

testname(::FSSDopt) = "Finite-Set Stein Discrepancy optimized (FSSD-opt)"
pvalue(t::FSSDopt) = t.p_val

################
### FSSDrand ###
################
mutable struct FSSDrand{K} <: FSSD where {K <: Kernel}
    stat::Float64     # FSSD estimate
    p_val::Float64    # p-value
    k::K              # kernel
    V::AbstractArray  # test locations
    num_simulate::Int # number of simulations used to approximation null-dist
end

# FSSDrand(x::AbstractArray, q, k::Kernel, V::AbstractArray; num_simulate = 3000) = FSSDrand(x, q, k, V, num_simulate)
# FSSDrand(x::AbstractArray, q, k::Kernel; J::Int = 5, num_simulate = 3000) = begin
#     d = size(x, 1)
#     V = zeros(d, J)
#     FSSDrand(x, q, k, V, num_simulate)
# end
# FSSDrand(x::AbstractArray, q; J::Int = 5, num_simulate = 3000) = begin
#     FSSDrand(x, q, GaussianRBF(1.0); J = J, num_simulate = num_simulate)
# end

FSSDrand(x::AbstractArray, q, k::Kernel, V::AbstractArray; nsim = 3000) = begin
    res = perform(k, V, x, q; num_simulate = nsim)
    FSSDrand(res.stat, res.p_val, k, V, nsim)
end

FSSDrand(x::AbstractArray, q, k::Kernel; J::Int = 5, kwargs...) = begin
    # initialization
    p = fit_mle(MvNormal, x)
    V = rand(p, size(x, 2))
    FSSDrand(x, q, k, V; kwargs...)
end

FSSDrand(x::AbstractArray, q; J = 5, kwargs...) = begin
    # initialization
    p = fit_mle(MvNormal, x)
    V = rand(p, J)
    k = GaussianRBF(maximum(cov(p)))

    return FSSDrand(x, q, k, V; kwargs...)
end


testname(::FSSDrand) = "Finite-Set Stein Discrepancy randomized (FSSD-rand)"
pvalue(t::FSSDrand) = t.p_val


# Objective is to maximize FSSD^2 / σ₁ where

# σ₁² = 4 μᵀ Σₚ μ

# Compute unbiased estimate of covariance matrix need to compute the
# asymptotic (normal) distribution under H₁

# Compute ∇ of 
function ξ(k::Kernel, p::Distribution, x::AbstractVector, v::AbstractVector)
    logp_dx = gradlogpdf(p, x)
    kdx = KernelGoodnessOfFit.k_dx(k, x, v)

    return logp_dx * kernel(k, x, v) + kdx
end

function compute_Ξ(k, p, xs, vs)
    # TODO: change to use `size(vs, 2)` and so on
    J = size(vs)[2]  # number of test points
    n = size(xs)[2]  # number of samples
    d = size(xs)[1]  # dimension of the inputs

    [hcat([ξ(k, p, xs[:, i], vs[:, m]) / sqrt(d * J) for i = 1:n]...) for m = 1:J]
end

function compute_Ξ(k, p::D where D <: Distribution{Univariate, Continuous}, xs, vs)
    # TODO: change to use `size(vs, 2)` and so on
    J = size(vs)[2]  # number of test points
    n = size(xs)[2]  # number of samples
    d = size(xs)[1]  # dimension of the inputs

    [hcat([ξ(k, p, xs[1, i], vs[1, m]) / sqrt(d * J) for i = 1:n]...) for m = 1:J]
end


function fssd(Ξ)
    J = size(Ξ)[1]
    d, n = size(Ξ[1])

    tot = 0.0

    # FIXME: Uhmm, this is correct buuuut definitively not O(n); it,'s O(n^2)
    for m = 1:J
        for i = 1:n - 1
            for j = i + 1:n
                tot = tot + (2 / (n * (n - 1))) * dot(Ξ[m][:, i], Ξ[m][:, j])
            end
        end
    end
    
    tot
end

function fssd(k, p, xs, vs)
    J = size(vs)[2]  # number of test points
    n = size(xs)[2]  # number of samples
    d = size(xs)[1]  # dimension of the inputs

    tot = 0.0

    Ξ = compute_Ξ(k, p, xs, vs)
    return fssd(Ξ)
end

function fssd_old(k, p, xs, vs)
    J = size(vs)[2]  # number of test points
    n = size(xs)[2]  # number of samples
    d = size(xs)[1]  # dimension of the inputs

    tot = 0.0
    
    for m = 1:J
        for i = 1:n - 1
            ξᵢ = ξ(k, p, xs[:, i], vs[:, m]) ./ sqrt(d * J)
            
            for j = i + 1:n
                ξⱼ = ξ(k, p, xs[:, j], vs[:, m]) ./ sqrt(d * J)

                # compute Δ(x_i, x_j)ᵐ, where ᵐ denotes the m-th component, i.e.
                # instead of vec(Ξ), we do column by column
                # => d-dim vector which we then dot
                # can think of it as τ(x)_{d * J: (d + 1) * J} • τ(y)_{d * J: (d + 1) * J} <= YOU WHAT MATE?
                # i.e. dotting the columns Ξᵢ • Ξⱼ 

                # TODO: should this be a dot-product?
                # should just correspond to summing over the dimensions, as we want
                tot = tot + (2 / (n * (n - 1))) * dot(ξᵢ, ξⱼ)
            end
            
        end
    end

    # tot, Ξ
    tot
end

function τ_from_Ξ(Ξ)
    vcat(Ξ...)
end

function Σₚ(τ)
    dJ, n = size(τ)
    Σ = zeros((dJ, dJ))

    # take the mean over the rows, i.e. mean of observations
    μ = mean(τ, dims=2)

    for i = 1:n
        Σ = Σ + (1 / n) .* τ[:, i] * transpose(τ[:, i])
    end

    # 𝐄[τ(x)²] - 𝐄[τ(x)]²
    Σ = Σ - μ * transpose(μ)

    μ, Σ
end

σ²_H₁(μ, Σ) = begin
    # should be 1×1 matrix → extract the element
    (4 * transpose(μ) * Σ * μ)[1]
end

function fssd_H₁_opt_factor(k, p, xs, vs; ε = 0.01, β_H₁ = 0.01)
    Ξ = compute_Ξ(k, p, xs, vs)
    s = fssd(Ξ)
    τ = τ_from_Ξ(Ξ)
    μ, Σ = Σₚ(τ)
    σ₁ = σ²_H₁(μ, Σ)

    # asymptotic under H₁ depends O(√n) on (FSSD² / σ₁² + ε)
    # subtract regularization term because we're going to multiply by minus
    # also, we want to stop it from being too SMALL, so regularize inverse of it
    return (s ./ (σ₁ + ε)) .- β_H₁ * (σ₁ + (σ₁ + 1e-6)^(-1))
end

### Defaults
pack(k::Kernel, vs::AbstractArray) = vcat(params(k)..., vs...)
unpack(k::Kernel, θ::AbstractVector, d::Integer, J::Integer) = begin
    k_dim = length(params(k))
    return θ[1:k_dim], reshape(θ[k_dim + 1: end], d, J)
end

# only test-locations
pack(vs::AbstractArray) = vcat(vs...)
unpack(θ::AbstractVector, d::Integer, J::Integer) = reshape(θ, d, J)


### Gaussian kernel optimization: {σₖ, V}
pack(k::GaussianRBF, vs::AbstractArray) = vcat(log(k.gamma), vs...)
unpack(k::GaussianRBF, θ::AbstractVector, d::Integer, J::Integer) = (exp(θ[1]), ), reshape(θ[2:end], d, J)


### Exponential kernel pack / unpack
pack(k::ExponentialKernel, vs::AbstractArray) = vcat(vs...)
unpack(k::ExponentialKernel, θ::AbstractVector, d::Integer, J::Integer) = (), reshape(θ, d, J)

# ### Matern Kernel: do the log-exp transform to enforce positive
# pack(k::MaternKernel, vs::AbstractArray) = vcat(log(k.ν), log(k.ρ), vs...)
# unpack(k::MaternKernel, θ::AbstractArray, d::Integer, J::Integer) = exp.(θ[1:2]), reshape(θ[3:end], d, J)
pack(k::Matern25Kernel, vs::AbstractArray) = vcat(log(k.ρ), vs...)
unpack(k::Matern25Kernel, θ::AbstractVector, d::Integer, J::Integer) = exp.(θ[1:1]), reshape(θ[2:end], d, J)

### InverseMultiQuadratic (IMQ): c > 0, b < 0
pack(k::InverseMultiQuadratic, vs::AbstractArray) = vcat(log(k.c), log(- k.b), vs...)
unpack(k::InverseMultiQuadratic, θ::AbstractVector, d::Integer, J::Integer) = (exp(θ[1]), - exp(θ[2])), reshape(θ[3:end], d, J)


function optimize_power(k::K, vs, xs, p; method::Symbol = :lbfgs, diff::Symbol = :forward, num_steps = 10, step_size = 0.1, β_σ = 0.0, β_V = 0.0, β_H₁ = 0.0, ε = 0.01, lower::AbstractArray = [], upper::AbstractArray = [], test_locations_only = false) where K <: Kernel
    d, J = size(vs)

    # pack and combine
    if test_locations_only
        θ₀ = pack(vs)
    else
        θ₀ = pack(k, vs)
    end

    # define objective (don't call unwrap_θ for that perf yo)
    function f(θ)
        if test_locations_only
            V = unpack(θ, d, J)
            return - fssd_H₁_opt_factor(k, p, xs, V; ε = ε, β_H₁ = β_H₁)
        else
            kernel_params, V = unpack(k, θ, d, J)
            ker = update(k, kernel_params...)

            # add regularization to the parameter
            # TODO: currently using matrix norm for `V` => should we use a vector for β_V and use vector nor?
            if β_σ > 0.0 || β_V > 0.0
                # TODO: remove this; don't wnat to only be able to use one kernel
                return - fssd_H₁_opt_factor(ker, p, xs, V; ε = ε, β_H₁ = β_H₁) + β_σ ./ (σ^2 + 1e-6) + β_V * norm(V)
            else
                return - fssd_H₁_opt_factor(ker, p, xs, V; ε = ε, β_H₁ = β_H₁)
            end
        end
    end

    # define gradient
    if diff == :forward
        # ∇f! = (F, θ) -> ForwardDiff.gradient!(F, f, θ)
        ∇f! = nothing
    elseif diff == :backward
        ∇f! = (F, ζ) -> ReverseDiff.gradient!(F, f, ζ)
    elseif diff == :difference
        ∇f! = nothing
    else
        throw(ArgumentError("diff = $diff not not supported"))
    end

    # optimize
    if diff == :forward
        if (length(lower) > 0) && (length(upper) > 0)
            @info "optimizing using bounds"
            opt_res = optimize(f, lower, upper, clamp.(θ₀, lower, upper), Fminbox(LBFGS()), autodiff = diff)
        else 
            opt_res = optimize(f, θ₀, LBFGS(), autodiff = diff)
        end
    elseif diff == :backward
        if (length(lower) > 0) && (length(upper) > 0)
            @info "optimizing using bounds"
            opt_res = optimize(f, ∇f!, lower, upper, clamp.(θ₀, lower, upper), Fminbox(LBFGS()))
        else 
            opt_res = optimize(f, ∇f!, θ₀, LBFGS())
        end
    else
        if (length(lower) > 0) && (length(upper) > 0)
            @info "optimizing using bounds"
            opt_res = optimize(f, lower, upper, clamp.(θ₀, lower, upper), Fminbox(LBFGS()))
        else
            opt_res = optimize(f, θ₀, LBFGS())
        end
    end

    θ = opt_res.minimizer

    if test_locations_only
        vs_ = unpack(θ, d, J)
        return (), vs_, opt_res
    else
        kernel_params_, vs_ = unpack(k, θ, d, J)
        return kernel_params_, vs_, opt_res
    end
end


# 1. Choose kernel and initialize kernel parameters + test locations
# 2. [Optional] Optimize over kernel parameters + test locations
# 3. Perform test

function perform(t::FSSD)
    perform(t.k, t.V, t.x, t.q; num_simulate = t.num_simulate)
end

function perform(k::Kernel, vs, xs, p; α = 0.05, num_simulate = 1000)
    d, n = size(xs)

    # compute
    Ξ = compute_Ξ(k, p, xs, vs)
    test_stat = n * fssd(Ξ)

    # compute asymptotics under H₀
    μ, Σ̂ = Σₚ(τ_from_Ξ(Ξ))

    # HACK: this sometimes end up with complex-valued eigenvalues (imaginary party < e^{-18}) → conert to real
    ω = real.(eigvals(Σ̂))

    # simulate under H₀
    draws = randn(length(ω), num_simulate)
    sim_stat = transpose(ω) * (draws.^2 .- 1)

    # estimate P(FSSD² > \hat{FSSD²}), i.e. p-value
    # FIXME: sim_stat > test_stat 100% of the time in the case where test_stat == 0.0
    # Should this be ≥, since the case where test_stat == 0.0 and all sim_stat == 0.0,
    # then clearly H₀ is true, but using > we will have p-val of 0.0
    p_val = mean(sim_stat .> test_stat)

    # P(FSSD² > \hat{FSSD²}) ≤ α ⟺ P(FSSD² ≤ \hat{FSSD²}) ≥ 1 - α
    # ⟹ reject since that means that \hat{FSSD²}
    # lies outside the (1 - α)-quantile ⟹ "unlikely" for H₀ to be true
    if p_val ≤ α
        res = :reject
    else
        res = :accept
    end

    FSSDResult(test_stat, p_val, α, res, vs)
end


function perform(t::FSSDTest, q, xs)
    return perform(t.kernel, t.initial_V, x, q; α = t.α, num_simulate = t.num_simulate)
end
