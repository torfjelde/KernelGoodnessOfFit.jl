using LinearAlgebra
using Statistics

using Distributions
using ForwardDiff, ReverseDiff
using Optim

import HypothesisTests: pvalue, testname, show_params, default_tail, population_param_of_interest


struct FSSDTest <: GoodnessOfFitTest
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


struct FSSDResult <: GoodnessOfFitResult
    stat::Float64
    
    p_val # p-value of the test
    α
    result::Symbol # :reject or :accept

    V  # test locations
end

abstract type FSSD <: GoodnessOfFitTest end
set_locs!(t::FSSD, V::AbstractArray) = begin
    t.V = V
end

# test stuff
population_param_of_interest(t::T where T <: FSSD) = ("Finite-Set Stein Discrepancy (FSSD)", 0, "?")
default_tail(t::T where T <: FSSD) = :tail
show_params(io::IO, t::FSSD, ident) = begin
    println(io, ident, "kernel:           ", t.k)
    println(io, ident, "test locations:   ", t.V)
    println(io, ident, "num. simulate H₀: ", t.num_simulate)
    println(io, ident, "q:                ", t.q)
end

###############
### FSSDopt ###
###############
mutable struct FSSDopt{T} <: FSSD where T <: Kernel
    x::AbstractArray # data
    q                # a "distribution"; only requires `gradlogpdf(d, x)` to be defined
    k::T
    V::AbstractArray # have some default value
    num_simulate::Int
    train_test_ratio::Float64
end

FSSDopt(x::AbstractArray, q, k::Kernel, V::AbstractArray; num_simulate = 3000, train_test_ratio = 0.5) = FSSDopt(x, q, k, V, num_simulate, train_test_ratio)
FSSDopt(x::AbstractArray, q, k::Kernel; J::Int = 5, num_simulate = 3000, train_test_ratio = 0.5) = begin
    d = size(x, 1)
    V = zeros(d, J)
    FSSDopt(x, q, k, V, num_simulate, train_test_ratio)
end
FSSDopt(x::AbstractArray, q; J::Int = 5, num_simulate = 3000, train_test_ratio = 0.5) = begin
    FSSDopt(x, q, GaussianRBF(1.0); J = J, num_simulate = num_simulate, train_test_ratio = train_test_ratio)
end

testname(::FSSDopt) = "Finite-Set Stein Discrepancy optimized (FSSD-opt)"

################
### FSSDrand ###
################
mutable struct FSSDrand{T} <: FSSD where T  <: Kernel
    x::AbstractArray
    q
    k::T
    V::AbstractArray # have some default value
    num_simulate::Int
end

FSSDrand(x::AbstractArray, q, k::Kernel, V::AbstractArray; num_simulate = 3000) = FSSDrand(x, q, k, V, num_simulate)
FSSDrand(x::AbstractArray, q, k::Kernel; J::Int = 5, num_simulate = 3000) = begin
    d = size(x, 1)
    V = zeros(d, J)
    FSSDrand(x, q, k, V, num_simulate)
end
FSSDrand(x::AbstractArray, q; J::Int = 5, num_simulate = 3000) = begin
    FSSDrand(x, q, GaussianRBF(1.0); J = J, num_simulate = num_simulate)
end


testname(::FSSDrand) = "Finite-Set Stein Discrepancy randomized (FSSD-rand)"

# initialization of parameters
initialize!(t::FSSD) = begin
    # initialize test locations
    p = fit_mle(MvNormal, t.x)
    t.V = rand(p, size(t.V, 2))
end

initialize!(t::FSSDrand{GaussianRBF}) = begin
    # initialize test locations
    p = fit_mle(MvNormal, t.x)
    t.V = rand(p, size(t.V, 2))

    # set Gaussian kernel bandwith to maximum variance
    t.k = GaussianRBF(maximum(cov(p)))
end

pvalue(t::FSSDrand) = begin
    initialize!(t)
    
    # perform test
    res = perform(t)
    res.p_val
end

pvalue(t::FSSDopt) = begin
    initialize!(t)

    # optimize params
    optimize_power!(t)

    # perform test
    res = perform(t)

    res.p_val
end

optimize_power!(t::FSSDopt{T} where T <: Kernel) = begin
    n = size(t.x, 2)
    d, J = size(t.V)

    split_idx = Integer(floor(n * t.train_test_ratio))
    train = @view t.x[:, 1:split_idx]

    # update kernel parameters inplace
    kernel_params, V, opt_res = optimize_power(t.k, t.V, train, t.q)

    # update kernel
    set_params!(t.k, kernel_params...)

    # update test locations
    set_locs!(t, V)
end

# Objective is to maximize FSSD^2 / σ₁ where

# σ₁² = 4 μᵀ Σₚ μ

# Compute unbiased estimate of covariance matrix need to compute the
# asymptotic (normal) distribution under H₁

# Compute ∇ of 
function ξ(k, p, x, v)
    # CHECKED
    
    # TODO: maybe switch to computing the Jacobian
    # logp_dx = ForwardDiff.gradient(z -> logpdf(p, z), x)
    logp_dx = gradlogpdf(p, x)
    kdx = GoodnessOfFit.k_dx(k, x, v)

    return logp_dx * kernel(k, x, v) + kdx
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

function compute_Ξ(k, p, xs, vs)
    # TODO: change to use `size(vs, 2)` and so on
    J = size(vs)[2]  # number of test points
    n = size(xs)[2]  # number of samples
    d = size(xs)[1]  # dimension of the inputs

    [hcat([ξ(k, p, xs[:, i], vs[:, m]) / sqrt(d * J) for i = 1:n]...) for m = 1:J]
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

function fssd_H₁_opt_factor(k, p, xs, vs; ε = 0.01, β_H₁ = 0.0)
    Ξ = compute_Ξ(k, p, xs, vs)
    s = fssd(Ξ)
    τ = τ_from_Ξ(Ξ)
    μ, Σ = Σₚ(τ)
    σ₁ = σ²_H₁(μ, Σ)

    # asymptotic under H₁ depends O(√n) on (FSSD² / σ₁² + ε)
    # subtract regularization term because we're going to multiply by minus
    # also, we want to stop it from being too SMALL, so regularize inverse of it
    return (s ./ (σ₁ + ε)) .- (β_H₁ / σ₁)
end

### Gaussian kernel optimization
function pack(k::GaussianRBF, vs::AbstractArray)
    vcat(log(k.gamma), vs...)
end

function unpack(k::GaussianRBF, ζ::AbstractVector, d::Integer, J::Integer)
    # σₖ, V
    return (exp(ζ[end]), ), reshape(ζ[1:end - 1], d, J)
end


### Exponential kernel pack / unpack
pack(k::ExponentialKernel, vs::AbstractArray) = vcat(vs...)
unpack(k::ExponentialKernel, ζ::AbstractVector, d::Integer, J::Integer) = (), reshape(ζ, d, J)

# ### Matern Kernel: do the log-exp transform to enforce positive
# pack(k::MaternKernel, vs::AbstractArray) = vcat(log(k.ν), log(k.ρ), vs...)
# unpack(k::MaternKernel, ζ::AbstractArray, d::Integer, J::Integer) = exp.(ζ[1:2]), reshape(ζ[3:end], d, J)

function optimize_power(k::K, vs, xs, p; method::Symbol = :lbfgs, diff::Symbol = :backward, num_steps = 10, step_size = 0.1, β_σ = 0.0, β_V = 0.0, β_H₁ = 0.0, ε = 0.01) where K <: Kernel
    d, J = size(vs)

    # define objective (don't call unwrap_ζ for that perf yo)
    f(ζ) = begin
        kernel_params, V = unpack(k, ζ, d, J)
        ker = isempty(kernel_params) ? K() : K(kernel_params...)

        # add regularization to the parameter
        # TODO: currently using matrix norm for `V` => should we use a vector for β_V and use vector nor?
        if β_σ > 0.0 || β_V > 0.0
            - fssd_H₁_opt_factor(ker, p, xs, V; ε = ε, β_H₁ = β_H₁) + β_σ ./ (σ^2 + 1e-6) + β_V * norm(V)
        else
            - fssd_H₁_opt_factor(ker, p, xs, V; ε = ε, β_H₁ = β_H₁)
        end
    end

    # define gradient
    if diff == :forward
        ∇f! = (F, ζ) -> ForwardDiff.gradient!(F, f, ζ)
    elseif diff == :backward
        ∇f! = (F, ζ) -> ReverseDiff.gradient!(F, f, ζ)
    elseif diff == :difference
        ∇f! = nothing
    else
        throw(ArgumentError("diff = $diff not not supported"))
    end

    # pad and combine
    ζ₀ = pack(k, vs)

    if method == :lbfgs
        # optimize
        if diff != :difference
            opt_res = optimize(f, ∇f!, ζ₀, LBFGS())
        else
            opt_res = optimize(f, ζ₀, LBFGS())
        end

        ζ = opt_res.minimizer
        kernel_params_, vs_ = unpack(k, ζ, d, J)
        
    elseif method == :sgd
        ζ = ζ₀

        # setup container for gradient
        F = zeros(size(ζ))

        # step
        opt_res = @elapsed for i = 1:num_steps
            # update
            ζ = ζ - step_size * ∇f!(F, ζ)
        end

        kernel_params_, vs_ = unpack(k, ζ, d, J)
    else
        error("$method not recogized as a supported method")
    end

    kernel_params_, vs_, opt_res
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
