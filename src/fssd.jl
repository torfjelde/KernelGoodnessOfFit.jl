using LinearAlgebra
using Statistics

using Distributions
using ForwardDiff
using Optim


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

function fssd_H₁_opt_factor(k, p, xs, vs)
    Ξ = compute_Ξ(k, p, xs, vs)
    s = fssd(Ξ)
    τ = τ_from_Ξ(Ξ)
    μ, Σ = Σₚ(τ)

    # asymptotic under H₁ depends O(√n) on (FSSD² / σ₁² + ε)
    return s ./ (σ²_H₁(μ, Σ) + 0.01)
end


# 1. Choose kernel and initialize kernel parameters + test locations
# 2. [Optional] Optimize over kernel parameters + test locations
# 3. Perform test

function perform(k::Kernel, vs, xs, p; α = 0.05, num_simulate = 1000)
    d, n = size(xs)

    # compute
    Ξ = compute_Ξ(k, p, xs, vs)
    test_stat = n * fssd(Ξ)

    # compute asymptotics under H₀
    μ, Σ̂ = Σₚ(τ_from_Ξ(Ξ))
    ω = eigvals(Σ̂)

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


### Gaussian kernel optimization
function wrap_ζ(k::GaussianRBF, vs)
    d, J = size(vs)
    
    # pad matrix
    σₖ_arr = zeros(d, 1)
    σₖ_arr[1] = k.gamma

    # combine
    hcat(vs, σₖ_arr)
end

function unwrap_ζ(k::GaussianRBF, ζ)
    # σₖ, V
    return first(ζ[:, end]), ζ[:, 1:end - 1]
end

function optimize_power(k::GaussianRBF, vs, xs, p; method::Symbol = :lbfgs, num_steps = 10, γ = 0.1, β_σ = 0.0, β_V = 0.0)
    d, J = size(vs)

    # define objective (don't call unwrap_ζ for that perf yo)
    f(ζ) = begin
        # TODO: add regularization?
        σ = first(ζ[:, end])
        V = ζ[:, 1:end - 1]

        # add regularization to the parameter
        # TODO: currently using matrix norm for `V` => should we use a vector for β_V and use vector norm?
        if β_σ > 0.0 || β_V > 0.0
            - fssd_H₁_opt_factor(GaussianRBF(σ), p, xs, V) + β_σ ./ (σ^2 + 1e-6) + β_V * norm(V)
        else
            - fssd_H₁_opt_factor(GaussianRBF(σ), p, xs, V)
        end
    end

    # define gradient
    ∇f! = (F, ζ) -> ForwardDiff.gradient!(F, f, ζ)

    # pad and combine
    ζ₀ = wrap_ζ(k, vs)

    if method == :lbfgs
        # optimize
        opt_res = optimize(f, ∇f!, ζ₀, LBFGS())

        ζ = opt_res.minimizer
        σ_, vs_ = unwrap_ζ(k, ζ)
        
    elseif method == :sgd
        ζ = ζ₀

        # setup container for gradient
        F = zeros(size(ζ))

        # step
        opt_res = @elapsed for i = 1:num_steps
            # update
            ζ = ζ - γ * ∇f!(F, ζ)
        end

        σ_, vs_ = unwrap_ζ(k, ζ)
    else
        error("$method not recogized as a supported method")
    end

    σ_, vs_, opt_res
end


function perform(t::FSSDTest, q, xs)
    return perform(t.kernel, t.initial_V, x, q; α = t.α, num_simulate = t.num_simulate)
end
