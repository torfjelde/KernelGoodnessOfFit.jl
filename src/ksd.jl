using Distributed, SharedArrays, LinearAlgebra

using ForwardDiff
using Distributions

struct KSDTest <: GoodnessOfFitTest
    kernel::Kernel
    m::Integer
    α::Float64
end

struct KSDResult <: GoodnessOfFitResult
    stat::Float64

    "p-value of test"
    p_val
    "size of test"
    α # size of test
    "`:reject` or `:accept` to indicate rejection or acceptance of H₀"
    result::Symbol # :reject or :accept
end

u_p(k::Kernel, x::AbstractVector, y::AbstractVector, s_x::AbstractVector, s_y::AbstractVector) = begin
    kernel(k, x, y) * dot(s_x, s_y) +
        dot(s_x, k_dy(k, x, y)) +
        dot(k_dx(k, x, y), s_y) +
        tr(k_dxdy(k, x, y))
end

u_p(k::Kernel, p::Distribution, x::AbstractVector, y::AbstractVector) = begin
    s_x = gradlogpdf(p, x)
    s_y = gradlogpdf(p, y)
    
    u_p(k, x, y, s_x, s_y)
end

# compute unbiased estimate
Sᵤ_estimate(samples::AbstractArray, f) = begin
    n = size(samples, 2)
    res = 0.0
    
    for i = 1:n
        for j = 1:n
            if i == j
                continue
            end
            r = f(samples[:, i], samples[:, j])
            res += (r / (n * (n - 1)))
        end
    end

    res
end

"Performs a goodness-of-fit test based on the Kernelized Stein Discrepancy (KSD)."
perform(t::KSDTest, q, xs) = begin
    bootstrapped = SharedArray{Float64}(t.m);

    f = (x, y) -> u_p(t.kernel, q, x, y)

    @sync @distributed for b = 1:t.m
        bootstrapped[b] = bootstrap_degenerate(xs, f)

        if (b % 10000) == 0
            println("[$b / $m] Done!")
        end
    end

    # Unbiased estimate
    # println("Computing unbiased estimate")
    Ŝᵤ = Sᵤ_estimate(xs, f)

    # compute average times this occurs
    pval = reduce(+, bootstrapped .> Ŝᵤ) / t.m

    res = :accept
    if pval < t.α
        res = :reject;
    end

    # TODO: return some result containing information
    KSDResult(Ŝᵤ, pval, t.α, res)
end
