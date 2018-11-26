using Distributions


"Boostrapping for degnerate U-statistic."
bootstrap_degenerate(samples::AbstractArray, f) = begin
    # bootstrap sample
    res = 0.0

    n = size(samples, 2)

    # sample bootstrap weights
    sample_p = Multinomial(n, n)
    w = (rand(sample_p) / n) .- (1.0 / n)
    
    for i = 1:n
        for j = 1:n
            if i == j
                continue
            end
            r = f(samples[:, i], samples[:, j])
            res += ((w[i] * w[j]) * r)
        end
    end

    return res
end
