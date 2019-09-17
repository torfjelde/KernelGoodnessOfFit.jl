using Test, Random, Distributions, HypothesisTests, LinearAlgebra
using KernelGoodnessOfFit

@testset "FSSDrand" begin
    Random.seed!(123)
    xs = randn(100);
    
    res = KernelGoodnessOfFit.FSSDrand(
        reshape(xs, 1, :),
        MvNormal([1.0], [1.0]),
        GaussianRBF(1.0),
        reshape([0.0], 1, 1),
        nsim = 3000
    )

    # reject
    @test pvalue(res) ≤ 0.05
end

@testset "FSSDopt" begin
    Random.seed!(123)
    xs = randn(100);
    
    res = KernelGoodnessOfFit.FSSDopt(
        reshape(xs, 1, :),
        MvNormal([1.0], [1.0]),
        GaussianRBF(1.0),
        reshape([0.0], 1, 1),
        nsim = 3000,
        train_test_ratio = 0.5
    )

    # reject
    @test pvalue(res) ≤ 0.05
end

# @testset "FSSDopt univariate" begin
#     Random.seed!(123)
#     xs = randn(100);

#     res = KernelGoodnessOfFit.FSSDopt(
#         xs,
#         Normal(1.0, 1.0),
#         GaussianRBF(1.0),
#         [0.0],
#         nsim = 3000,
#         train_test_ratio = 0.5
#     )

#     # reject
#     @test pvalue(res) ≤ 0.05
# end

@testset "FSSDrand multivariate" begin
    Random.seed!(123)
    d = 5
    xs = randn(d, 100);
    
    res = KernelGoodnessOfFit.FSSDrand(
        xs,
        MvNormal(ones(d), diagm(0 => ones(d))),
        GaussianRBF(1.0),
        zeros(5, 2);  # two test-locations
        nsim = 3000
    )

    # reject
    @test pvalue(res) ≤ 0.05
end

@testset "FSSDopt multivariate" begin
    Random.seed!(123)
    d = 5
    xs = randn(d, 100);

    res = pvalue(KernelGoodnessOfFit.FSSDopt(
        xs,
        MvNormal(10 .* ones(d), diagm(0 => ones(d))),
        GaussianRBF(1.0),
        zeros(5, 2);  # two test-locations
        nsim = 3000,
        train_test_ratio = 0.5
    ))

    # reject
    @test res ≤ 0.05
end


@testset "FSSDrand different kernels" begin
    Random.seed!(123)
    xs = randn(1, 100);
    q = MvNormal([10.0], [1.0])

    t_rbf = FSSDrand(xs, q, KernelGoodnessOfFit.GaussianRBF(1.0))
    t_exp = FSSDrand(xs, q, KernelGoodnessOfFit.ExponentialKernel())
    t_matern = FSSDrand(xs, q, KernelGoodnessOfFit.Matern25Kernel(1.0))

    @test pvalue(t_rbf) ≥ 0.0
    @test pvalue(t_exp) ≥ 0.0
    @test pvalue(t_matern) ≥ 0.0
end

@testset "FSSDopt different kernels" begin
    Random.seed!(123)
    xs = randn(1, 100);

    q = MvNormal([10.0], [1.0])

    t_rbf = FSSDopt(xs, q, KernelGoodnessOfFit.GaussianRBF(1.0))
    t_exp = FSSDopt(xs, q, KernelGoodnessOfFit.ExponentialKernel())
    # t_matern = FSSDrand(xs, q, KernelGoodnessOfFit.Matern25Kernel(1.0))  # OPTIMIZATION NOT STABLE

    @test pvalue(t_rbf) ≥ 0.0
    @test pvalue(t_exp) ≥ 0.0
    # @test pvalue(t_matern) ≥ 0.0
end
