using Test, Distributions, HypothesisTests, LinearAlgebra
using GoodnessOfFit

@testset "FSSDrand" begin
    xs = randn(100);
    
    res = GoodnessOfFit.FSSDrand(
        reshape(xs, 1, :),
        MvNormal([1.0], [1.0]),
        GaussianRBF(1.0),
        reshape([0.0], 1, 1),
        3000
    )

    println(res)

    # reject
    @test pvalue(res) ≤ 0.05
end

@testset "FSSDopt" begin
    xs = randn(100);
    
    res = GoodnessOfFit.FSSDopt(
        reshape(xs, 1, :),
        MvNormal([1.0], [1.0]),
        GaussianRBF(1.0),
        reshape([0.0], 1, 1),
        3000
    )

    println(res)

    # reject
    @test pvalue(res) ≤ 0.05
end

@testset "FSSDrand multivariate" begin
    d = 5
    xs = randn(d, 100);
    
    res = GoodnessOfFit.FSSDrand(
        xs,
        MvNormal(ones(d), diagm(0 => ones(d))),
        GaussianRBF(1.0),
        zeros(5, 2),  # two test-locations
        3000
    )

    println(res)

    # reject
    @test pvalue(res) ≤ 0.05
end

@testset "FSSDopt multivariate" begin
    d = 5
    xs = randn(d, 100);
    
    res = GoodnessOfFit.FSSDopt(
        xs,
        MvNormal(ones(d), diagm(0 => ones(d))),
        GaussianRBF(1.0),
        zeros(5, 2),  # two test-locations
        3000
    )
    
    println(res)

    # reject
    @test pvalue(res) ≤ 0.05
end
