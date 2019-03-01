using Test, Distributions, HypothesisTests, LinearAlgebra
using GoodnessOfFit

xs = randn(1, 100);

@testset "Initialization FSSDrand" begin
    q = MvNormal([1.0], [1.0])
    t1 = FSSDrand(xs, q)  # default construction
    t2 = FSSDrand(xs, q, GaussianRBF(1.0))

    GoodnessOfFit.initialize!(t1)
    GoodnessOfFit.initialize!(t2)

    @test t1.k.gamma == t2.k.gamma
end

@testset "Initialization FSSDopt" begin
    q = MvNormal([1.0], [1.0])
    t1 = FSSDopt(xs, q)  # default construction
    t2 = FSSDopt(xs, q, GaussianRBF(1.0))

    GoodnessOfFit.initialize!(t1)
    GoodnessOfFit.initialize!(t2)

    @test t1.k.gamma == t2.k.gamma
end

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
        3000,
        0.5
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
        3000,
        0.5
    )
    
    println(res)

    # reject
    @test pvalue(res) ≤ 0.05
end
