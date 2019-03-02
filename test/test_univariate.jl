using Distributions
using Test
using GoodnessOfFit

@testset "Univeriate Normal" begin
    xs = randn(1, 100);
    q = Normal(1.0, 1.0)

    t1 = FSSDrand(xs, q)

    println(t1)

    @test pvalue(t1) ≥ 0.0
end
