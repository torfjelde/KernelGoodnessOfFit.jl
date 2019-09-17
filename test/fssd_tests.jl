using Test, Distributions, ForwardDiff, ReverseDiff, KernelGoodnessOfFit

# setup the problem
rbf_ = GaussianRBF(1.0)

μ = 0.0
σ² = 1.0
p_ = MultivariateNormal([μ, 0], [σ² 0; 0 σ²])  # true
q_ = MultivariateNormal([μ, 5.0], [σ² 0; 0 σ²])  # false

vs_ = [0.0 ; 0.0]
vs_ = reshape(vs_, 2, 1)

n_ = 50
xs_ = rand(p_, n_)

# new vs old: verify that new implementation is correct
@testset "fssd vs fssd_old: eq" begin
    @test KernelGoodnessOfFit.fssd(rbf_, q_, xs_, vs_) == KernelGoodnessOfFit.fssd_old(rbf_, q_, xs_, vs_)
    @test KernelGoodnessOfFit.fssd(rbf_, p_, xs_, vs_) == KernelGoodnessOfFit.fssd_old(rbf_, p_, xs_, vs_)
end

# ensure new method is faster :)
@testset "fssd vs fssd_old: perf" begin
    t1 = @elapsed KernelGoodnessOfFit.fssd(rbf_, q_, xs_, vs_) 
    t2 = @elapsed KernelGoodnessOfFit.fssd_old(rbf_, q_, xs_, vs_)

    @test t1 < t2
end

# sizes: testing of sizes
@testset "shapes" begin
    # setup for tests
    Ξ_ = KernelGoodnessOfFit.compute_Ξ(rbf_, q_, xs_, vs_)

    J, = size(Ξ_)
    d, n = size(Ξ_[1])
    
    @test (d, J) == size(vs_)
    @test (d, n) == size(xs_)

    τ = KernelGoodnessOfFit.τ_from_Ξ(Ξ_)
    @test size(τ) == (d * J, n)
    
    μ, Σ = KernelGoodnessOfFit.Σₚ(τ)
    @test size(μ) == (d * J, 1)
    @test size(Σ) == (d * J, d * J)
    
    σ = KernelGoodnessOfFit.σ²_H₁(μ, Σ)
    @test size(σ) == ()
end

# gradients: verify gradients
@testset "gradient shapes" begin
    Δv = ForwardDiff.gradient(v -> KernelGoodnessOfFit.fssd_H₁_opt_factor(rbf_, q_, xs_, v), vs_)
    @test size(Δv) == size(vs_)

    Δv = ReverseDiff.gradient(v -> KernelGoodnessOfFit.fssd_H₁_opt_factor(rbf_, q_, xs_, v), vs_)
    @test size(Δv) == size(vs_)
    
    # fails for some reason; but does not fail when I actually run the test...interesting, eh?
    # Δσₖ = ForwardDiff.derivative(σₖ -> fssd_H₁(GaussianRBF(σₖ), q_, xs_, vs_), 1.0)
end


# FIXME: find exact values for smaller example (just reduced `n` from 400 to 50 so values are invalid)
# # values: testing specific values
# @testset "specific values" begin
#     V = reshape([0; 0], (2, 1))

#     X = [1.0 2.0 3.0; 2.0 3.0 4.0]

#     Ξ_1_true = [
#         -0.116086 -0.00425237 -1.58109e-5; 
#         0.0580429 -0.00106309 -7.90543e-6
#     ]
#     Ξ_ = KernelGoodnessOfFit.compute_Ξ(rbf_, q_, X, V)

#     @test Ξ_[1] ≈ Ξ_1_true atol=0.01

#     fssd_true = 0.0004333865124975432

#     @test KernelGoodnessOfFit.fssd_from_Ξ(Ξ_) * size(X)[2] ≈ fssd_true

#     σ₁² = 2.8732542480282987e-5

#     τ_ = KernelGoodnessOfFit.τ_from_Ξ(Ξ_)
#     μ_, Σ_ = KernelGoodnessOfFit.Σₚ(τ_)

#     @test KernelGoodnessOfFit.σ²_H₁(μ_, Σ_) ≈ σ₁²
# end

# optimizing the power
@testset "optimize_power" begin
    # because of compilation and stuff, running them in sequence is heavily biased towards
    # the ones being run last
    KernelGoodnessOfFit.optimize_power(rbf_, vs_, xs_, q_; diff = :difference)
    KernelGoodnessOfFit.optimize_power(rbf_, vs_, xs_, q_; diff = :backward)
    KernelGoodnessOfFit.optimize_power(rbf_, vs_, xs_, q_; diff = :forward)

    μs = 1:10

    t_difference = mean([@elapsed KernelGoodnessOfFit.optimize_power(rbf_, vs_, xs_, MultivariateNormal([μ, 5.0], [σ² 0; 0 σ²]); diff = :difference) for μ ∈ μs])
    t_forward = mean([@elapsed KernelGoodnessOfFit.optimize_power(rbf_, vs_, xs_, MultivariateNormal([μ, 5.0], [σ² 0; 0 σ²]); diff = :forward) for μ ∈ μs])
    t_backward = mean([@elapsed KernelGoodnessOfFit.optimize_power(rbf_, vs_, xs_, MultivariateNormal([μ, 5.0], [σ² 0; 0 σ²]); diff = :backward) for μ ∈ μs])

    @test t_backward ≥ 0.0
    @test t_forward ≥ 0.0
    @test t_forward < t_difference
end

# NEW STUFF
@testset "FSSD: ξ" begin
    using KernelGoodnessOfFit: ξ

    k = GaussianRBF(1.0)

    # univariate
    d = Normal(1.0, 1.0)
    x = rand(d)
    v = randn()

    @test size(ξ(k, d, x, v)) == ()

    # multivariate
    d = MvNormal(ones(2), ones(2))
    x = rand(d)
    v = randn(2)

    @test size(ξ(k, d, x, v)) == size(x)
end

@testset "FSSD: Ξ" begin
    using KernelGoodnessOfFit: ξ, Ξ

    k = GaussianRBF(1.0)

    J = 3
    n = 5

    # Univariate
    p = Normal(1.0, 1.0)
    xs = rand(p, n)
    x = xs[1]

    vs = randn(J)
    v = vs[1]

    @test Ξ(k, p, x, vs) == [ξ(k, p, x, vs[i]) / sqrt(J) for i = 1:J]
    @test size(ξ(k, p, x, v)) == size(x)
    @test size(Ξ(k, p, x, vs)) == (J, )
    @test size(Ξ(k, p, xs, v, J)) == (n, )
    @test size(Ξ(k, p, xs, vs)) == (n, J)

    # test objective-maker for `FSSDopt` on univariate distribution
    using KernelGoodnessOfFit: make_objective, pack, unpack
    obj_f = make_objective(k, p, xs, vs, Val{false}())
    obj_t = make_objective(k, p, xs, vs, Val{true}())
    @test obj_f(pack(k, vs)) == obj_t(vs)

    # Multivariate
    d = 2
    p = MvNormal(ones(d), ones(d))
    xs = rand(p, n)
    x = xs[:, 1]

    vs = randn(2, J)
    v = vs[:, 1]

    @test Ξ(k, p, x, vs, J) == hcat([ξ(k, p, x, vs[:, i]) / sqrt(d * J) for i = 1:J]...)
    @test size(ξ(k, p, x, v)) == size(x)
    @test size(Ξ(k, p, x, vs, J)) == (d, J)
    @test size(Ξ(k, p, xs, v, J)) == (d, n)
    @test size(Ξ(k, p, xs, vs, J)) == (d, n, J)

    # test objective-maker for `FSSDopt` on multivariate distribution
    obj_f = make_objective(k, p, xs, vs, Val{false}())
    obj_t = make_objective(k, p, xs, vs, Val{true}())
    @test obj_f(pack(k, vs)) == obj_t(pack(vs))

    # @btime Ξ($k, $p, $xs, $vs, $J)
    # @btime [Ξ($k, $p, $xs, $vs[:, i], $J) for i = 1:$J]

    τ_size = (d * J, n)

    Ξ_xs = [Ξ(k, p, xs, vs[:, i], J) for i = 1:J]  # <= more efficient that Ξ(k, p, xs, vs) which is (d × n × J)
    τ = vcat(Ξ_xs...)
    @test size(τ) == τ_size

    @test (sum(τ' * τ) - sum(diag(τ' * τ))) / (n * (n - 1)) ≈ fssd(k, p, xs, vs)

    using KernelGoodnessOfFit: fssd_from_τ, fssd_from_Ξ
    @test fssd_from_τ(τ) ≈ fssd_from_Ξ(Ξ_xs)
end
