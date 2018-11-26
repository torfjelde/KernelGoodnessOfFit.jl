using Distributions, ForwardDiff, GoodnessOfFit

# setup the problem
rbf_ = GaussianRBF(1.0)

μ = 0.0
σ² = 1.0
p_ = MultivariateNormal([μ, 0], [σ² 0; 0 σ²])  # true
q_ = MultivariateNormal([μ, 5.0], [σ² 0; 0 σ²])  # false

vs_ = [0.0 ; 0.0]
vs_ = reshape(vs_, 2, 1)

n_ = 400
xs_ = rand(p_, n_)

# new vs old: verify that new implementation is correct
begin
    @test GoodnessOfFit.fssd(rbf_, q_, xs_, vs_) == GoodnessOfFit.fssd_old(rbf_, q_, xs_, vs_)
    @test GoodnessOfFit.fssd(rbf_, p_, xs_, vs_) == GoodnessOfFit.fssd_old(rbf_, p_, xs_, vs_)
end

# ensure new method is faster :)
begin
    t1 = @elapsed GoodnessOfFit.fssd(rbf_, q_, xs_, vs_) 
    t2 = @elapsed GoodnessOfFit.fssd_old(rbf_, q_, xs_, vs_)

    @test t1 < t2
end

# sizes: testing of sizes
begin
    # setup for tests
    Ξ_ = GoodnessOfFit.compute_Ξ(rbf_, q_, xs_, vs_)

    J, = size(Ξ_)
    d, n = size(Ξ_[1])
    
    @test (d, J) == size(vs_)
    @test (d, n) == size(xs_)

    τ = GoodnessOfFit.τ_from_Ξ(Ξ_)
    @test size(τ) == (d * J, n)
    
    μ, Σ = GoodnessOfFit.Σₚ(τ)
    @test size(μ) == (d * J, 1)
    @test size(Σ) == (d * J, d * J)
    
    σ = GoodnessOfFit.σ²_H₁(μ, Σ)
    @test size(σ) == ()
end

# gradients: verify gradients
begin
    Δv = ForwardDiff.gradient(v -> GoodnessOfFit.fssd_H₁_opt_factor(rbf_, q_, xs_, v), vs_)
    @test size(Δv) == size(vs_)
    
    # fails for some reason; but does not fail when I actually run the test...interesting, eh?
    # Δσₖ = ForwardDiff.derivative(σₖ -> fssd_H₁(GaussianRBF(σₖ), q_, xs_, vs_), 1.0)
end


# values: testing specific values
begin
    V = reshape([0; 0], (2, 1))

    X = [1.0 2.0 3.0; 2.0 3.0 4.0]

    Ξ_1_true = [
        -0.116086 -0.00425237 -1.58109e-5; 
        0.0580429 -0.00106309 -7.90543e-6
    ]
    Ξ_ = GoodnessOfFit.compute_Ξ(rbf_, q_, X, V)

    @test Ξ_[1] ≈ Ξ_1_true atol=0.01

    fssd_true = 0.0004333865124975432

    @test GoodnessOfFit.fssd(Ξ_) * size(X)[2] ≈ fssd_true

    σ₁² = 2.8732542480282987e-5

    τ_ = GoodnessOfFit.τ_from_Ξ(Ξ_)
    μ_, Σ_ = GoodnessOfFit.Σₚ(τ_)

    @test GoodnessOfFit.σ²_H₁(μ_, Σ_) ≈ σ₁²
end
