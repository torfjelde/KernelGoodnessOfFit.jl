using Distributions, ForwardDiff
using GoodnessOfFit

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

# simple example
T = GoodnessOfFit.KSDTest(rbf_, 100, 0.05)

@test perform(T, q_, xs_)
perform(T, p_, xs_)
