using Test, ForwardDiff, KernelGoodnessOfFit

using KernelGoodnessOfFit:
    GaussianRBF,
    Matern25Kernel,
    k_dx, k_dy, k_dxdy

@testset "Kernel: derivatives / gradients" begin
    ks = [
        GaussianRBF(1.0),
        Matern25Kernel(1.0)
    ]

    for k in ks
        @testset "$k" begin
            x = randn(10)
            y = randn(10)

            @test k_dx(k, x, y) ≈ ForwardDiff.gradient(z -> kernel(k, z, y), x);
            @test k_dy(k, x, y) ≈ ForwardDiff.gradient(z -> kernel(k, x, z), y);
            @test k_dxdy(k, x, y) ≈ ForwardDiff.jacobian(z -> k_dx(k, x, z), y);
        end
    end
end
