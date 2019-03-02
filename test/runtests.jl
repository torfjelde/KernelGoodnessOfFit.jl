using Test

@testset "Hypothesis tests" begin include("test_hypothesis_tests.jl") end
@testset "FSSD" begin include("fssd_tests.jl") end
@testset "KSD" begin include("ksd_tests.jl") end


