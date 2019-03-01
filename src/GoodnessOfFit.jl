module GoodnessOfFit

# exports
export
    # kernels
    Kernel, kernel, GaussianRBF,

    # ksd
    u_p,
    Sᵤ_estimate,
    KSDTest,

    # bootstrap
    degenerate_bootstrap,

    # Finite Set Stein Discrepancy
    ξ,
    fssd,
    compute_Ξ,
    τ_from_Ξ,
    Σₚ,
    σ²_H₁,
    fssd_H₁_opt_factor,
    FSSDTest,
    FSSDopt,
    FSSDrand,

    # core
    perform

include("core.jl")
include("bootstrap.jl")
include("kernels.jl")
include("ksd.jl")
include("fssd.jl")

end # module
