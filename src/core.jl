using HypothesisTests
import HypothesisTests: HypothesisTest, pvalue

"""
Goodness of Fit Test.
"""
abstract type KernelGoodnessOfFitTest <: HypothesisTest end

# implement this for whatever gof-test
"""
Performs the specified goodness-of-fit test.

# Arguments
- `t::KSDTest|FSSDTest`: the parameters for the test.
- `q`: model which needs to have an implementation for `gradlogpdf`.
- `x::AbstractArray`: samples to test against.
"""
function perform(::KernelGoodnessOfFitTest, q, xs) end

# HypothesisTests.jl interface
function pvalue(::KernelGoodnessOfFitTest) end

abstract type KernelGoodnessOfFitResult end
