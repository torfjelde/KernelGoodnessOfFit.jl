"""
Goodness of Fit Test.
"""
abstract type GoodnessOfFitTest end

# implement this for whatever gof-test
"""
Performs the specified goodness-of-fit test.

# Arguments
- `t::KSDTest|FSSDTest`: the parameters for the test.
- `q`: model which needs to have an implementation for `gradlogpdf`.
- `x::AbstractArray`: samples to test against.
"""
function perform(::GoodnessOfFitTest, q, xs) end

abstract type GoodnessOfFitResult end
