"""
    H_statistic(pd::PolarityDataset) -> Vector{Float64}

Compute the ``H_i`` statistic of Carbone et al. (2005) for the interior points of the
full crossing-time sequence underlying `pd` (e.g. from `simulate_chrons`).

Every chron start/stop in `pd` is itself a reversal (reverse→normal / normal→reverse
respectively), so `t = sort(vcat(pd.start, pd.stop))` recovers the complete,
alternation-ordered sequence of crossings.

For each interior index `i` of `t` (with `i-2` and `i+2` both valid, i.e.
`3 ≤ i ≤ n-2`):

    δt_i = min(t[i+1] - t[i], t[i] - t[i-1])
    δτ_i = t[i+2] - t[i+1]   if δt_i == t[i+1] - t[i]
         = t[i-1] - t[i-2]   otherwise
    H_i  = 2δt_i / (2δt_i + δτ_i)

`δt_i` is the shorter of the two chrons adjacent to `t[i]`; `δτ_i` is the chron on the
far side of that shorter one. Under the null hypothesis that the lags between
crossings are i.i.d. Exponential (a Poisson point process), `H_i` is
Uniform(0,1)-distributed and independent of the other `H_j`.
"""
function H_statistic(pd::PolarityDataset)
    t = sort(vcat(pd.start, pd.stop))
    n = length(t)
    n >= 5 || throw(ArgumentError("need at least 5 crossing times, got $n"))

    H = Vector{Float64}(undef, n - 4)
    for i in 3:n-2
        forward  = t[i+1] - t[i]
        backward = t[i]   - t[i-1]
        if forward <= backward
            δt = forward
            δτ = t[i+2] - t[i+1]
        else
            δt = backward
            δτ = t[i-1] - t[i-2]
        end
        H[i-2] = 2δt / (2δt + δτ)
    end
    return H
end
