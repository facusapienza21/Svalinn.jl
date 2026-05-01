"""
    ami(x, τ; nbins) -> Float64

Compute the Average Mutual Information (AMI) between `x` and its lagged version `x[τ:]`
using a histogram estimator with `nbins` bins. If `nbins=0` (default), the number of
bins is chosen automatically as `round(Int, √(N/5))`.
"""
function ami(x::Vector{<:Real}, τ::Int; nbins::Int = 0)
    N = length(x)
    if nbins == 0
        nbins = max(2, round(Int, sqrt(N / 5)))
    end

    xmin, xmax = minimum(x), maximum(x)
    edges  = range(xmin, xmax; length = nbins + 1)
    binof(v) = clamp(searchsortedlast(edges, v), 1, nbins)

    bx  = [binof(x[i]) for i in 1:N]
    bx1 = bx[1:N-τ]
    bx2 = bx[1+τ:N]
    M   = length(bx1)

    joint = zeros(nbins, nbins)
    for i in 1:M
        joint[bx1[i], bx2[i]] += 1
    end
    joint ./= M
    px = vec(sum(joint; dims = 2))
    py = vec(sum(joint; dims = 1))

    I = 0.0
    for i in 1:nbins, j in 1:nbins
        if joint[i, j] > 0 && px[i] > 0 && py[j] > 0
            I += joint[i, j] * log2(joint[i, j] / (px[i] * py[j]))
        end
    end
    return I
end

"""
    ami_curve(x, τmax; nbins) -> Vector{Float64}

Return the AMI values for lags `τ = 1, 2, …, τmax`.
"""
function ami_curve(x::Vector{<:Real}, τmax::Int = 50; nbins::Int = 0)
    return [ami(x, τ; nbins = nbins) for τ in 1:τmax]
end

"""
    best_tau(x, τmax; nbins) -> (τ_opt::Int, ami_vals::Vector{Float64})

Find the optimal embedding delay `τ` as the first local minimum of the AMI curve.
If no local minimum exists within `1:τmax`, falls back to the global minimum.

Returns the optimal `τ` and the full AMI curve for inspection.
"""
function best_tau(x::Vector{<:Real}, τmax::Int = 50; nbins::Int = 0)
    ami_vals = ami_curve(x, τmax; nbins = nbins)

    for i in 2:length(ami_vals)-1
        if ami_vals[i] < ami_vals[i-1] && ami_vals[i] < ami_vals[i+1]
            return i, ami_vals
        end
    end

    return argmin(ami_vals), ami_vals
end

"""
    acf(x, τmax; window) -> Vector{Float64}

Compute the normalized autocorrelation function of `abs.(x)` for lags `τ = 1, …, τmax`.
Working on the absolute value handles bimodal signals (e.g. ±polarity reversals).
A running mean with the given `window` (default 50) is subtracted before computing
the ACF to remove slow trends.
"""
function acf(x::Vector{<:Real}, τmax::Int; window::Int = 50)
    # Take absolute value first to handle bimodal signals (e.g. ±polarity),
    # then subtract a running mean with the given window to remove slow trends
    a    = abs.(x)
    n    = length(a)
    half = window ÷ 2
    ā    = [mean(a[max(1, i-half):min(n, i+half)]) for i in 1:n]
    ac   = a .- ā
    denom = sum(ac .^ 2)
    return [sum(ac[1:end-τ] .* ac[1+τ:end]) / denom for τ in 1:τmax]
end

"""
    best_tau_acf(x, τmax) -> (τ_opt::Int, r::Vector{Float64})

Find the optimal embedding delay as the last positive lag before the first zero
crossing of the ACF. Falls back to the lag closest to zero if no crossing is found
within `1:τmax`.

Returns the optimal `τ` and the full ACF for inspection.
"""
function best_tau_acf(x::Vector{<:Real}, τmax::Int = 50; window::Int = 50)
    r = acf(x, τmax; window = window)
    for τ in 2:lastindex(r)
        if r[τ-1] > 0 && r[τ] <= 0
            return τ - 1, r
        end
    end
    return argmin(abs.(r)), r
end
