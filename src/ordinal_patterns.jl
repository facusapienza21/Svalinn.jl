using ComplexityMeasures
using Plots

"""
    plot_CH_plane(p; d::Int=5, τ::Int=1,
                 xlim=(0.0, 1.0), ylim=(0.0, 1.0),
                 num_max::Int=1, num_min::Int=1000,
                 curve_color=:black, curve_linewidth=1)

Mutate the `Plots` plot object `p` by drawing the theoretical maximum- and
minimum-complexity boundary curves that bound the region of physically
realizable (H, C) pairs in the complexity-entropy (CH) plane, for embedding
dimension `d` and delay `τ` (Rosso et al., 2007).

The function adds only the template boundary curves to `p`; the caller can
then overlay data points with `scatter!` or additional `plot!` calls.

- `d`              : embedding dimension, must match the value used for signals
- `τ`              : embedding delay (default 1)
- `xlim`           : H-axis limits; default `(0.0, 1.0)`
- `ylim`           : C-axis limits; default `(0.0, 1.0)`
- `num_max`        : multiplier for points on the maximum curve
- `num_min`        : number of points on the minimum curve
- `curve_color`    : color of the boundary lines and labels; default `:black`
- `curve_linewidth`: line width; default `1`

Both curves are drawn with a dashed line style. "max complexity" is labelled at the
left edge of the upper curve; "min complexity" at the right edge of the lower curve.
"""
function plot_CH_plane(p::Plots.AbstractPlot; d::Int=5, τ::Int=1,
                       xlim=(0.0, 1.0), ylim=(0.0, 0.5),
                       num_max::Int=1, num_min::Int=1000,
                       curve_color=:black, curve_linewidth=1)
    o = OrdinalPatterns(; m=d, τ=τ)
    c = StatisticalComplexity(; o=o)
    min_raw, max_raw = entropy_complexity_curves(c; num_max=num_max, num_min=num_min)

    H_max = [Float64(pt[1]) for pt in max_raw]
    C_max = [Float64(pt[2]) for pt in max_raw]
    H_min = [Float64(pt[1]) for pt in min_raw]
    C_min = [Float64(pt[2]) for pt in min_raw]

    ord_max = sortperm(H_max)
    ord_min = sortperm(H_min)

    plot!(p, H_max[ord_max], C_max[ord_max];
          color=curve_color, linewidth=curve_linewidth,
          linestyle=:dash, label="", xlims=xlim, ylims=ylim,
          xlabel="Entropy", ylabel="Complexity", legend=:topleft)
    plot!(p, H_min[ord_min], C_min[ord_min];
          color=curve_color, linewidth=curve_linewidth,
          linestyle=:dash, label="")

    # Label max-complexity curve above its peak (highest C point, open space above)
    i_peak = argmax(C_max[ord_max])
    annotate!(p, H_max[ord_max][i_peak], C_max[ord_max][i_peak],
              text("max complexity", curve_color, :center, :bottom, 7))

    # Label min-complexity curve above its midpoint (open space between the two curves)
    i_mid = length(H_min[ord_min]) ÷ 2
    annotate!(p, H_min[ord_min][i_mid], C_min[ord_min][i_mid],
              text("min complexity", curve_color, :center, :bottom, 7))


    return p
end

"""
    ordinal_entropy(x::AbstractVector; d::Int=5, τ::Int=1) -> Float64

Compute the normalized permutation entropy H ∈ [0, 1] of a uniformly-sampled
discrete time series `x`. Only the values of `x` are used — no time axis is needed.

- `d` : embedding dimension (pattern length), must be ≥ 2
- `τ` : embedding delay

H = 0 implies a perfectly regular (periodic) signal.
H = 1 implies a maximally random signal.
"""
function ordinal_entropy(x::AbstractVector; d::Int=5, τ::Int=1)
    if length(x) < factorial(d + 1)
        @warn "Time series length ($(length(x))) is shorter than (d+1)! = $(factorial(d+1)) for d=$d. Entropy estimate may be unreliable."
    end
    ordinal_patterns = OrdinalPatterns(; m=d, τ=τ)
    return information_normalized(Shannon(), ordinal_patterns, x)
end

"""
    ordinal_complexity(x::AbstractVector; d::Int=5, τ::Int=1) -> Float64

Compute the statistical complexity of a uniformly-sampled discrete time series
`x` using the Jensen-Shannon divergence from the uniform ordinal distribution.
Only the values of `x` are used — no time axis is needed.

- `d` : embedding dimension (pattern length)
- `τ` : embedding delay
"""
function ordinal_complexity(x::AbstractVector; d::Int=5, τ::Int=1)
    if length(x) < factorial(d + 1)
        @warn "Time series length ($(length(x))) is shorter than (d+1)! = $(factorial(d+1)) for d=$d. Complexity estimate may be unreliable."
    end
    ordinal_patterns = OrdinalPatterns(; m=d, τ=τ)
    c = StatisticalComplexity(; o=ordinal_patterns)
    return complexity(c, x)
end

"""
    ordinal_entropy_complexity(x::AbstractVector; d::Int=5, τ::Int=1) -> Tuple{Float64, Float64}

Compute both the normalized permutation entropy H and the statistical complexity C
of a uniformly-sampled discrete time series `x`, returning them as `(H, C)`.

- `d` : embedding dimension (pattern length)
- `τ` : embedding delay
"""
function ordinal_entropy_complexity(x::AbstractVector; d::Int=5, τ::Int=1)
    H = ordinal_entropy(x; d=d, τ=τ)
    C = ordinal_complexity(x; d=d, τ=τ)
    return H, C
end
