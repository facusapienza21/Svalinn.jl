using ComplexityMeasures
using Plots
using Statistics
using LinearAlgebra

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
- `noise_curve`    : if `true`, overlay the colored-noise (OU) reference curve via
                     `plot_noise_curve!`; all `noise_*` kwargs are forwarded to it

Both curves are drawn with a dashed line style. "max complexity" is labelled at the
left edge of the upper curve; "min complexity" at the right edge of the lower curve.
"""
function plot_CH_plane(p::Plots.AbstractPlot; d::Int=5, τ::Int=1,
                       xlim=(0.0, 1.0), ylim=(0.0, 0.5),
                       num_max::Int=1, num_min::Int=1000,
                       curve_color=:black, curve_linewidth=1,
                       noise_curve::Bool=false, noise_kwargs...)
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
          xlabel="Entropy", ylabel="Complexity")
    plot!(p, H_min[ord_min], C_min[ord_min];
          color=curve_color, linewidth=curve_linewidth,
          linestyle=:dash, label="")

    if noise_curve
        plot_noise_curve!(p; d=d, τ=τ, noise_kwargs...)
    end

    # Labels drawn last so they sit on top of all curves
    i_peak = argmax(C_max[ord_max])
    annotate!(p, H_max[ord_max][i_peak], C_max[ord_max][i_peak],
              text("max complexity", curve_color, :center, :bottom, 7))

    i_mid = length(H_min[ord_min]) ÷ 2
    annotate!(p, H_min[ord_min][i_mid], C_min[ord_min][i_mid],
              text("min complexity", curve_color, :center, :bottom, 7))

    return p
end

"""
    plot_noise_curve!(p; d, τ, n_realizations, σ, λ_max, n_λ, curve_color, curve_linewidth)

Add a colored-noise reference curve to the CH plane by sweeping the correlation
length λ of the exponential (OU) kernel via `add_noise`. The reference signal is a
flat zero `SynthesisDataset` on a uniform time grid from -2.0 to 0.0 Ma (step 0.001 Ma),
matching the standard G12 simulation grid. For each λ, `n_realizations` noise
realisations are averaged to give one point on the curve.

λ = 0 is included as the white-noise limit; λ is then swept log-spaced up to `λ_max`
(in Ma). All λ values are in Ma, consistent with the time axis of the reference signal.

- `n_realizations` : realisations averaged per λ (default 20)
- `σ`              : noise standard deviation (default 1.0; only the shape of the
                     curve matters, not the amplitude)
- `λ_min`, `λ_max` : correlation-length range in Ma, log-spaced (default 0.001 to 2.0)
- `n_λ`            : number of λ values, excluding the λ=0 endpoint (default 50)
- `curve_color`    : line color (default `:darkorange`)
- `curve_linewidth`: line width (default 1.5)
"""
function plot_noise_curve!(p::Plots.AbstractPlot;
    d               ::Int     = 5,
    τ               ::Int     = 1,
    n_realizations  ::Int     = 20,
    σ                         = 1.0,
    λ_min           ::Float64 = 0.001,
    λ_max           ::Float64 = 2.0,
    n_λ             ::Int     = 10,
    curve_color               = :darkorange,
    curve_linewidth           = 1.5,
    )
    # Flat zero reference signal on the standard G12 time grid (-2.0 to 0.0 Ma, step 0.001)
    t_ref   = collect(range(-2.0, 0.0; step=0.001))
    ds_flat = SynthesisDataset(:_noise_ref, t_ref, SignedPaleointensity.(zeros(length(t_ref))))

    # λ sweep: include λ=0 (white noise) as first point, then log-spaced
    λ_values = [0.0; exp.(range(log(λ_min), log(λ_max); length=n_λ))]
    H_curve  = Vector{Float64}(undef, length(λ_values))
    C_curve  = Vector{Float64}(undef, length(λ_values))

    for (i, λ) in enumerate(λ_values)
        Hs = Vector{Float64}(undef, n_realizations)
        Cs = Vector{Float64}(undef, n_realizations)
        for r in 1:n_realizations
            ds_noisy   = add_noise(ds_flat; σ=σ, λ=λ)
            Hs[r], Cs[r] = ordinal_entropy_complexity(
                [x.value for x in ds_noisy.value]; d=d, τ=τ
            )
        end
        H_curve[i] = mean(Hs)
        C_curve[i] = mean(Cs)
    end

    @infiltrate

    plot!(p, H_curve, C_curve;
          color=curve_color, linewidth=curve_linewidth, label="colored noise (OU)")
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
