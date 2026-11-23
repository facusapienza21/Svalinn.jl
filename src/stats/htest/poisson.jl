using Plots

"""
    poisson_ks_test(pd::PolarityDataset; exact::Bool = false)

One-sample Kolmogorov–Smirnov test, following Carbone et al. (2005), of whether the
crossing-time sequence underlying `pd` (see `H_statistic`) is consistent with a
Poisson process (i.i.d. Exponential lags). Tests `H_statistic(pd)` against
`Uniform(0, 1)` and returns the corresponding `HypothesisTests.jl` result
(`ExactOneSampleKSTest` if `exact = true`, otherwise the large-sample
`ApproximateOneSampleKSTest`).

A small p-value rejects the Poisson-process null hypothesis, indicating clustering
(or anti-clustering) of the crossing times.
"""
function poisson_ks_test(pd::PolarityDataset; exact::Bool = false)
    H = H_statistic(pd)
    return exact ?
        ExactOneSampleKSTest(H, Uniform(0, 1)) :
        ApproximateOneSampleKSTest(H, Uniform(0, 1))
end

"""
    plot_H_cdf(pd::PolarityDataset; ecdf_color=:steelblue, uniform_color=:firebrick, kwargs...)
        -> Plots.Plot

Plot the empirical CDF of `H_statistic(pd)` against the Uniform(0,1) CDF (the identity
line), as in Carbone et al. (2005). The empirical CDF is drawn as an explicit
right-continuous step function — it jumps by `1/n` exactly at each sorted `H_i` and is
constant in between — rather than a smoothed/interpolated curve, since `H` is a finite
discrete sample. Under the Poisson-process null hypothesis the two curves should
closely track each other; see `poisson_ks_test` for the corresponding significance
test.

`kwargs` are forwarded to the underlying `Plots.plot` call.
"""
function plot_H_cdf(pd::PolarityDataset;
                     ecdf_color = :steelblue, uniform_color = :firebrick, kwargs...)
    H = sort(H_statistic(pd))
    n = length(H)

    xs = Float64[0.0]
    ys = Float64[0.0]
    for (i, h) in enumerate(H)
        push!(xs, h); push!(ys, ys[end])
        push!(xs, h); push!(ys, i / n)
    end
    push!(xs, 1.0); push!(ys, ys[end])

    p = plot(xs, ys;
        label        = "empirical CDF of H",
        color        = ecdf_color,
        lw           = 1.5,
        xlabel       = "H",
        ylabel       = "CDF",
        xlims        = (0, 1),
        ylims        = (0, 1),
        aspect_ratio = :equal,
        size         = (500, 500),
        legend       = :bottomright,
        kwargs...,
    )
    plot!(p, [0.0, 1.0], [0.0, 1.0];
        label = "Uniform(0,1) CDF",
        color = uniform_color,
        lw    = 1.5,
        ls    = :dash,
    )
    return p
end
