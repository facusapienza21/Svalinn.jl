using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Svalinn
using Plots
using LaTeXStrings
using Statistics: mean
using StatsBase: ecdf
using HypothesisTests: OneSampleADTest
using Distributions: Uniform

default(fontfamily = "Computer Modern")

# Reads the chrons saved by sweep_theta_calibrated.jl (one file per θ under
# data/simulations/, plus master.txt) and reproduces the p-value / H-statistic plots,
# without rerunning any SDE simulations.

min_age = 80.0   # Ma; keep only chrons within the last min_age Ma
θ₀ = 0.5 / 1000
σ₀ = 1 / (2θ₀)

# Discard chrons shorter than this (Ma) by iteratively merging the globally shortest one
# into a neighbor (see `filter_short_reversals`). Applied here, at analysis time, to the
# raw (unfiltered) saved chrons — not baked into the simulation — so it can be changed
# without rerunning the (expensive) SDEs. 10,000 years = 0.01 Ma.
# min_chron_length = 1 / 1000
min_chron_length = 0.0

# `starts`/`stops` are a replicate's saved *normal*-polarity chrons (time-sorted,
# contiguous with the implied reverse-polarity gaps between them). Reconstruct the full
# alternating reversal sequence and wrap it as a minimal SynthesisDataset so `chrons`
# (which now uses the updated `filter_short_reversals`) can do the actual filtering,
# instead of duplicating its polarity-parity bookkeeping here.
function filtered_normal_chrons(starts::Vector{Float64}, stops::Vector{Float64};
                                 min_length::Real)
    reversals_full = Vector{Float64}(undef, 2 * length(starts))
    reversals_full[1:2:end] = starts
    reversals_full[2:2:end] = stops
    ds = SynthesisDataset(:sim, [reversals_full[1], reversals_full[end]],
                           [SignedPaleointensity(1.0), SignedPaleointensity(1.0)],
                           reversals_full[2:end-1])
    return chrons(ds; polarity = :normal, min_length = min_length)
end

# Polarity timescales tested against:
#   main       — CK95 (Cande & Kent 1995), ≤ min_age: data/GPTS_CK95.txt
#   background — CK95, full record (no age cut)
#   background — Ogg2012 (GPTS 2012), ≤ min_age: data/GPTS_2012.txt (converted from the last 4
#                columns of data/CK95-GPTS2012.csv by data/convert_GPTS2012.jl)
data_dir  = joinpath(@__DIR__, "data")
sim_dir   = joinpath(data_dir, "simulations")
gpts_file      = joinpath(data_dir, "GPTS_CK95.txt")
gpts2012_file  = joinpath(data_dir, "GPTS_2012.txt")
gpts      = read_polarity(gpts_file; name = :GPTS_CK95, min_age = min_age)
gpts_full = read_polarity(gpts_file; name = :GPTS_CK95_full, min_age = Inf)
gpts2012  = read_polarity(gpts2012_file; name = :GPTS2012, min_age = min_age)
println("CK95: $(length(gpts.start)) normal chrons ≤ $(round(Int, min_age)) Ma, " *
        "$(length(gpts_full.start)) in total; Ogg2012: $(length(gpts2012.start)) ≤ $(round(Int, min_age)) Ma")

# --- Reference run ---
ref_chrons = read_polarity(joinpath(sim_dir, "reference.txt"); name = :reference)
L_ref      = mean(chron_lengths(ref_chrons))
println("Reference Davis2021 (θ₀=$θ₀, σ₀=$σ₀): mean chron length = $(round(L_ref, digits = 3)) Ma")

# --- master.txt: one row per θ, with θ, σ_calib (from calibrate_sigma_theta.jl), mean
# chron length already computed by the run script (see sweep_theta_calibrated.jl) ---
master_rows = NamedTuple[]
open(joinpath(sim_dir, "master.txt")) do io
    readline(io)  # header
    for line in eachline(io)
        isempty(line) && continue
        c = split(line, '\t')
        push!(master_rows, (index = parse(Int, c[1]), file = c[2], theta = parse(Float64, c[3]),
                             sigma_calib = parse(Float64, c[4]),
                             mean_chron_length = parse(Float64, c[5]),
                             n_replicates_valid = parse(Int, c[6])))
    end
end
sort!(master_rows; by = r -> r.index)

n_grid  = length(master_rows)
θ_grid  = [r.theta for r in master_rows]
σ_calib = [r.sigma_calib for r in master_rows]

# --- per-θ chron files: group rows by Simulation (replicate) to reconstruct each
# replicate's PolarityDataset, compute H_statistic per replicate, then pool ---
H_data       = H_statistic(gpts)
H_data_full  = H_statistic(gpts_full)
H_data_2012  = H_statistic(gpts2012)
pvals_AD     = fill(NaN, n_grid)   # Anderson-Darling p-value vs the filtered data — the main test
A2vals       = fill(NaN, n_grid)   # AD test statistic (A²), same
pvals_AD_full = fill(NaN, n_grid)  # AD p-value vs the full (unfiltered) CK95 record
A2vals_full   = fill(NaN, n_grid)
pvals_AD_2012 = fill(NaN, n_grid)  # AD p-value vs Ogg2012 (≤min_age)
A2vals_2012   = fill(NaN, n_grid)
H_sim_grid   = [Float64[] for _ in 1:n_grid]

for (iθ, r) in enumerate(master_rows)
    by_rep = Dict{Int, Vector{Tuple{Float64,Float64}}}()
    open(joinpath(sim_dir, r.file)) do io
        readline(io)  # header
        for line in eachline(io)
            isempty(line) && continue
            c   = split(line, '\t')
            s, e = parse(Float64, c[1]), parse(Float64, c[2])
            rep  = parse(Int, c[3])
            push!(get!(by_rep, rep, Tuple{Float64,Float64}[]), (s, e))
        end
    end

    if isempty(by_rep)
        println("θ = $(r.theta) -> no valid replicates in $(r.file)")
        continue
    end

    H_pool = Float64[]
    for (rep, pairs) in by_rep
        starts = first.(pairs)
        stops  = last.(pairs)
        pd     = filtered_normal_chrons(starts, stops; min_length = min_chron_length)
        append!(H_pool, H_statistic(pd))
    end

    # Anderson-Darling via the probability integral transform: F_θ is the model's own
    # empirical CDF, built from the large pooled H_pool sample (well-defined enough to
    # treat as the reference here, though it does carry a little of its own sampling
    # noise since it's not the exact theoretical CDF). Under H0 (data ~ F_θ), the
    # transformed Y_i = F_θ(H_data_i) are i.i.d. Uniform(0,1); OneSampleADTest checks
    # that directly, weighting tail discrepancies more heavily than a KS test would — this
    # is the main test. Run against all three data variants.
    # ecdf() returns exactly 0.0/1.0 for query points at or beyond H_pool's own
    # min/max, and the AD statistic's log(Y)/log(1-Y) terms blow up (A² = Inf) at those
    # exact values, so Y is clamped away from {0,1} first.
    F_θ          = ecdf(H_pool)
    clamp01(y)   = clamp(y, 1e-4, 1 - 1e-4)
    ad_test      = OneSampleADTest(clamp01.(F_θ.(H_data)), Uniform())
    ad_test_full = OneSampleADTest(clamp01.(F_θ.(H_data_full)), Uniform())
    p_AD         = pvalue(ad_test)
    p_AD_full    = pvalue(ad_test_full)
    ad_test_2012 = OneSampleADTest(clamp01.(F_θ.(H_data_2012)), Uniform())
    pvals_AD_2012[iθ] = pvalue(ad_test_2012)
    A2vals_2012[iθ]   = ad_test_2012.A²

    pvals_AD[iθ]       = p_AD
    A2vals[iθ]         = ad_test.A²
    pvals_AD_full[iθ]  = p_AD_full
    A2vals_full[iθ]    = ad_test_full.A²
    H_sim_grid[iθ]     = H_pool
    println("θ = $(r.theta), σ_calib = $(r.sigma_calib) -> " *
            "AD A²=$(round(ad_test.A², digits = 4)) (p=$(round(p_AD, digits = 4))) " *
            "[vs full: A²=$(round(ad_test_full.A², digits = 4)) (p=$(round(p_AD_full, digits = 4)))] " *
            "[vs Ogg2012: A²=$(round(ad_test_2012.A², digits = 4)) (p=$(round(pvals_AD_2012[iθ], digits = 4)))], " *
            "L_sim = $(round(r.mean_chron_length, digits = 3)) Ma, " *
            "n_pooled = $(length(H_pool)) over $(r.n_replicates_valid) valid replicates")
end

function ecdf_xy(H::AbstractVector{<:Real})
    Hs = sort(H)
    n  = length(Hs)
    xs = Float64[0.0]
    ys = Float64[0.0]
    for (i, h) in enumerate(Hs)
        push!(xs, h); push!(ys, ys[end])
        push!(xs, h); push!(ys, i / n)
    end
    push!(xs, 1.0); push!(ys, ys[end])
    return xs, ys
end

# θ=0 (white-noise limit, see simulate_Davis2021) can't be placed on the log-scale θ
# axis used by panels b/c — log10(0) is undefined, and including 0 in the axis limits
# would break the whole log scale. It's handled separately: excluded from the usual
# per-θ colored series, and instead shown as a horizontal reference line (its p-value)
# in panel c, since that's the quantity that matters for comparing against it.
theta0_pos  = findfirst(r -> r.theta == 0.0, master_rows)
has_theta0  = theta0_pos !== nothing && !isnan(pvals_AD[theta0_pos])

valid_idx = findall(!isnan, pvals_AD)
has_theta0 && (valid_idx = filter(iθ -> iθ != theta0_pos, valid_idx))
xs_data, ys_data           = ecdf_xy(H_data)
xs_data_full, ys_data_full = ecdf_xy(H_data_full)
xs_data_2012, ys_data_2012 = ecdf_xy(H_data_2012)

# One color per θ (i.e. correlation length), shared across all three panels so a given
# θ can be tracked by eye between the CDF curve and its D/p points. Must be a
# materialized vector (not the ColorGradient itself) for per-point scatter coloring.
cmap   = cgrad(:amp, max(length(valid_idx), 2); categorical = true)
colors = [cmap[k] for k in 1:length(valid_idx)]

# Display θ in ky (kyr) rather than Ma — everything upstream stays in Ma internally,
# this is purely a plotting-units conversion. σ is now calibrated (see
# calibrate_sigma_theta.jl) so that mean chron length already matches the reference for
# every θ by construction — no post-hoc relabeling needed, raw θ is already the
# physically comparable coordinate.
θ_grid_ky = 1000 .* θ_grid
θ₀_ky     = 1000 * θ₀

xlims_θ = (0.8 * minimum(θ_grid_ky[valid_idx]), 1.2 * maximum(θ_grid_ky[valid_idx]))

# Plain decimal tick labels (0.1, 1, 10, ...) instead of GR's default scientific
# notation (10⁻¹, 10⁰, 10¹, ...) on the shared log-scale θ axis.
plain_number(x) = x >= 1 ? string(isinteger(x) ? Int(x) : round(x; digits = 2)) :
                            string(round(x; digits = ceil(Int, -log10(x)) + 1))

# --- Panel b (vertical strip, between the CDF panel and the c/d pair): GPTS_CK95
# polarity timescale as plain black/white stripes, age increasing downward, so the reader
# can see at a glance which record every other panel is being tested against.
function polarity_segments(pd::PolarityDataset, max_age::Real)
    segs = Tuple{Float64, Float64, Symbol}[]
    prev_stop = 0.0
    for (s, e) in zip(pd.start, pd.stop)
        s > prev_stop && push!(segs, (prev_stop, s, :reverse))
        push!(segs, (s, e, :normal))
        prev_stop = e
    end
    prev_stop < max_age && push!(segs, (prev_stop, max_age, :reverse))
    return segs
end

polarity_fig = plot(; ylims = (0, min_age), xlims = (0, 1), yflip = true, xaxis = false,
                       grid = false, legend = false, framestyle = :box, ylabel = "Age (Ma)",
                       bottom_margin = 8Plots.mm, left_margin = 12Plots.mm,
                       title = "b)", titlelocation = :left, titlefontsize = 11)
for (s, e, pol) in polarity_segments(gpts, min_age)
    plot!(polarity_fig, Shape([0, 1, 1, 0], [s, s, e, e]);
          color = (pol == :normal ? :black : :white), linecolor = :black, linewidth = 0.3,
          label = false)
end

# --- Panel a (left): H CDF — data vs a subset of θ. Spans the full figure height, with the
# same top/bottom margins as the c/d stack so its x-axis lines up with panel d's; it's
# kept square (equal x/y scale, both [0,1]) by the width given to its layout cell below,
# not by aspect_ratio — which would shrink and re-center the box and break the
# alignment. θ curves are drawn first so both data curves (the
# min_age-filtered one used for the tests, and the full, unfiltered CK95 record for
# reference) are layered on top and stay visible.
# Background comparison datasets, same color in every panel: full CK95 (gray) and
# Ogg2012 ≤min_age (teal). The main AD-test curve (CK95 ≤min_age, panels c/d) is a thick
# black line with θ-colored markers.
color_full = colorant"#7f8c8d"
color_2012 = colorant"#16a085"
lw_bg      = 1.5
lw_main    = 2.5

cdf_fig = plot(; xlabel = L"H", ylabel = L"\mathrm{CDF}(H)", xlims = (0, 1), ylims = (0, 1),
                  legend = false, left_margin = 9Plots.mm, bottom_margin = 8Plots.mm,
                  title = "Distribution of clustering statistics", titlefontsize = 11)
best_k = argmax(pvals_AD[valid_idx])   # best-fitting θ (highest AD p-value vs the filtered data)
# Only a subset of θ is drawn, to keep the panel readable: n_cdf_theta curves evenly spaced
# (in log θ) across the grid, always including the best-fitting θ (it replaces whichever
# picked θ is nearest to it).
n_cdf_theta = 8
cdf_ks = unique(round.(Int, range(1, length(valid_idx); length = n_cdf_theta)))
best_k in cdf_ks || (cdf_ks[argmin(abs.(cdf_ks .- best_k))] = best_k)
# All of panel a's curves, as a function so the zoom inset (added to the composed figure
# below) draws exactly the same thing; `kw` passes `subplot = …` for the inset.
function draw_cdf_curves!(p; kw...)
    for k in cdf_ks
        k == best_k && continue   # drawn last/thicker below, so it isn't covered by others
        xs_sim, ys_sim = ecdf_xy(H_sim_grid[valid_idx[k]])
        plot!(p, xs_sim, ys_sim; color = colors[k], lw = 1.3, label = false, kw...)
    end
    plot!(p, [0.0, 1.0], [0.0, 1.0]; color = :gray, lw = 1, ls = :dash, label = false, kw...)
    xs_best, ys_best = ecdf_xy(H_sim_grid[valid_idx[best_k]])
    plot!(p, xs_best, ys_best; color = colors[best_k], lw = 3, label = false, kw...)
    plot!(p, xs_data_full, ys_data_full; color = color_full, lw = lw_bg, label = false, kw...)
    plot!(p, xs_data_2012, ys_data_2012; color = color_2012, lw = lw_bg, label = false, kw...)
    plot!(p, xs_data, ys_data; color = :black, lw = 3, label = false, kw...)
    # θ=0 (white-noise limit) — not part of the θ-colored gradient (it isn't on the log-θ
    # grid at all), so drawn distinctly: black dashed, same style used for it in panels c/d.
    if has_theta0
        xs_0, ys_0 = ecdf_xy(H_sim_grid[theta0_pos])
        plot!(p, xs_0, ys_0; color = :black, ls = :dash, lw = 2, label = false, kw...)
    end
    return p
end
draw_cdf_curves!(cdf_fig)
annotate!(cdf_fig, 0.04, 0.96, text("a)", 11, :left))

# Zoom region shown as an inset in panel a's lower-right corner (inset added after the
# figure is composed, see below); outlined here on the main panel.
zoom_x, zoom_y = (0.18, 0.33), (0.07, 0.22)
plot!(cdf_fig, [zoom_x[1], zoom_x[2], zoom_x[2], zoom_x[1], zoom_x[1]],
               [zoom_y[1], zoom_y[1], zoom_y[2], zoom_y[2], zoom_y[1]];
      color = :black, lw = 0.8, label = false)

# --- Panel c (top right): AD statistic A² vs θ — the main test vs CK95 ≤min_age (filled
# circles colored by θ, thick black line), plus the two background comparisons (full CK95,
# Ogg2012) as plain thin colored lines, no markers, drawn first so the main curve sits on
# top. Shares x-axis with panel d below, so its x tick labels are blanked and its bottom
# margin is squeezed to zero. ---
A2_ymax = 1.1 * maximum(maximum(v[valid_idx]) for v in (A2vals, A2vals_full, A2vals_2012))
D_fig = plot(θ_grid_ky[valid_idx], A2vals_full[valid_idx];   # full CK95 AD test
    color  = color_full,
    lw     = lw_bg,
    xscale = :log10,
    xlims  = xlims_θ,
    ylims  = (0, A2_ymax),
    minorgrid = true,
    xformatter = _ -> "",
    ylabel = L"A^2",
    label  = false,
    legend = false,
    bottom_margin = 0Plots.mm,
    left_margin   = 8Plots.mm,
    title  = "Hypothesis testing",
    titlefontsize = 11,
)
plot!(D_fig, θ_grid_ky[valid_idx], A2vals_2012[valid_idx];
      color = color_2012, lw = lw_bg, label = false)   # Ogg2012 (≤min_age) AD test
plot!(D_fig, θ_grid_ky[valid_idx], A2vals[valid_idx];    # main test, on top
      marker = :circle, markercolor = colors, linecolor = :black, lw = lw_main, label = false)
vline!(D_fig, [θ₀_ky]; color = :black, lw = 1, ls = :dot, label = false)
# θ=0 (white-noise limit): horizontal reference line at its own A², same reasoning/style
# as the p-value line in panel c (no valid position on the log-scale θ axis).
has_theta0 && hline!(D_fig, [A2vals[theta0_pos]]; color = :black, lw = 1.5, ls = :dash, label = false)
annotate!(D_fig, 1.3 * xlims_θ[1], 0.93 * A2_ymax, text("c)", 11, :left))

# --- Panel d (bottom right): AD p-value vs θ, same three data variants ---
# Plotted on a "shifted logit" scale centered at α=0.05 (so p=0.05 sits exactly at
# y=0), which stretches out the region near the significance threshold that a linear
# [0,1] axis compresses. p is clamped away from {0,1} first so the log-ratio stays
# finite for p-values that round to exactly 0 or 1.
shifted_logit(p, α = 0.05) = (pc = clamp(p, 1e-4, 1 - 1e-4);
                               log10((pc * (1 - α)) / (α * (1 - pc))))

p_ticks = [0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
y_ticks = shifted_logit.(p_ticks)

p_fig = plot(; xscale = :log10,
    xlims  = xlims_θ,
    minorgrid = true,
    xformatter = plain_number,
    xlabel = L"\theta\ (\mathrm{ky})",
    ylabel = L"p\mathrm{-value}",
    yticks = (y_ticks, string.(p_ticks)),
    ylims  = (minimum(y_ticks), maximum(y_ticks)),
    legend = false,
    top_margin    = 0Plots.mm,
    bottom_margin = 8Plots.mm,
    left_margin   = 8Plots.mm,
)
# Rejection region (p < 0.05) as a light reddish background shade, with a solid (not
# dashed) line marking the exact threshold — easier to read at a glance than a dashed
# line alone, and doesn't compete visually with the dashed/dotted background series.
hspan!(p_fig, [minimum(y_ticks), shifted_logit(0.05)]; color = :firebrick, alpha = 0.08, label = false)
hline!(p_fig, [shifted_logit(0.05)]; color = :firebrick, lw = 1, ls = :solid, alpha = 0.6, label = false)
# Background comparisons: full CK95 and Ogg2012 (≤min_age) AD tests — thin colored lines.
plot!(p_fig, θ_grid_ky[valid_idx], shifted_logit.(pvals_AD_full[valid_idx]);
      color = color_full, lw = lw_bg, label = false)
plot!(p_fig, θ_grid_ky[valid_idx], shifted_logit.(pvals_AD_2012[valid_idx]);
      color = color_2012, lw = lw_bg, label = false)
# Main test: AD p-value vs the filtered data — colored, solid, filled circles.
plot!(p_fig, θ_grid_ky[valid_idx], shifted_logit.(pvals_AD[valid_idx]);
      marker = :circle, markercolor = colors, linecolor = :black, lw = lw_main, label = false)
vline!(p_fig, [θ₀_ky]; color = :black, lw = 1, ls = :dot, label = false)
# θ=0 (white-noise limit): shown as a horizontal line at its own AD p-value, since it
# has no valid position on the log-scale θ axis (see note above valid_idx).
if has_theta0
    hline!(p_fig, [shifted_logit(pvals_AD[theta0_pos])]; color = :black, lw = 1.5,
           ls = :dash, label = false)
end
annotate!(p_fig, 1.3 * xlims_θ[1], 0.9 * maximum(y_ticks), text("d)", 11, :left))

# --- Shared legend panel (far right), instead of a per-subplot legend. Raw material only
# — the final legend is laid out by hand in Illustrator. The per-θ colors are not listed
# here; they're given by the colorbar below it instead. ---
legend_fig = plot(; legend = :left, framestyle = :none, grid = false,
                     xaxis = false, yaxis = false)
plot!(legend_fig, [NaN], [NaN]; color = :black, lw = 3, label = "CK95 (≤$(round(Int, min_age)) Ma)")
plot!(legend_fig, [NaN], [NaN]; color = color_full, lw = lw_bg, label = "CK95 (full)")
plot!(legend_fig, [NaN], [NaN]; color = color_2012, lw = lw_bg, label = "Ogg2012 (≤$(round(Int, min_age)) Ma)")
plot!(legend_fig, [NaN], [NaN]; color = :gray, lw = 3, label = "Best-fit θ (max AD p-value)")
plot!(legend_fig, [NaN], [NaN]; color = :black, marker = :circle, lw = lw_main,
      label = "AD (main) vs CK95 ≤$(round(Int, min_age)) Ma")
plot!(legend_fig, [NaN], [NaN]; color = :gray, lw = 1, ls = :dash, label = "Uniform")
plot!(legend_fig, [NaN], [NaN]; color = :black, lw = 1, ls = :dot,
      label = L"\mathrm{reference}\ \theta_0")
if has_theta0
    plot!(legend_fig, [NaN], [NaN]; color = :black, lw = 1.5, ls = :dash,
          label = L"\theta = 0\ \mathrm{(white\ noise)}")
end
plot!(legend_fig, [NaN], [NaN]; color = :firebrick, lw = 1, label = "p = 0.05")

# --- θ colorbar: one rectangle per grid θ, in that θ's color, spanning the geometric
# midpoints to its neighbors on a log θ axis (the grid is log-spaced, so the categorical
# colors map linearly onto log θ). Drawn as plain shapes rather than a Plots colorbar so
# the colors are exactly the ones used in the panels and every power-of-10 tick is
# placed explicitly. ---
θ_cb   = θ_grid_ky[valid_idx]
θ_mid  = sqrt.(θ_cb[1:end-1] .* θ_cb[2:end])
θ_lo   = [θ_cb[1]^2 / θ_mid[1]; θ_mid]
θ_hi   = [θ_mid; θ_cb[end]^2 / θ_mid[end]]
cb_ticks = [10.0^p for p in ceil(Int, log10(θ_lo[1])):floor(Int, log10(θ_hi[end]))]
cbar_fig = plot(; xscale = :log10, xlims = (θ_lo[1], θ_hi[end]), ylims = (0, 1),
                   xticks = (cb_ticks, plain_number.(cb_ticks)), yaxis = false,
                   grid = false, legend = false, framestyle = :box,
                   xlabel = L"\theta\ (\mathrm{ky})",
                   top_margin = 2Plots.mm, bottom_margin = 8Plots.mm)
for k in eachindex(θ_cb)
    plot!(cbar_fig, Shape([θ_lo[k], θ_hi[k], θ_hi[k], θ_lo[k]], [0, 0, 1, 1]);
          color = colors[k], linecolor = colors[k], linewidth = 0, label = false)
end

# The @layout DSL doesn't reliably honor a width fraction applied to a stacked pair of
# cells (only the first cell ends up sized; tested directly and confirmed) — composing
# the right (legend + colorbar) column as its own fully-rendered sub-figure first, then
# combining it with the other cells via a plain single-row layout (which *does* size
# correctly), sidesteps the bug entirely.
right_fig = plot(legend_fig, cbar_fig; layout = grid(2, 1, heights = [0.85, 0.15]))

# a{…w} is tuned so panel a's plot area comes out square at this figure size — if `size`
# changes, re-tune it.
l = @layout [a{0.407w} b{0.042w} [c; d] e{0.16w}]
fig = plot(cdf_fig, polarity_fig, D_fig, p_fig, right_fig;
    layout = l,
    size   = (1650, 600),
)

# Zoom inset in panel a (fig's subplot 1), lower-right corner — the region below the
# diagonal there is empty. Added to the composed figure rather than to cdf_fig because
# insets don't reliably survive being combined into a layout. Both zoom ranges are 0.15
# wide and the inset box is square, so it keeps panel a's equal x/y scale.
inset_sp = length(fig.subplots) + 1
plot!(fig, [NaN], [NaN]; inset = (1, bbox(0.03, 0.08, 0.36, 0.36, :right, :bottom)),
      subplot = inset_sp, xlims = zoom_x, ylims = zoom_y, label = false, legend = false,
      framestyle = :box, grid = false,
      xticks = range(zoom_x...; step = 0.05), yticks = range(zoom_y...; step = 0.05),
      xformatter = _ -> "", yformatter = _ -> "",   # tick marks only, no numbers
      background_color_inside = :white)
draw_cdf_curves!(fig; subplot = inset_sp)

mkpath(joinpath(@__DIR__, "figures"))
savefig(fig, joinpath(@__DIR__, "figures", "test1_figure.pdf"))
display(fig)
