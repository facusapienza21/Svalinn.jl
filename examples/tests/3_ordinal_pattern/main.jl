# Test No3: Fluctuations — ordinal-pattern majorization test (main.tex, Section "Test No3:
# Fluctuations", Figure fig:ordinal-pattern).
#
# For every calibrated (θ, σ) of the Davis2021 SDE (examples/tests/data/sigma_calibration.txt,
# shared with tests 1 and 2) and every valid RPI record, test
#
#   H0: the ordinal-pattern distribution of the model majorizes that of the RPI record
#       (i.e. the RPI can be a noise-distorted version of the model, sapienza2026noise)
#
# with the bootstrap majorization test `test_op_majorization` of StatsOrdinalPatterns.jl,
# and count the records that pass (p ≥ α). The Alberti et al. (2023) record is analysed
# separately in main_alberti.jl.
#
# The majorization machinery no longer lives in NoisyCE: Lorenz curves and the
# entropy–complexity plane come from (dev) ComplexityMeasures.jl, the test from (dev)
# StatsOrdinalPatterns.jl. NoisyCE itself is not needed here.
#
# PARALLEL EXECUTION — the θ loop is multithreaded; start Julia with several threads:
#
#   julia -t auto --project=examples examples/tests/3_ordinal_pattern/main.jl
#
# Results are cached per θ in data/results/theta_XX.txt with a stamp of every parameter
# that affects them; a θ is recomputed automatically when any of them changes, so
# re-running to restyle the figure is fast.
#
# Output (figures/):
#   summary.pdf   main.tex Figure fig:ordinal-pattern —
#                 a) RPI examples, b) % records passing vs θ, c) entropy–complexity plane

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Svalinn
ENV["GKSwstype"] = "nul"   # headless GR: figures are only saved to disk
using Plots
using LaTeXStrings
using ComplexityMeasures                        # OrdinalPatterns, StatisticalComplexity, …
using ComplexityMeasures: lorenz_curve, entropy_complexity, entropy_complexity_curves
import StatsOrdinalPatterns as SOP   # test internals, so that gap-filtered patterns can be used
using Statistics: mean, std
using Random
using MAT

default(fontfamily = "Computer Modern")

println("Running with $(Threads.nthreads()) thread(s)")
Threads.nthreads() == 1 &&
    @warn "Single thread: θ values will run sequentially. Start Julia with `-t auto` (see header)."

# ======================================================================================
# Parameters
# ======================================================================================
m_op      = 5        # ordinal-pattern length (d in main.tex)
τ_rpi     = 10       # embedding delay for RPI records / models, in grid steps (= 10 kyr)
α_level   = 0.05
# B_boot    = 1000     # bootstrap replications
B_boot    = 10000     # bootstrap replications
# Resampling scheme of the test (see `test_op_majorization`):
#   :blockbootstrap — moving blocks of the pattern sequence; keeps the dependence between
#                     patterns from overlapping windows (as in NoisyCE's figure5.jl)
#   :multinomial    — patterns treated as i.i.d.
sampling  = :blockbootstrap
block_size(τ) = (m_op - 1) * τ               # span of one ordinal pattern (grid steps)

model_dt     = 0.001   # Ma per grid step (1 kyr) — models and RPIs on this grid
model_burnin = 2.0     # Ma
model_span   = 50.0    # Ma kept after burn-in (5·10⁴ patterns ≫ any record's; the block
                       # bootstrap resamples the model sequence too, so this sets the cost)
min_span     = 0.2     # Ma: minimum RPI record span
n_min        = 5 * factorial(m_op)   # minimum number of patterns in an RPI record
# Gap filter: a grid point is "interpolated across a gap" when it falls strictly inside a raw
# sampling interval longer than gap_factor × τ_rpi (10 kyr); every ordinal pattern that
# uses such a point is dropped.
# Only a gap longer than τ can put two points of one pattern on the same straight
# interpolation segment, which forces a monotone relation that no real record or model
# produces; shorter gaps only replace a value by a local linear estimate. Patterns made of
# real samples on both sides of a gap are kept. (gap_factor = 1 keeps all 22 RPIs; a fixed
# 5 kyr threshold would drop up to 59% of a record and 3 records below n_min.)
gap_factor   = 1.0
gap_max(τ)   = gap_factor * τ * model_dt          # Ma
recompute    = false   # true -> ignore the cache and rerun every θ

data_dir    = joinpath(@__DIR__, "data")
results_dir = joinpath(data_dir, "results")
fig_dir     = joinpath(@__DIR__, "figures")
mkpath(results_dir); mkpath(fig_dir)

# Colours (same dataset colours as test 2)
const COL_PADM2M, COL_SINT = colorant"#2980b9", colorant"#c0392b"
const COL_G12, COL_DL1 = colorant"#27ae60", colorant"goldenrod"

vals(ds) = [v.value for v in ds.value]
n_patterns(x, τ) = length(x) - (m_op - 1) * τ

"Normalized permutation entropy and statistical complexity (H, C) of `x`."
HC(x, τ) = entropy_complexity(StatisticalComplexity(; o = OrdinalPatterns(; m = m_op, τ = τ)), x)

# --- ordinal patterns as codes, so that individual patterns can be dropped --------------
# Everything below works on the sequence of pattern codes (one per window k, k+τ, …,
# k+(m−1)τ) produced by StatsOrdinalPatterns' own encoder, so the test is exactly
# `test_op_majorization` (same resampling and GMS inference, c = 1) but can be fed only the
# patterns that survive the gap filter. With no pattern dropped it reproduces
# `test_op_majorization(x, y, B; m, d = τ, …)` exactly.
op_codes(x, τ) = SOP._op_codes(x, m_op, τ)
code_freqs(c)  = SOP._code_frequencies!(zeros(factorial(m_op)), c)   # all m! probabilities

"(H, C) from a code sequence (same normalization as `HC` on a series)."
function HC_codes(c, τ)
    sc = StatisticalComplexity(; o = OrdinalPatterns(; m = m_op, τ = τ))
    C  = complexity(sc, Probabilities(code_freqs(c)))
    return (sc.entr_val[], C)
end

"Grid points of `t_grid` interpolated across a raw sampling interval longer than `gap` (Ma)."
function in_gap(t_raw, t_grid, gap)
    tr = sort(t_raw)
    return map(t_grid) do t
        j = searchsortedlast(tr, t)
        1 <= j < length(tr) && t > tr[j] + 1e-9 && tr[j + 1] - tr[j] > gap
    end
end
"Mask over patterns: true if none of the m_op points of window k is gap-interpolated."
pattern_mask(gapmask, τ) = [!any(gapmask[k:τ:k + (m_op - 1) * τ]) for k in 1:length(gapmask) - (m_op - 1) * τ]

"Bootstrap p-value of H0: patterns `c_model` majorize patterns `c_obs` (code sequences)."
function majorization_pvalue(c_model::Vector{Int}, c_obs::Vector{Int}, τ)
    n_pat = factorial(m_op)
    p, q  = code_freqs(c_model), code_freqs(c_obs)
    nx, ny = length(c_model), length(c_obs)
    gaps_boot, bs = sampling === :blockbootstrap ?
        (SOP._bootstrap_gaps(c_model, c_obs, n_pat, B_boot, block_size(τ)), block_size(τ)) :
        (SOP._bootstrap_gaps(p, q, nx, ny, B_boot), 1)
    r = SOP._majorization_inference(SOP._lorenz_curve(p), SOP._lorenz_curve(q), gaps_boot,
                                    nx * ny / (nx + ny), α_level, 1.0, bs)
    return r.boot_pval
end

function interp_linear(xs::AbstractVector, ys::AbstractVector, xq::AbstractVector)
    out = similar(xq, Float64)
    for (i, x) in enumerate(xq)
        if x <= xs[1]
            out[i] = ys[1]
        elseif x >= xs[end]
            out[i] = ys[end]
        else
            j = searchsortedlast(xs, x)
            out[i] = ys[j] + (ys[j + 1] - ys[j]) * (x - xs[j]) / (xs[j + 1] - xs[j])
        end
    end
    return out
end

# ======================================================================================
# Data
# ======================================================================================
# RPI records (Ziegler et al. 2011 compilation): signed with the CK95 GPTS, resampled
# onto the 1 kyr grid over the last 2 Ma, short/sparse records dropped.
rpi  = read_rpi(joinpath(data_dir, "Ziegler2011_RPI.csv"); name = :Ziegler2011, min_age = 5.0)
gpts = read_polarity(joinpath(data_dir, "GPTS_CK95.txt"); name = :GPTS_CK95, polarity = :reverse)
t_common      = collect(range(-2.0, 0.0; step = model_dt))
rpi_resampled = [to_signed(resample(ds, t_common), gpts) for ds in rpi.datasets]
span(ds)      = last(ds.time) - first(ds.time)
# patterns kept after the gap filter (raw times of record i vs its 1 kyr grid)
rpi_pmask = [pattern_mask(in_gap(raw.time, rs.time, gap_max(τ_rpi)), τ_rpi)
             for (raw, rs) in zip(rpi.datasets, rpi_resampled)]
valid_idx = findall(i -> span(rpi_resampled[i]) >= min_span && count(rpi_pmask[i]) >= n_min,
                    eachindex(rpi_resampled))
x_rpi     = [vals(ds) for ds in rpi_resampled[valid_idx]]
c_rpi     = [op_codes(vals(rpi_resampled[i]), τ_rpi)[rpi_pmask[i]] for i in valid_idx]
rpi_names = [String(ds.name) for ds in rpi.datasets[valid_idx]]
n_rpi     = length(x_rpi)
println("Using $n_rpi / $(length(rpi_resampled)) RPI records (span ≥ $min_span Ma, ≥ $n_min patterns " *
        "after dropping patterns that use points interpolated across gaps > $(round(1000 * gap_max(τ_rpi), digits = 1)) kyr)")
for (k, i) in enumerate(valid_idx)
    n_drop = length(rpi_pmask[i]) - count(rpi_pmask[i])
    n_drop > 0 && println("  $(rpad(rpi_names[k], 12)) $n_drop / $(length(rpi_pmask[i])) patterns dropped " *
                          "($(round(100n_drop / length(rpi_pmask[i]), digits = 1))%)")
end

# Reference points for the entropy–complexity plane of the main figure (all at τ_rpi on the
# 1 kyr grid)
padm2m   = read_synthesis(joinpath(data_dir, "PADM2Mstar.mat"); name = :PADM2M)
sint2000 = read_synthesis(joinpath(data_dir, "Sint2000.mat");   name = :Sint2000)
HC_padm2m, HC_sint2000 = HC(vals(padm2m), τ_rpi), HC(vals(sint2000), τ_rpi)
HC_g12 = HC(vals(simulate(model = :G12, saveat = model_dt, tspan = (-2.0, 70.0))), τ_rpi)
# ThreeDL1: 3D geodynamo simulation (g10). Its time unit is undocumented; assumed years
# (gives a mean reversal interval of tens of kyr, printed as a sanity check).
dl1_file = joinpath(data_dir, "ThreeDL1.mat")
HC_dl1 = if isfile(dl1_file)
    dl1 = matread(dl1_file)
    t_dl1, g10 = vec(dl1["time"]) .* 1e-6, vec(dl1["g10"])
    n_rev = count(sign.(g10[1:end-1]) .!= sign.(g10[2:end]))
    println("ThreeDL1: mean reversal interval $(round(1000 * (t_dl1[end] - t_dl1[1]) / n_rev, digits = 1)) kyr (time assumed in years)")
    HC(interp_linear(t_dl1, g10, collect(range(t_dl1[1], t_dl1[end]; step = model_dt))), τ_rpi)
else
    @warn "ThreeDL1.mat not found — omitted from the entropy–complexity plane"; nothing
end

# ======================================================================================
# Calibrated (θ, σ) grid
# ======================================================================================
sigma_calibration_file = joinpath(@__DIR__, "..", "data", "sigma_calibration.txt")
isfile(sigma_calibration_file) ||
    error("sigma_calibration.txt not found — run examples/tests/calibrate_sigma_theta.jl first.")
rows = NamedTuple[]
open(sigma_calibration_file) do io
    readline(io)
    for line in eachline(io)
        isempty(line) && continue
        c = split(line, '\t')
        c[6] == "true" && push!(rows, (theta = parse(Float64, c[2]), sigma = parse(Float64, c[3])))
    end
end
sort!(rows; by = r -> r.theta)
n_grid = length(rows)
println("Loaded $n_grid calibrated (θ, σ) points")

# ======================================================================================
# Simulation + RPI tests per θ (cached)
# ======================================================================================
const SEP = "\n---\n"
pad(iθ) = lpad(iθ, ndigits(n_grid), '0')
rpi_file(iθ) = joinpath(results_dir, "theta_$(pad(iθ)).txt")
rpi_stamp(iθ) = join(["theta = $(rows[iθ].theta)", "sigma = $(rows[iθ].sigma)", "m = $m_op",
                      "alpha = $α_level", "B = $B_boot", "sampling = $sampling", "model_dt = $model_dt",
                      "model_burnin = $model_burnin", "model_span = $model_span",
                      "tau_rpi = $τ_rpi", "block_size_rpi = $(block_size(τ_rpi))",
                      "gap_max_kyr = $(1000 * gap_max(τ_rpi))", "rpi_records = $(join(rpi_names, ","))"], "\n")
# Order-insensitive comparison of stamps (lines "tau_alberti = …" from older cache files,
# written when the Alberti test still lived in this script, are ignored).
stamp_lines(s) = sort(filter(l -> !startswith(l, "tau_alberti"), split(s, '\n')))
rpi_ok(iθ) = !recompute && isfile(rpi_file(iθ)) &&
             stamp_lines(first(split(read(rpi_file(iθ), String), SEP))) == stamp_lines(rpi_stamp(iθ))

n_stale = count(!rpi_ok(iθ) for iθ in 1:n_grid)
println(n_stale == 0 ? "All θ cached with matching parameters" :
                       "$n_stale of $n_grid θ missing or stale — (re)computing them")
plog_lock = ReentrantLock()
plog(msg) = lock(() -> println(msg), plog_lock)

Threads.@threads for iθ in 1:n_grid
    rpi_ok(iθ) && continue
    θ, σ = rows[iθ].theta, rows[iθ].sigma
    ds = simulate(; model = :Davis2021, tspan = (-(model_span + model_burnin), 0.0), θ = θ, σ = σ,
                  saveat = model_dt, seed = iθ)
    x = vals(ds)[end - round(Int, model_span / model_dt):end]   # drop burn-in

    Random.seed!(10_000 + iθ)   # task-local RNG used by the bootstrap
    cx    = op_codes(x, τ_rpi)  # model: no gaps, every pattern kept
    p_rpi = [majorization_pvalue(cx, cr, τ_rpi) for cr in c_rpi]
    H, C  = HC(x, τ_rpi)
    body = join(["H\t$H", "C\t$C", "p_rpi\t$(join(p_rpi, ','))",
                 "probs\t$(join(code_freqs(cx), ','))"], "\n")
    write(rpi_file(iθ), rpi_stamp(iθ) * SEP * body)   # stamp first: only complete results
    plog("θ = $(round(1000θ, sigdigits = 3)) kyr: $(count(>=(α_level), p_rpi)) / $n_rpi RPI pass")
end

read_body(file) = Dict(split(l, '\t'; limit = 2) for l in split(last(split(read(file, String), SEP)), '\n'))
results = map(1:n_grid) do iθ
    kv = read_body(rpi_file(iθ))
    (theta = rows[iθ].theta, H = parse(Float64, kv["H"]), C = parse(Float64, kv["C"]),
     p_rpi = parse.(Float64, split(kv["p_rpi"], ',')),
     probs = parse.(Float64, split(kv["probs"], ',')))
end
pct_pass = [100 * count(>=(α_level), r.p_rpi) / n_rpi for r in results]

println("\nθ (kyr)   RPI passing")
for (r, pc) in zip(results, pct_pass)
    println("  $(rpad(round(1000 * r.theta, sigdigits = 3), 8)) $(round(Int, pc * n_rpi / 100))/$n_rpi ($(round(pc, digits = 1))%)")
end

# ======================================================================================
# Shared figure settings — same style as PIML-reversals figure6.jl (boxed frames, no grid,
# CE plane on [0.75, 1] × [0, 0.35], filled circles for reference series, hollow rings for
# the RPI records). Every legend lives in its own panel outside the plots.
# ======================================================================================
theta0_pos  = findfirst(r -> r.theta == 0.0, results)
nonzero_pos = findall(r -> r.theta > 0, results)
θky         = [1000 * results[k].theta for k in nonzero_pos]
cmap_θ      = cgrad(:amp, max(length(nonzero_pos), 2); categorical = true)
θ_colors    = [cmap_θ[i] for i in eachindex(nonzero_pos)]
xlims_θ     = (0.8 * minimum(θky), 1.2 * maximum(θky))
plain_number(x) = x >= 1 ? string(isinteger(x) ? Int(x) : round(x; digits = 2)) :
                           string(round(x; digits = ceil(Int, -log10(x)) + 1))
const STYLE = (guidefontsize = 10, tickfontsize = 8, framestyle = :box, grid = :none,
               legend = false, titlefontsize = 11, left_margin = 10Plots.mm,
               bottom_margin = 10Plots.mm, top_margin = 3Plots.mm, right_margin = 3Plots.mm)
# after STYLE (which turns the grid off): vertical lines at every tick of a log θ axis
const LOG_θ_GRID = (xgrid = true, xminorgrid = true, gridalpha = 0.25, minorgridalpha = 0.12)
CE_XLIMS, CE_YLIMS = (0.75, 1.0), (0.0, 0.35)
c_ec = StatisticalComplexity(; o = OrdinalPatterns(; m = m_op, τ = τ_rpi))   # curves depend on m only
min_raw, max_raw = entropy_complexity_curves(c_ec; num_max = 1, num_min = 1000)
curve(raw) = (o = sortperm([Float64(p[1]) for p in raw]);
              ([Float64(p[1]) for p in raw][o], [Float64(p[2]) for p in raw][o]))

"Empty entropy–complexity plane with the max/min complexity curves."
function ce_plane(title)
    p = plot(curve(max_raw)...; color = :black, lw = 0.9, ls = :dash, xlims = CE_XLIMS,
             ylims = CE_YLIMS, xlabel = "Entropy", ylabel = "Complexity", title = title, STYLE...)
    plot!(p, curve(min_raw)...; color = :black, lw = 0.9, ls = :dash)
    return p
end

"Davis2021 across θ on a CE plane: grey line + θ-coloured dots (labels at decades), θ = 0 black."
function davis_trajectory!(p, res)
    Hθ, Cθ = [res[k].H for k in nonzero_pos], [res[k].C for k in nonzero_pos]
    plot!(p, Hθ, Cθ; color = :gray, lw = 1.5)
    scatter!(p, Hθ, Cθ; markersize = 7, markercolor = θ_colors, markerstrokewidth = 0)
    theta0_pos === nothing ||
        scatter!(p, [res[theta0_pos].H], [res[theta0_pos].C]; markersize = 7, markercolor = :black,
                 markerstrokewidth = 0)
    for q in 10.0 .^ (floor(Int, log10(minimum(θky))):ceil(Int, log10(maximum(θky))))
        i = argmin(abs.(log10.(θky) .- log10(q)))
        annotate!(p, Hθ[i] + 0.005, Cθ[i], text("$(plain_number(θky[i])) kyr", 7, :left, "Computer Modern"))
    end
    return p
end

"Legend panel (outside every plot), square swatches as in figure6.jl."
legend_box() = plot(; framestyle = :none, axis = false, grid = false, ticks = nothing,
                    legend = :left, legendfontsize = 9, foreground_color_legend = nothing)
swatch!(lp, col, lab) = scatter!(lp, [NaN], [NaN]; markershape = :rect, markersize = 8,
                                 markercolor = col, markerstrokewidth = 0, label = lab)

# ======================================================================================
# Figure: main.tex fig:ordinal-pattern
# ======================================================================================
# a) a few RPI records (raw, not resampled), standardized and offset-stacked, last 100 kyr.
#    Time runs to the right; tick labels are ages (kyr).
#    On each record one ordinal pattern is drawn on top: the m_op points spaced τ_rpi apart
#    (values interpolated on the raw record), joined, with its rank permutation
#    (1 = smallest) written above it.
win_kyr, n_show, offset = 100.0, 5, 5.0
τ_kyr          = τ_rpi * model_dt * 1000
# Preferred start (oldest point, kyr) of each record's pattern. It is shifted as little as
# needed to fit inside both the panel window and the record's own coverage (some records
# stop before 0 kyr, e.g. ODP 848-851 ends at 33.5 kyr).
pattern_starts = [-95.0, -75.0, -88.0, -62.0, -80.0]
COL_PATTERN    = colorant"#c0392b"
panel_a = plot(; xlims = (-win_kyr, 0), xformatter = x -> string(round(Int, -x)),
               xlabel = "Age (kyr)", title = "a) Examples of RPIs", STYLE...)
ytk, ylb = Float64[], String[]
for (ds, name) in zip(rpi.datasets[valid_idx], rpi_names)
    length(ytk) >= n_show && break
    m = -win_kyr / 1000 .<= ds.time .<= 0
    count(m) < 2 && continue
    v = vals(ds)[m]
    μ, s, y0 = mean(v), std(v) + 1e-10, length(ytk) * offset
    y = (v .- μ) ./ s .+ y0
    hline!(panel_a, [y0]; color = :lightgray, lw = 0.6)
    plot!(panel_a, 1000 .* ds.time[m], y; color = :gray, lw = 0.8)
    scatter!(panel_a, 1000 .* ds.time[m], y; color = :gray, markersize = 2, markerstrokewidth = 0)
    # ordinal pattern: preferred start, shifted to fit the window and the record's coverage
    # (skipped only if the record is shorter than one pattern inside the window)
    ord_t   = sortperm(ds.time)
    lo_kyr  = max(-win_kyr, 1000 * ds.time[ord_t][1])
    hi_kyr  = min(0.0, 1000 * ds.time[ord_t][end]) - (m_op - 1) * τ_kyr   # latest possible start
    if lo_kyr <= hi_kyr
        t_pat = clamp(pattern_starts[length(ytk) + 1], lo_kyr, hi_kyr) .+ τ_kyr .* (0:(m_op - 1))
        y_pat = (interp_linear(ds.time[ord_t], vals(ds)[ord_t], t_pat ./ 1000) .- μ) ./ s .+ y0
        ranks = invperm(sortperm(y_pat))
        plot!(panel_a, t_pat, y_pat; color = COL_PATTERN, lw = 1.5)
        scatter!(panel_a, t_pat, y_pat; color = COL_PATTERN, markersize = 5, markerstrokewidth = 0)
        annotate!(panel_a, mean(t_pat), maximum(y_pat) + 1.0,
                  text("(" * join(ranks, ",") * ")", 7, COL_PATTERN, :center, "Computer Modern"))
    end
    push!(ytk, y0); push!(ylb, name)
end
plot!(panel_a; yticks = (ytk, ylb), tickfontsize = 6)

# b) % of RPI records passing vs θ: line + θ-coloured dots; θ = 0 as a black dashed line
panel_b = plot(θky, pct_pass[nonzero_pos]; color = :gray, lw = 1.5,
               xscale = :log10, xlims = xlims_θ, xformatter = plain_number, minorticks = true,
               ylims = (0, 100), xlabel = L"\theta\ (\mathrm{kyr})", ylabel = "RPI records passing (%)",
               title = "b) Majorization test", STYLE..., LOG_θ_GRID...)
hline!(panel_b, [50]; color = :lightgray, lw = 0.8)
theta0_pos === nothing ||
    hline!(panel_b, [pct_pass[theta0_pos]]; color = :black, lw = 1.5, ls = :dash)
scatter!(panel_b, θky, pct_pass[nonzero_pos]; markersize = 7, markercolor = θ_colors,
         markerstrokewidth = 0)

# c) complexity–entropy plane at τ_rpi: reference series (filled circles), Davis2021 vs θ,
#    RPI rings (drawn once the layout is final)
panel_c = ce_plane("c) Complexity-Entropy Causality Plane")
references = [(COL_PADM2M, HC_padm2m), (COL_SINT, HC_sint2000), (COL_G12, HC_g12)]
HC_dl1 === nothing || push!(references, (COL_DL1, HC_dl1))
for (col, (H0, C0)) in references
    scatter!(panel_c, [H0], [C0]; markercolor = col, markerstrokewidth = 0, markersize = 12)
end
davis_trajectory!(panel_c, results)
HC_rpi = [HC_codes(c, τ_rpi) for c in c_rpi]   # gap-filtered patterns

# Majorization (Lorenz) curves — removed from the summary figure; kept for reference.
# n_pat = factorial(m_op); ranks = collect(0:n_pat) ./ n_pat
# panel_maj = plot(ranks, ranks; color = :gray, ls = :dash, lw = 0.8, legend = false,
#                  xlims = (0, 1), ylims = (0, 1), xlabel = "Pattern rank",
#                  ylabel = "Accumulated probability", title = "Majorization curves")
# for c in c_rpi
#     plot!(panel_maj, ranks, [0.0; lorenz_curve(code_freqs(c))]; color = :gray, lw = 0.5, alpha = 0.3)
# end
# for (i, k) in enumerate(nonzero_pos)
#     plot!(panel_maj, ranks, [0.0; lorenz_curve(results[k].probs)]; color = θ_colors[i], lw = 1.5)
# end
# theta0_pos === nothing || plot!(panel_maj, ranks, [0.0; lorenz_curve(results[theta0_pos].probs)]; color = :black, lw = 2)

legend_panel = legend_box()
scatter!(legend_panel, [NaN], [NaN]; markershape = :circle, markersize = 7, markercolor = :white,
         markerstrokecolor = :black, markerstrokewidth = 1, label = "RPIs")
plot!(legend_panel, [NaN], [NaN]; color = COL_PATTERN, lw = 1.5, marker = :circle, markersize = 5,
      markerstrokewidth = 0, label = "Ordinal pattern (" * L"m = %$(m_op),\ \tau = %$(round(Int, τ_kyr))" * " kyr)")
swatch!(legend_panel, COL_PADM2M, "PADM2M")
swatch!(legend_panel, COL_SINT, "Sint-2000")
swatch!(legend_panel, COL_G12, "G12")
HC_dl1 === nothing || swatch!(legend_panel, COL_DL1, "ThreeDL1 (g10)")
swatch!(legend_panel, θ_colors[end], "Davis2021 (colour = " * L"\theta" * ")")
swatch!(legend_panel, :black, "Davis2021, " * L"\theta = 0")
plot!(legend_panel, [NaN], [NaN]; color = :black, lw = 1.5, ls = :dash, label = L"\theta = 0" * " (panel b)")

fig = plot(panel_a, panel_b, panel_c, legend_panel;
           layout = @layout([a{0.27w} b{0.28w} c{0.30w} d]), size = (2000, 560))

# RPI rings in the CE plane. GR has no hollow markers (marker alpha hides the stroke too),
# so each RPI is drawn as a closed line; the radius is set in mm of the final axis box so
# the rings stay circular (same construction as figure6.jl's draw_rpi_rings!).
function draw_rings!(fig, sp, points, xl, yl; r = 1.5, color = :black, linewidth = 1.0)
    Plots.prepare_output(fig)
    bb = Plots.plotarea(fig[sp])
    w, h = Plots.width(bb).value, Plots.height(bb).value
    φ  = range(0, 2π; length = 60)
    rx, ry = r / w * (xl[2] - xl[1]), r / h * (yl[2] - yl[1])
    for (x, y) in points
        plot!(fig, x .+ rx .* cos.(φ), y .+ ry .* sin.(φ); subplot = sp, color = color,
              linewidth = linewidth, label = "")
    end
    return fig
end
draw_rings!(fig, 3, HC_rpi, CE_XLIMS, CE_YLIMS)
savefig(fig, joinpath(fig_dir, "summary.pdf"))

println("\nFigure saved to $(joinpath(fig_dir, "summary.pdf"))")
