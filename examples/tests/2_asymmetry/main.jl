# Asymmetry across reversals — the R_i statistic exactly as defined in main.tex
# (Section "Test No2: Asymmetry across reversals").
#
# PARALLEL EXECUTION — the θ loop is multithreaded (one independent simulation per θ,
# `Threads.@threads`), so Julia must be STARTED with several threads; it cannot be
# changed from inside a running session. From the repo root:
#
#   julia -t auto --project=examples examples/tests/2_asymmetry/main.jl     # all cores
#   julia -t 8    --project=examples examples/tests/2_asymmetry/main.jl     # 8 threads
#
# or `export JULIA_NUM_THREADS=auto` before launching. In the VS Code REPL, set
# "julia.NumThreads": "auto" in settings.json and restart the REPL. Check with
# `Threads.nthreads()` (the script prints it and warns if it is 1). The θ = 0 run (fixed
# small-step EM solver) is the slowest one and sets the total wall time.
#
# Definitions (h0 = hysteresis threshold, h = asymmetry threshold, both in units of x0):
#   T_i              : hysteresis (Gammaitoni) crossings at ±h0 — chron boundaries.
#   For the transition at T_{i+1}, between chron [T_i, T_{i+1}] and [T_{i+1}, T_{i+2}],
#   with y = s·X (s = polarity of the old chron, so y > 0 before, y < 0 after):
#   T↓   = sup{t ∈ [T_i, T_{i+1}]     : y_t >  h}      (last time above h in old chron)
#   T↑   = inf{t ∈ [T_{i+1}, T_{i+2}] : y_t < -h}      (first time below -h in new chron)
#   T0−  = inf{t > T↓ : y_t = 0},  T0+ = sup{t < T↑ : y_t = 0}
#   δ↓   = T0− − T↓,   δ↑ = T↑ − T0+,   R = (δ↓ − δ↑)/(δ↓ + δ↑)
# An event is discarded (R = NaN) if either T↓ or T↑ does not exist. The rule is symmetric
# under time reversal + sign flip, so E[R] = 0 exactly at θ = 0 (checked below).
#
# Everything is computed on the FULL simulated trajectory (no ±window cutoff in the
# search), so no window can truncate one leg more than the other.
#
# Aligned mean curves (panels b–e) are aligned on the MIDPOINT of the reversal interval,
# T0 = (T0− + T0+)/2. Time reversal + sign flip maps T0− ↔ T0+, so T0 is the only choice
# that keeps the θ = 0 curve symmetric (aligning on T0− adds the whole hovering interval
# T0+ − T0− to the recovery side). For the data T0− = T0+, so T0 = T0−.
#
# Outputs (simulation cache is separate from sweep_theta_reversal_windows.jl's; NOTE: the
# old analyze_theta_reversal_windows.jl writes asymmetry_summary.pdf and
# reversal_data_closeup.pdf to the same figures/ folder and would overwrite these):
#   data/simulations_R/   theta_XX_events.txt, theta_XX_curve.txt, theta_XX_params.txt
#                         (per-θ parameter stamp: a θ is recomputed automatically if any
#                         parameter, e.g. h or h0, changed), master.txt, params.txt
#   figures/asymmetry_summary.pdf
#   figures/reversal_data_closeup.pdf
#   figures/delta_distribution.pdf
#   figures/sde_times_check.pdf    (all crossing times drawn on a few short SDE runs)

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GeoReversal
ENV["GKSwstype"] = "nul"   # headless GR: figures are only saved to disk, never shown
using Plots
using LaTeXStrings
using Statistics: mean, std, median
using Random

default(fontfamily = "Computer Modern")
# Per-panel margins for the summary figure. They must be set on each panel: margins passed
# to the outer `plot(...)` that assembles the layout override every subplot's margins.
const PANEL_MARGINS = (left_margin = 12Plots.mm, bottom_margin = 11Plots.mm,
                       top_margin = 3Plots.mm, right_margin = 3Plots.mm)

println("Running with $(Threads.nthreads()) thread(s)")
Threads.nthreads() == 1 &&
    @warn "Single thread: θ values will run sequentially. Start Julia with `-t auto` (see header)."

# ======================================================================================
# Parameters
# ======================================================================================
h0          = 0.5      # hysteresis threshold for T_i (x0 = 1 in both model and data)
h_thr       = 0.8      # asymmetry threshold h (main.tex: h = 0.8 x0)
n_reversals = 5000     # transitions per θ
burnin      = 1.0      # Ma
saveat      = 0.0005   # Ma = 0.5 kyr
sim_age0    = 10.0     # Ma; doubled until enough transitions are found
recompute   = false    # true -> ignore cached simulation results and rerun everything

# Window (kyr) of the aligned mean-trajectory curves — for plotting ONLY; the R_i
# search itself never uses a window.
curve_left, curve_right = 100.0, 60.0
curve_kyr = collect(range(-curve_left, curve_right; length = 321))
# Window (kyr) actually SHOWN in the aligned panels (b–e). Kept separate from the cached
# curve window above so that changing it does not invalidate the simulation cache.
plot_window = 40.0
plot_xlims  = (-plot_window, plot_window)

sim_dir = joinpath(@__DIR__, "data", "simulations_R")
fig_dir = joinpath(@__DIR__, "figures")
mkpath(sim_dir); mkpath(fig_dir)

# ======================================================================================
# Core: crossing times and R for one transition
# ======================================================================================

"""
    hysteresis_crossings(t, x, h0) -> Vector{Float64}

Chron boundaries T_i of a sampled signed series: polarity switches to normal when x rises
through +h0 and to reverse when it falls through −h0 (linear interpolation between
samples). Same rule as the simulator's `:Gammaitoni1998` criterion.
"""
function hysteresis_crossings(t::AbstractVector, x::AbstractVector, h0::Real)
    T = Float64[]
    state = x[1] >= 0 ? :normal : :reverse
    for i in 2:length(t)
        if state === :reverse && x[i - 1] < h0 && x[i] >= h0
            push!(T, t[i - 1] + (h0 - x[i - 1]) / (x[i] - x[i - 1]) * (t[i] - t[i - 1]))
            state = :normal
        elseif state === :normal && x[i - 1] > -h0 && x[i] <= -h0
            push!(T, t[i - 1] + (-h0 - x[i - 1]) / (x[i] - x[i - 1]) * (t[i] - t[i - 1]))
            state = :reverse
        end
    end
    return T
end

# time at which the segment (t1,y1)-(t2,y2) crosses level c
_cross(t1, y1, t2, y2, c) = t1 + (y1 - c) / (y1 - y2) * (t2 - t1)

const NAN_EVENT = (s = 0, TD = NaN, T0m = NaN, T0p = NaN, TU = NaN,
                   δD = NaN, δU = NaN, R = NaN, nzc = 0)

"""
    reversal_event(t, x, Tprev, Tmid, Tnext, h) -> NamedTuple

Crossing times (main.tex definitions) for the transition at `Tmid` = T_{i+1}, between
chrons [Tprev, Tmid] and [Tmid, Tnext]. `t` sorted ascending (Ma). Threshold and zero
crossings are linearly interpolated between samples. If T↓ or T↑ does not exist the event
is discarded (R = NaN), but whichever of T↓/T↑ DOES exist is still returned, so the
reason for discarding can be read off the output (TD = NaN: the old chron never rose above
+h after T_i; TU = NaN: the new chron never fell below −h before T_{i+2}).
"""
function reversal_event(t::AbstractVector, x::AbstractVector,
                        Tprev::Real, Tmid::Real, Tnext::Real, h::Real)
    i_prev = searchsortedfirst(t, Tprev)
    i_mid  = searchsortedlast(t, Tmid)
    i_next = searchsortedlast(t, Tnext)
    (i_prev < i_mid < i_next && i_next <= length(t)) || return NAN_EVENT

    # At Tmid the path has just reached ∓h0 of the NEW polarity -> old polarity s:
    s = x[min(i_mid + 1, length(t))] < 0 ? 1 : -1
    y(i) = s * x[i]

    jD = findlast(i -> y(i) > h, i_prev:i_mid)
    jU = findfirst(i -> y(i) < -h, (i_mid + 1):i_next)
    iD = jD === nothing ? 0 : (i_prev:i_mid)[jD]
    iU = jU === nothing ? 0 : ((i_mid + 1):i_next)[jU]
    TD = iD == 0 ? NaN : _cross(t[iD], y(iD), t[iD + 1], y(iD + 1), h)     # y: > h -> ≤ h
    TU = iU == 0 ? NaN : _cross(t[iU - 1], y(iU - 1), t[iU], y(iU), -h)    # y: ≥ -h -> < -h
    (iD == 0 || iU == 0) && return merge(NAN_EVENT, (s = s, TD = TD, TU = TU))

    zc = Float64[]
    for j in iD:(iU - 1)
        (y(j) > 0) != (y(j + 1) > 0) && push!(zc, _cross(t[j], y(j), t[j + 1], y(j + 1), 0.0))
    end
    T0m, T0p = first(zc), last(zc)    # nonempty: y(iD) > h > 0 > -h > y(iU)

    δD, δU = T0m - TD, TU - T0p
    R = (δD - δU) / (δD + δU)
    return (s = s, TD = TD, T0m = T0m, T0p = T0p, TU = TU, δD = δD, δU = δU, R = R,
            nzc = length(zc))
end

"""
    aligned_curve(t, x, T0, grid_kyr) -> Vector{Float64}

|x| interpolated onto `T0 + grid_kyr/1000` (NaN outside the record).
"""
function aligned_curve(t::AbstractVector, x::AbstractVector, T0::Real, grid_kyr::AbstractVector)
    out = fill(NaN, length(grid_kyr))
    for (k, g) in enumerate(grid_kyr)
        tq = T0 + g / 1000
        j = searchsortedlast(t, tq)
        (1 <= j < length(t)) || continue
        w = (tq - t[j]) / (t[j + 1] - t[j])
        out[k] = (1 - w) * abs(x[j]) + w * abs(x[j + 1])
    end
    return out
end

nanmean(v) = (u = filter(isfinite, v); isempty(u) ? NaN : mean(u))
nanstd(v)  = (u = filter(isfinite, v); length(u) < 2 ? NaN : std(u))

# ======================================================================================
# Simulations: one long Davis2021 realization per calibrated θ (cached)
# ======================================================================================
sigma_calibration_file = joinpath(@__DIR__, "..", "data", "sigma_calibration.txt")
isfile(sigma_calibration_file) ||
    error("sigma_calibration.txt not found at $sigma_calibration_file — run " *
          "examples/tests/calibrate_sigma_theta.jl first.")

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
ndig   = ndigits(n_grid)
events_file(iθ) = joinpath(sim_dir, "theta_$(lpad(iθ, ndig, '0'))_events.txt")
curve_file(iθ)  = joinpath(sim_dir, "theta_$(lpad(iθ, ndig, '0'))_curve.txt")
println("Loaded $n_grid calibrated (θ, σ) points")

progress_lock = ReentrantLock()
plog(msg) = lock(() -> println(msg), progress_lock)

# Cache validity: every θ's results carry a stamp (theta_XX_params.txt) of ALL parameters
# that affect them. A θ is reused only if its stamp matches the current settings exactly;
# otherwise (missing stamp, changed h/h0/saveat/…, changed calibrated σ) it is recomputed.
# This prevents silently mixing data analysed at one threshold with cached simulations
# computed at another.
stamp_file(iθ) = joinpath(sim_dir, "theta_$(lpad(iθ, ndig, '0'))_params.txt")
param_stamp(iθ) = join(["theta = $(rows[iθ].theta)", "sigma = $(rows[iθ].sigma)", "h0 = $h0",
                        "h = $h_thr", "n_reversals = $n_reversals", "burnin = $burnin",
                        "saveat = $saveat", "sim_age0 = $sim_age0",
                        "curve_window_kyr = $((curve_left, curve_right))",
                        "curve_npts = $(length(curve_kyr))", "curve_align = midpoint"], "\n")
cache_ok(iθ) = !recompute && isfile(events_file(iθ)) && isfile(curve_file(iθ)) &&
               isfile(stamp_file(iθ)) && read(stamp_file(iθ), String) == param_stamp(iθ)
n_stale = count(!cache_ok(iθ) for iθ in eachindex(rows))
println(n_stale == 0 ? "All $n_grid θ cached with matching parameters — no simulation needed" :
                       "$n_stale of $n_grid θ missing or computed with different parameters — (re)computing them")

Threads.@threads for iθ in eachindex(rows)
    cache_ok(iθ) && continue
    θ, σ = rows[iθ].theta, rows[iθ].sigma

    age = sim_age0
    local t, x, Ts
    for attempt in 1:10
        maxiters = θ == 0.0 ? 100_000 : ceil(Int, 300 * (age + burnin) / θ)
        ds = simulate(; model = :Davis2021, tspan = (-(age + burnin), 0.0), θ = θ, σ = σ,
                      saveat = saveat, seed = iθ, detect_reversals = true,
                      criterion = :Gammaitoni1998, h0 = h0, maxiters = maxiters)
        t  = collect(Float64, ds.time)
        x  = [v.value for v in ds.value]
        Ts = ds.reversals
        # transitions Tmid = Ts[k] need Ts[k-1] >= -age (post burn-in) and Ts[k+1] to exist
        count(>=(-age), Ts) >= n_reversals + 2 && break
        age *= 2
    end

    k0 = findfirst(>=(-age), Ts)
    ks = (k0 + 1):min(k0 + n_reversals, length(Ts) - 1)

    sum_c  = zeros(length(curve_kyr)); sum_c2 = zeros(length(curve_kyr))
    n_c    = zeros(Int, length(curve_kyr))
    open(events_file(iθ), "w") do io
        println(io, "k\tTprev\tTmid\tTnext\ts\tTD\tT0m\tT0p\tTU\tdeltaD\tdeltaU\tR\tnzc")
        for k in ks
            ev = reversal_event(t, x, Ts[k - 1], Ts[k], Ts[k + 1], h_thr)
            println(io, join((k, Ts[k - 1], Ts[k], Ts[k + 1], ev.s, ev.TD, ev.T0m, ev.T0p,
                              ev.TU, ev.δD, ev.δU, ev.R, ev.nzc), '\t'))
            isfinite(ev.R) || continue
            c = aligned_curve(t, x, (ev.T0m + ev.T0p) / 2, curve_kyr)
            for j in eachindex(c)
                isfinite(c[j]) || continue
                sum_c[j] += c[j]; sum_c2[j] += c[j]^2; n_c[j] += 1
            end
        end
    end
    open(curve_file(iθ), "w") do io
        println(io, "RelTimeKyr\tMeanAbsX\tStdAbsX\tN")
        for j in eachindex(curve_kyr)
            m = sum_c[j] / n_c[j]
            sd = sqrt(max(sum_c2[j] / n_c[j] - m^2, 0.0) * n_c[j] / max(n_c[j] - 1, 1))
            println(io, "$(curve_kyr[j])\t$m\t$sd\t$(n_c[j])")
        end
    end
    write(stamp_file(iθ), param_stamp(iθ))   # written last: only a complete result is stamped
    plog("θ=$(round(θ, sigdigits = 4)): age=$age Ma, $(length(ks)) transitions written")
end

open(joinpath(sim_dir, "master.txt"), "w") do io
    println(io, "Index\tEventsFile\tCurveFile\tTheta\tSigma")
    for (iθ, r) in enumerate(rows)
        println(io, "$iθ\t$(basename(events_file(iθ)))\t$(basename(curve_file(iθ)))\t$(r.theta)\t$(r.sigma)")
    end
end
# Human-readable summary only — the authoritative per-θ record is theta_XX_params.txt.
open(joinpath(sim_dir, "params.txt"), "w") do io
    for (k, v) in (("h0", h0), ("h", h_thr), ("n_reversals", n_reversals), ("burnin", burnin),
                   ("saveat", saveat), ("sim_age0", sim_age0), ("curve_window_kyr", (curve_left, curve_right)))
        println(io, "$k = $v")
    end
end
all(cache_ok, eachindex(rows)) || error("Some θ results still do not match the current parameters")

# ======================================================================================
# Read simulation results back
# ======================================================================================
function read_events(path)
    R = Float64[]; δD = Float64[]; δU = Float64[]; nzc = Int[]; n_all = 0
    # Start of decline / end of recovery in the frame aligned on T0 = (T0− + T0+)/2:
    #   dn = T↓ − T0 = −(δ↓ + (T0+ − T0−)/2),   up = T↑ − T0 = δ↑ + (T0+ − T0−)/2
    dn = Float64[]; up = Float64[]
    n_noTD = 0; n_noTU = 0; n_none = 0
    open(path) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            c = split(line, '\t'); n_all += 1
            r = parse(Float64, c[12])
            if !isfinite(r)
                noD, noU = isnan(parse(Float64, c[6])), isnan(parse(Float64, c[9]))
                n_noTD += noD && !noU; n_noTU += noU && !noD; n_none += noD && noU
                continue
            end
            push!(R, r); push!(δD, parse(Float64, c[10])); push!(δU, parse(Float64, c[11]))
            push!(nzc, parse(Int, c[13]))
            mid = (parse(Float64, c[7]) + parse(Float64, c[8])) / 2
            push!(dn, parse(Float64, c[6]) - mid); push!(up, parse(Float64, c[9]) - mid)
        end
    end
    return (R = R, δD = 1000 .* δD, δU = 1000 .* δU, dn = 1000 .* dn, up = 1000 .* up,
            nzc = nzc, n_all = n_all,
            n_noTD = n_noTD, n_noTU = n_noTU, n_none = n_none)
end
function read_curve(path)
    m = Float64[]; s = Float64[]
    open(path) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            c = split(line, '\t'); push!(m, parse(Float64, c[2])); push!(s, parse(Float64, c[3]))
        end
    end
    return m, s
end

theta_stats = map(eachindex(rows)) do iθ
    ev = read_events(events_file(iθ))
    m, s = read_curve(curve_file(iθ))
    n = length(ev.R)
    (theta = rows[iθ].theta, R = ev.R, δD = ev.δD, δU = ev.δU,
     R_mean = mean(ev.R), R_std = std(ev.R), R_sem = std(ev.R) / sqrt(n), n = n,
     n_all = ev.n_all, n_noTD = ev.n_noTD, n_noTU = ev.n_noTU, n_none = ev.n_none,
     nzc_median = median(ev.nzc),
     δD_mean = mean(ev.δD), δU_mean = mean(ev.δU), dn_mean = mean(ev.dn), up_mean = mean(ev.up),
     mean_curve = m, std_curve = s)
end

println("\nSimulated R per θ (δ in kyr):")
for s in theta_stats
    println("  θ=$(rpad(round(1000 * s.theta, sigdigits = 3), 7)) ky  n=$(s.n)/$(s.n_all)  " *
            "R=$(round(s.R_mean, digits = 3)) ± $(round(s.R_sem, digits = 3)) (sem)  " *
            "δ↓=$(round(s.δD_mean, digits = 1))  δ↑=$(round(s.δU_mean, digits = 1))  " *
            "median #zero-crossings=$(s.nzc_median)  " *
            "discarded: no T↓=$(s.n_noTD), no T↑=$(s.n_noTU), neither=$(s.n_none)")
end
println("(Files written before the discard-reason fix store TD=TU=NaN for every discarded " *
        "event, so they show up as 'neither' — rerun with recompute=true to split them.)")
s0 = theta_stats[findfirst(s -> s.theta == 0.0, theta_stats)]
z0 = s0.R_mean / s0.R_sem
println("Sanity check θ=0: mean R / sem = $(round(z0, digits = 2)) " *
        (abs(z0) < 3 ? "(consistent with E[R]=0)" : "(WARNING: inconsistent with E[R]=0)"))

# ======================================================================================
# Real data: PADM2M / Sint-2000, same pipeline
# ======================================================================================
data_dir = joinpath(@__DIR__, "data")
padm2m   = read_synthesis(joinpath(data_dir, "PADM2Mstar.mat"); name = :PADM2M)
sint2000 = read_synthesis(joinpath(data_dir, "Sint2000.mat");   name = :Sint2000)

# CK95 ages, used ONLY to label the detected transitions (and to drop anything else).
known_reversals = [
    ("Brunhes-Matuyama", 0.780, :steelblue),
    ("Upper Jaramillo",  0.990, :darkorange),
    ("Lower Jaramillo",  1.070, :seagreen),
    ("Upper Olduvai",    1.770, :firebrick),
    ("Lower Olduvai",    1.950, :purple),
]
match_tol = 0.030   # Ma

# One colour (and line style) per reconstruction, used consistently in the summary figure:
# panel a, the per-reversal panels, the data R bands in e and the p-value curves in f.
real_datasets = [(name = "PADM2M",    ds = padm2m,   ls = :dash, color = colorant"#2980b9"),
                 (name = "Sint-2000", ds = sint2000, ls = :dot,  color = colorant"#c0392b")]

real_events = NamedTuple[]
println("\nReal data (times in kyr relative to T_{i+1}):")
for d in real_datasets
    o = sortperm(d.ds.time)
    t = collect(Float64, d.ds.time[o])
    x = [v.value for v in d.ds.value[o]]
    Ts = hysteresis_crossings(t, x, h0)
    # Record ends act as the outer chron boundaries (main.tex: T_1 = start of record)
    edges = [t[1]; Ts; t[end]]
    for k in 2:(length(edges) - 1)
        Tmid = edges[k]
        jm = findfirst(r -> abs(-r[2] - Tmid) <= match_tol, known_reversals)
        if jm === nothing
            println("  $(d.name): transition at $(round(-Tmid, digits = 3)) Ma not in list — skipped")
            continue
        end
        ev = reversal_event(t, x, edges[k - 1], Tmid, edges[k + 1], h_thr)
        rname, age, color = known_reversals[jm]
        # color = reversal colour (diagnostic close-up figure); dcolor = dataset colour (summary)
        push!(real_events, (dataset = d.name, ls = d.ls, name = rname, color = color, dcolor = d.color,
                            Tprev = edges[k - 1], Tmid = Tmid, Tnext = edges[k + 1],
                            Ts = Ts, t = t, x = x, ev = ev))
        f(v) = isnan(v) ? "NaN" : string(round(1000 * (v - Tmid), digits = 1))
        println("  $(rpad(d.name, 9)) $(rpad(rname, 17)) T_{i+1}=$(round(-Tmid, digits = 3)) Ma  " *
                "T↓=$(f(ev.TD)) T0−=$(f(ev.T0m)) T0+=$(f(ev.T0p)) T↑=$(f(ev.TU))  " *
                "δ↓=$(round(1000 * ev.δD, digits = 1)) δ↑=$(round(1000 * ev.δU, digits = 1)) " *
                "R=$(round(ev.R, digits = 3))")
    end
end
# Test of means, one per record (n = 5 reversals each). PADM2M and Sint-2000 are tested
# separately — no averaging across records, since they observe the same 5 reversals.
# H0: the 5 observed R_i are draws from the model's distribution of R at this θ.
# Statistic: the sample mean R̄_obs of the 5 values. Its null distribution is obtained by
# Monte Carlo — the mean of 5 events drawn (with replacement) from the ~5000 simulated R
# at that θ, repeated n_mc times — so no normality assumption is needed (R is bounded and
# skewed). Two-sided p-value, with the +1 correction so p ≥ 1/(n_mc+1).
n_mc = 200_000
function mc_mean_test(obs::AbstractVector, sim::AbstractVector; n_mc = n_mc, seed = 1)
    rng = Random.Xoshiro(seed)
    n, m_obs = length(obs), mean(obs)
    n_ge = 0; n_le = 0
    for _ in 1:n_mc
        m = 0.0
        for _ in 1:n
            m += sim[rand(rng, 1:length(sim))]
        end
        m /= n
        n_ge += m >= m_obs; n_le += m <= m_obs
    end
    return min(1.0, 2 * (min(n_ge, n_le) + 1) / (n_mc + 1))
end

real_R_by_ds = Dict(d.name => [e.ev.R for e in real_events if e.dataset == d.name && isfinite(e.ev.R)]
                    for d in real_datasets)
pvals_by_ds = Dict(d.name => [mc_mean_test(real_R_by_ds[d.name], s.R) for s in theta_stats]
                   for d in real_datasets)

println("\nTest of means per record (Monte Carlo, n=5, two-sided):")
for d in real_datasets
    r = real_R_by_ds[d.name]
    println("  $(d.name): n=$(length(r)), R = $(round(mean(r), digits = 3)) ± " *
            "$(round(std(r), digits = 3)) (std)")
end
for (k, s) in enumerate(theta_stats)
    println("  θ=$(rpad(round(1000 * s.theta, sigdigits = 3), 7)) ky  " *
            join(["p($(d.name))=$(round(pvals_by_ds[d.name][k], sigdigits = 3))" for d in real_datasets], "  "))
end

# ======================================================================================
# Figure 1: summary
# ======================================================================================
nonzero_pos = findall(s -> s.theta > 0, theta_stats)
cmap_theta  = cgrad(:amp, max(length(nonzero_pos), 2); categorical = true)
color_of    = Dict{Int, Any}(k => cmap_theta[i] for (i, k) in enumerate(nonzero_pos))
theta0_pos  = findfirst(s -> s.theta == 0.0, theta_stats)
color_of[theta0_pos] = :black

# Mean decline / recovery times in the frame aligned on T0 = (T0− + T0+)/2 (vertical lines
# in panels b–e): start of decline at mean(T↓ − T0), end of recovery at mean(T↑ − T0).
# For the data T0 = T0− = T0+, so these are exactly −mean(δ↓) and +mean(δ↑).
# Dark grey (not blue/orange) so they cannot be confused with the PADM2M/Sint-2000 colours.
# Vertical-line convention used in every panel of the summary figure:
#   decline start / recovery end (T↓, T↑ or their means) : black dashed
#   T0                                                     : black solid
const DECLINE_COLOR, RECOVERY_COLOR = :black, :black
const DECLINE_LS, RECOVERY_LS = :dash, :dash
function decay_recovery_lines!(p, dn_mean, up_mean)
    vline!(p, [dn_mean]; color = DECLINE_COLOR, lw = 1.8, ls = DECLINE_LS, label = false)
    vline!(p, [up_mean]; color = RECOVERY_COLOR, lw = 1.8, ls = RECOVERY_LS, label = false)
    return p
end
# EXPLORATORY: double-headed arrows near the top of panels a–e marking the decay region
# [T↓, T0] (purple) and the recovery region [T0, T↑] (green). Set to false to remove them.
show_region_arrows = true
const DECAY_ARROW_COLOR, RECOVERY_ARROW_COLOR = colorant"#8e44ad", colorant"#27ae60"
const ALIGNED_YLIMS, ALIGNED_ARROW_Y = (0.0, 2.2), 2.08   # panels b–e: headroom for the arrows
# Single closed head at the RIGHT end (later time) of each arrow. The head size is NOT set
# by Plots.arrow (GR ignores its length/width); it is the global GR.setarrowsize, applied
# right before the summary figure is saved (REGION_ARROW_SIZE), so it is identical everywhere.
const REGION_ARROW_SIZE = 0.45   # GR default is 1
function region_arrows!(p, x_dn, x0, x_up, y)
    show_region_arrows || return p
    a = Plots.arrow(:closed, :head)
    plot!(p, [x_dn, x0], [y, y]; color = DECAY_ARROW_COLOR, lw = 2, arrow = a, label = false)
    plot!(p, [x0, x_up], [y, y]; color = RECOVERY_ARROW_COLOR, lw = 2, arrow = a, label = false)
    return p
end
midpoint(ev) = (ev.T0m + ev.T0p) / 2
const XLAB_ALIGNED = "Relative age (kyr)"

# a) data, aligned on T0 (= T0− = T0+ for the data)
snapshot_panel = plot(; xlabel = XLAB_ALIGNED,
                       title = "b) Reversals (data), aligned on " * L"T^{0}", titlefontsize = 10,
                       xlims = plot_xlims, ylims = ALIGNED_YLIMS, legend = false, PANEL_MARGINS...)
vline!(snapshot_panel, [0.0]; color = :black, lw = 1.0)   # T0
hline!(snapshot_panel, [h_thr]; color = :gray, lw = 0.5, ls = :dash)
data_curves = Vector{Vector{Float64}}()
for e in real_events
    isfinite(e.ev.R) || continue
    c = aligned_curve(e.t, e.x, midpoint(e.ev), curve_kyr)
    push!(data_curves, c)
    plot!(snapshot_panel, curve_kyr, c; color = e.dcolor, ls = :solid, lw = 1.2, alpha = 0.85)
end
data_mean = [nanmean([c[j] for c in data_curves]) for j in eachindex(curve_kyr)]
data_std  = [nanstd([c[j] for c in data_curves])  for j in eachindex(curve_kyr)]
plot!(snapshot_panel, curve_kyr, data_mean; ribbon = data_std, color = :black, lw = 3,
      fillalpha = 0.2, fillcolor = :black)
# pooled over the same events as the black mean curve (both records)
data_kept     = [e for e in real_events if isfinite(e.ev.R)]
data_dn_mean  = mean(1000 * (e.ev.TD - midpoint(e.ev)) for e in data_kept)
data_up_mean  = mean(1000 * (e.ev.TU - midpoint(e.ev)) for e in data_kept)
decay_recovery_lines!(snapshot_panel, data_dn_mean, data_up_mean)
region_arrows!(snapshot_panel, data_dn_mean, 0.0, data_up_mean, ALIGNED_ARROW_Y)
println("Data (pooled, n=$(length(data_kept))): mean T↓−T0 = $(round(data_dn_mean, digits = 1)) kyr, " *
        "mean T↑−T0 = $(round(data_up_mean, digits = 1)) kyr")
for s in theta_stats
    println("  θ=$(rpad(round(1000 * s.theta, sigdigits = 3), 7)) ky  mean T↓−T0 = " *
            "$(round(s.dn_mean, digits = 1)) kyr, mean T↑−T0 = $(round(s.up_mean, digits = 1)) kyr")
end

# Panel a: one small panel per observed reversal (top row): signed X_t vs age (kyr, older
# to the left), both reconstructions in their dataset colour, vertical lines at T↓, T0 and
# T↑ of PADM2M, and each record's R_i.
# The x axis is TIME (1000·t, kyr, increasing to the right = forward in time) with tick
# labels printed as positive ages. (Not `xflip` on the age: GR misplaces arrow heads on a
# flipped axis.)
# The five panels are drawn as one strip: no axis lines (framestyle = :grid, grid off),
# thick black vertical spines at each panel edge as separators, a black line at y = 0,
# x tick labels only, y tick labels on the first panel only and no y label.
rev_ylims  = (-1.5, 1.5)
nrev_panel = length(known_reversals)
# Inner edges: REV_SEAM on each side of every inner edge (on top of GR's own fixed padding),
# so neighbouring reversals are clearly separated.
# The first panel's left margin is 3.5 mm smaller than PANEL_MARGINS because its "-1.0"
# tick label is wider than the labels of rows 2–3; this lines the plot areas up.
const REV_SEAM = 3Plots.mm
rev_margins(j) = (left_margin  = j == 1 ? PANEL_MARGINS.left_margin - 3.5Plots.mm : REV_SEAM,
                  right_margin = j == nrev_panel ? PANEL_MARGINS.right_margin : REV_SEAM,
                  bottom_margin = PANEL_MARGINS.bottom_margin, top_margin = PANEL_MARGINS.top_margin)
age_label(x) = string(round(Int, -x))   # time 1000·t (kyr, negative) -> age (kyr, positive)
# Oldest reversal on the left, most recent (Brunhes-Matuyama) on the right, following time.
reversal_panels = map(enumerate(sort(known_reversals; by = r -> -r[2]))) do (j, (rname, _, _))
    evs = [e for e in real_events if e.name == rname]
    p = plot(; ylims = rev_ylims, legend = false, title = rname, titlefontsize = 9,
             framestyle = :grid, grid = false, tickfontsize = 7, xformatter = age_label,
             yticks = j == 1 ? ([-1, 0, 1], ["-1.0", "0.0", "1.0"]) : false,   # same label width as rows 2–3
             xlabel = j == cld(nrev_panel, 2) ? "Age (kyr)" : "", rev_margins(j)...)
    isempty(evs) && return plot!(p; title = "$rname: not detected")
    # window CENTRED on T0 (mean of the two records' T0; T_{i+1} if an event was discarded),
    # half-width = farthest T↓/T↑ of either record + 15 kyr padding (50 kyr if none exist)
    Tc = mean(isfinite(e.ev.R) ? midpoint(e.ev) : e.Tmid for e in evs)
    Ts_lr = filter(isfinite, vcat([e.ev.TD for e in evs], [e.ev.TU for e in evs]))
    halfw = (isempty(Ts_lr) ? 0.050 : maximum(abs.(Ts_lr .- Tc))) + 0.015
    lo, hi = Tc - halfw, Tc + halfw          # Ma (negative); lo = older = left edge
    span = 1000 * (hi - lo)
    plot!(p; xlims = (1000lo, 1000hi))
    # The spines sit exactly on the plot-area edges and GR clips everything outside the plot
    # area, so only the inner HALF of the width is drawn: lw = 6 shows as ~3.
    vline!(p, [1000lo, 1000hi]; color = :black, lw = 6)
    hline!(p, [0.0]; color = :black, lw = 0.8)
    hline!(p, [-h_thr, h_thr]; color = :gray, lw = 0.5, ls = :dash)
    for e in evs
        m = (e.t .>= lo) .& (e.t .<= hi)
        # signal: thick solid line in the dataset colour
        plot!(p, 1000 .* e.t[m], e.x[m]; color = e.dcolor, ls = :solid, lw = 3.5)
        # vertical lines (and region arrows) for PADM2M only (Sint-2000 still reported
        # through its R value): T↓/T↑ black dashed, T0 black solid
        e.dataset == "PADM2M" || continue
        isnan(e.ev.TD) || vline!(p, [1000 * e.ev.TD]; color = DECLINE_COLOR, ls = DECLINE_LS, lw = 1.5)
        isnan(e.ev.TU) || vline!(p, [1000 * e.ev.TU]; color = RECOVERY_COLOR, ls = RECOVERY_LS, lw = 1.5)
        isfinite(e.ev.R) || continue
        vline!(p, [1000 * midpoint(e.ev)]; color = :black, ls = :solid, lw = 1.0)
        region_arrows!(p, 1000 * e.ev.TD, 1000 * midpoint(e.ev), 1000 * e.ev.TU, 1.35)
    end
    # R_i of both records stacked in the top-left corner, under the region arrows
    for (d, ypos) in zip(real_datasets, (1.1, 0.87))
        i = findfirst(e -> e.dataset == d.name, evs)
        lab = i === nothing ? "not detected" :
              isfinite(evs[i].ev.R) ? "R = $(round(evs[i].ev.R, digits = 2))" : "discarded"
        annotate!(p, 1000lo + 0.03 * span, ypos, text(lab, 8, d.color, :left))
    end
    # panel letter "a)" above the top-left corner of the row
    j == 1 && annotate!(p, 1000lo - 0.10 * span, rev_ylims[2] + 0.05 * (rev_ylims[2] - rev_ylims[1]),
                        text("a)", 10, :left, "Computer Modern"))
    p
end

# c–e) simulated mean ± std for 3 θ
target_ky = [0.0, 10.0, 100.0]   # θ examples shown in panels c–e (closest calibrated θ)
selected = map(target_ky) do tgt
    tgt == 0.0 && return theta0_pos
    nonzero_pos[argmin([abs(1000 * theta_stats[k].theta - tgt) for k in nonzero_pos])]
end
mean_panels = map(enumerate(selected)) do (j, k)
    s = theta_stats[k]
    p = plot(curve_kyr, s.mean_curve; ribbon = s.std_curve, color = color_of[k], lw = 3.5,
             fillalpha = 0.25, fillcolor = color_of[k], label = false,
             xlims = plot_xlims, ylims = ALIGNED_YLIMS,
             xlabel = XLAB_ALIGNED,
             title = "$(('c','d','e')[j])) θ=$(round(1000 * s.theta, sigdigits = 3)) kyr, " *
                     "R=$(round(s.R_mean, digits = 2))", titlefontsize = 9, PANEL_MARGINS...)
    vline!(p, [0.0]; color = :black, lw = 1.0, label = false)   # T0
    hline!(p, [h_thr]; color = :gray, lw = 0.5, ls = :dash, label = false)
    decay_recovery_lines!(p, s.dn_mean, s.up_mean)
    region_arrows!(p, s.dn_mean, 0.0, s.up_mean, ALIGNED_ARROW_Y)
    p
end

# R vs θ
θky   = [1000 * theta_stats[k].theta for k in nonzero_pos]
xlims_θ = (0.8 * minimum(θky), 1.2 * maximum(θky))
# Drawn in layers (Plots' `yerror` colours the bar caps with the marker colour and thick):
#   gray line joining the means → thin black ±std bars with short thin caps → θ-coloured dots
θ_colors = [color_of[k] for k in nonzero_pos]
R_means  = [theta_stats[k].R_mean for k in nonzero_pos]
R_stds   = [theta_stats[k].R_std for k in nonzero_pos]
R_panel = plot(θky, R_means; color = :gray, lw = 1,
    xscale = :log10, xlims = xlims_θ, minorgrid = true, legend = false,
    xlabel = L"\theta\ (\mathrm{kyr})",   # y label drawn by hand below
    title = "f) " * L"R = (\delta t^\downarrow - \delta t^\uparrow)/(\delta t^\downarrow + \delta t^\uparrow)" *
            ": mean ± std", titlefontsize = 10, PANEL_MARGINS...)
cap = 1.06   # half-width of the caps, as a factor on the log θ axis
for (x, m, s) in zip(θky, R_means, R_stds)
    plot!(R_panel, [x, x], [m - s, m + s]; color = :black, lw = 0.8)
    plot!(R_panel, [x / cap, x * cap], [m - s, m - s]; color = :black, lw = 0.8)
    plot!(R_panel, [x / cap, x * cap], [m + s, m + s]; color = :black, lw = 0.8)
end
hline!(R_panel, [0.0]; color = :gray, lw = 1, ls = :dot)
hline!(R_panel, [s0.R_mean - s0.R_std, s0.R_mean + s0.R_std]; color = :black, lw = 0.75, ls = :dash, alpha = 0.6)
hline!(R_panel, [s0.R_mean]; color = :black, lw = 1.5, ls = :dash)
# Data: one mean ± std band per record, in the dataset colour (PADM2M dashed, Sint-2000 dotted)
for d in real_datasets
    r = real_R_by_ds[d.name]
    hspan!(R_panel, [mean(r) - std(r), mean(r) + std(r)]; color = d.color, alpha = 0.10)
    hline!(R_panel, [mean(r)]; color = d.color, lw = 1.5, ls = d.ls)
end
# θ-coloured dots last, so no reference line or band is drawn over them
scatter!(R_panel, θky, R_means; marker = :circle, markersize = 6, markercolor = θ_colors,
         markerstrokewidth = 0)

# p-value vs θ — one test of means per record (n = 5 each), in the dataset colour:
# PADM2M dashed/circles, Sint-2000 dotted/diamonds; θ = 0 as horizontal black lines in the
# same styles.
# Same style as Test No1 (1_chron_length/analyze_theta_calibrated.jl, panel d): "shifted
# logit" axis centred at α = 0.05 (p = 0.05 sits at y = 0), light red rejection region
# p < 0.05 with a solid threshold line, θ = 0 as a horizontal black dashed line. One extra
# tick at 1e-4 (= the clamp floor) so very small Monte Carlo p-values stay on the panel.
shifted_logit(p, α = 0.05) = (pc = clamp(p, 1e-4, 1 - 1e-4);
                              log10((pc * (1 - α)) / (α * (1 - pc))))
plain_number(x) = x >= 1 ? string(isinteger(x) ? Int(x) : round(x; digits = 2)) :
                           string(round(x; digits = ceil(Int, -log10(x)) + 1))
p_ticks = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
y_ticks = shifted_logit.(p_ticks)

pvalue_panel = plot(; xscale = :log10, xlims = xlims_θ, minorgrid = true,
    xformatter = plain_number, xlabel = L"\theta\ (\mathrm{kyr})",   # y label drawn by hand below
    yticks = (y_ticks, string.(p_ticks)),
    ylims = (minimum(y_ticks) - 0.1, maximum(y_ticks)), legend = false,
    title = "g) Test of means per record (n = 5)", titlefontsize = 10, PANEL_MARGINS...)
hspan!(pvalue_panel, [minimum(y_ticks) - 0.1, shifted_logit(0.05)]; color = :firebrick, alpha = 0.08, label = false)
hline!(pvalue_panel, [shifted_logit(0.05)]; color = :firebrick, lw = 1, ls = :solid, alpha = 0.6, label = false)
for (d, mk) in zip(real_datasets, (:circle, :diamond))
    pv = pvals_by_ds[d.name]
    y = shifted_logit.([pv[k] for k in nonzero_pos])
    # line in the dataset colour, then the markers on top in the θ colour (no outline, so
    # the diamonds keep their shape)
    plot!(pvalue_panel, θky, y; color = d.color, ls = :solid, lw = 2.5, label = false)
    scatter!(pvalue_panel, θky, y; marker = mk, markersize = mk === :diamond ? 8 : 6,
             markercolor = θ_colors, markerstrokewidth = 0, label = false)
    hline!(pvalue_panel, [shifted_logit(pv[theta0_pos])]; color = :black, lw = 1.5, ls = d.ls,
           label = false)
end

# y labels of e and f drawn as rotated annotations at a relative x position (fraction of the
# plot-area width, negative = left of the axis). GR's own `ylabel` sits a fixed, large
# distance from the tick labels with no option to reduce it; these offsets put the labels
# ~10 px from the widest tick label ("−0.25" in e, "0.0001" in f).
annotate!(R_panel, (-0.16, 0.5), text(L"R", 11, :center, rotation = 90))
annotate!(pvalue_panel, (-0.20, 0.5), text(L"p\mathrm{-value}", 11, :center, rotation = 90))

legend_fig = plot(; legend = :left, framestyle = :none, grid = false, xaxis = false, yaxis = false,
                  legendfontsize = 11)
# GR has no legend row-spacing option: every entry is followed by an invisible blank entry
# so the items are not packed together.
function leg!(; kw...)
    plot!(legend_fig, [NaN], [NaN]; kw...)
    plot!(legend_fig, [NaN], [NaN]; label = " ", linealpha = 0, markeralpha = 0)
end
for d in real_datasets
    leg!(; color = d.color, ls = :solid, lw = 2.5, label = d.name)
end
leg!(; color = :black, lw = 3, label = "Data mean (panel b)")
leg!(; color = DECLINE_COLOR, lw = 1.5, ls = DECLINE_LS, label = "T↓, T↑")
leg!(; color = :black, lw = 1.0, ls = :solid, label = "T0")
if show_region_arrows
    leg!(; color = DECAY_ARROW_COLOR, lw = 2, label = "Decay region [T↓, T0]")
    leg!(; color = RECOVERY_ARROW_COLOR, lw = 2, label = "Recovery region [T0, T↑]")
end
for (j, k) in enumerate(selected)
    leg!(; color = color_of[k], lw = 3.5,
         label = "θ = $(round(1000 * theta_stats[k].theta, sigdigits = 3)) kyr (panel $(('c','d','e')[j]))")
end
leg!(; color = :black, lw = 1.5, ls = :dash, label = "θ = 0 (model)")
for d in real_datasets
    leg!(; color = d.color, lw = 1.5, ls = d.ls, label = "Data R: $(d.name) (mean ± std)")
end
for (d, mk) in zip(real_datasets, (:circle, :diamond))
    leg!(; color = d.color, ls = :solid, lw = 2.5, marker = mk, markersize = mk === :diamond ? 8 : 6,
         markercolor = :gray60, markerstrokewidth = 0, label = "p-value: $(d.name)")
end
leg!(; seriestype = :shape, color = :firebrick, alpha = 0.15, lw = 0, label = "p < 0.05")

# Three rows of equal height (1/3 each), legend in its own column outside them:
#   row 1: panel a, the five observed reversals (full width of the main block)
#   row 2: panel b | R (f)
#   row 3: panels c–e | p-value (g)
# Rows 2 and 3 use the same column split, so f lines up with b and g with c–e.
rev_strip = plot(reversal_panels...; layout = (1, length(reversal_panels)))
row2      = plot(snapshot_panel, R_panel; layout = @layout([a{0.74w} b]))
row3      = plot(plot(mean_panels...; layout = (1, 3)), pvalue_panel; layout = @layout([a{0.74w} b]))
main_fig  = plot(rev_strip, row2, row3; layout = grid(3, 1))
# No margins here: they would override the per-panel margins of every subplot.
fig = plot(main_fig, legend_fig; layout = @layout([a{0.84w} b]), size = (2300, 1350))
Plots.GR.setarrowsize(REGION_ARROW_SIZE)   # arrow-head size of the region arrows
savefig(fig, joinpath(fig_dir, "asymmetry_summary.pdf"))

# ======================================================================================
# Plot helpers for the diagnostic figures (real time axis, Ma; signed X_t)
# ======================================================================================
const TIMES_LEGEND = "black dashed = chron boundaries T_i (±h0 hysteresis);  green = T↓;  " *
                     "grey band = [T0−, T0+];  purple = T↑;  red dotted = discarded transition;  " *
                     "horizontal: dashed = ±h, dotted = ±h0"

function threshold_lines!(p)
    hline!(p, [0.0]; color = :gray, lw = 0.5, label = false)
    hline!(p, [-h_thr, h_thr]; color = :gray, lw = 0.7, ls = :dash, label = false)
    hline!(p, [-h0, h0]; color = :gray, lw = 0.7, ls = :dot, label = false)
end

"""Draw T↓, [T0−, T0+], T↑ of `ev` (or a red marker at `Tmid` if the event was discarded)."""
function mark_event!(p, ev, Tmid)
    if !isfinite(ev.R)
        vline!(p, [Tmid]; color = :red, lw = 1.5, ls = :dot, label = false)
        isnan(ev.TD) || vline!(p, [ev.TD]; color = :seagreen, lw = 1.2, label = false)
        isnan(ev.TU) || vline!(p, [ev.TU]; color = :purple, lw = 1.2, label = false)
        return p
    end
    vspan!(p, [ev.T0m, max(ev.T0p, ev.T0m + 1e-7)]; color = :gray, alpha = 0.35, label = false)
    vline!(p, [ev.T0m, ev.T0p]; color = :gray30, lw = 0.8, label = false)
    vline!(p, [ev.TD]; color = :seagreen, lw = 1.2, label = false)
    vline!(p, [ev.TU]; color = :purple, lw = 1.2, label = false)
    return p
end

"""Signed series in [tlo, thi] with every T_i and every event's times marked."""
function times_panel(t, x, Ts, events, tlo, thi; color = :steelblue, title = "", markers = false)
    mask = (t .>= tlo) .& (t .<= thi)
    p = plot(t[mask], x[mask]; color = color, lw = 0.8, label = false, xlims = (tlo, thi),
             title = title, titlefontsize = 8, xlabel = "Time (Ma)", ylabel = L"X_t",
             marker = markers ? :circle : :none, ms = 1.5, msw = 0)
    threshold_lines!(p)
    Tin = filter(T -> tlo <= T <= thi, Ts)
    isempty(Tin) || vline!(p, Tin; color = :black, lw = 0.8, ls = :dash, label = false)
    for (Tmid, ev) in events
        mark_event!(p, ev, Tmid)
    end
    return p
end

# ======================================================================================
# Figure 2: data — full records + close-up of each event, real time axis
# ======================================================================================
full_panels = map(real_datasets) do d
    evs = [(e.Tmid, e.ev) for e in real_events if e.dataset == d.name]
    e1 = real_events[findfirst(e -> e.dataset == d.name, real_events)]
    times_panel(e1.t, e1.x, e1.Ts, evs, e1.t[1], e1.t[end]; color = d.color,
                title = "$(d.name): full record")
end
zoom_panels = Any[]
for d in real_datasets, (rname, _, _) in known_reversals
    i = findfirst(e -> e.dataset == d.name && e.name == rname, real_events)
    if i === nothing
        push!(zoom_panels, plot(; framestyle = :none, title = "$(d.name) / $rname: not detected",
                                 titlefontsize = 7)); continue
    end
    e = real_events[i]
    push!(zoom_panels, times_panel(e.t, e.x, e.Ts, [(e.Tmid, e.ev)], e.Tmid - 0.15, e.Tmid + 0.10;
        color = d.color, markers = true,   # one colour per dataset (blue PADM2M, red Sint-2000)
        title = "$(d.name) / $rname   R=$(round(e.ev.R, digits = 2))"))
end
nrev = length(known_reversals)
diag_fig = plot(full_panels..., zoom_panels...;
                layout = @layout([a{0.22h}; b{0.22h}; grid(2, nrev)]),
                size = (380 * nrev, 1050),
                plot_title = "Data — " * TIMES_LEGEND, plot_titlefontsize = 9)
savefig(diag_fig, joinpath(fig_dir, "reversal_data_closeup.pdf"))

# ======================================================================================
# Figure 3: δ↓ vs δ↑ distributions for the 4 selected θ, data overlaid
# ======================================================================================
real_δD = [1000 * e.ev.δD for e in real_events if isfinite(e.ev.R)]
real_δU = [1000 * e.ev.δU for e in real_events if isfinite(e.ev.R)]
δ_panels = map(enumerate(selected)) do (j, k)
    s = theta_stats[k]
    p = histogram(s.δD; bins = 0:2:120, normalize = :pdf, alpha = 0.5, color = :steelblue,
                  label = L"\delta t^\downarrow", xlabel = "kyr", ylabel = "density",
                  title = "$(('c','d','e')[j])) θ=$(round(1000 * s.theta, sigdigits = 3)) kyr (n=$(s.n))",
                  titlefontsize = 9)
    histogram!(p, s.δU; bins = 0:2:120, normalize = :pdf, alpha = 0.5, color = :darkorange,
               label = L"\delta t^\uparrow")
    vline!(p, real_δD; color = :steelblue, lw = 1.2, label = false)
    vline!(p, real_δU; color = :darkorange, lw = 1.2, ls = :dash, label = false)
    p
end
δ_fig = plot(δ_panels...; layout = (1, 3), size = (1050, 350),
             plot_title = "Decline δt↓ and recovery δt↑ (simulated histograms; data as vertical lines)",
             plot_titlefontsize = 11, bottom_margin = 5Plots.mm)
savefig(δ_fig, joinpath(fig_dir, "delta_distribution.pdf"))

# ======================================================================================
# Figure 4: SDE sanity check — a few short runs, every time involved drawn on the path.
# One row per θ: the full 3 Ma run (all T_i, all events, discarded ones in red) followed
# by close-ups of 3 transitions (two kept + one discarded, when the run has one).
# Uses exactly the same `reversal_event` as the main sweep.
# ======================================================================================
diag_targets_ky = [0.0, 5.0, 50.0]
diag_span       = 30.0    # Ma simulated per run (long enough to contain a discarded event)
diag_show       = 3.0     # Ma shown in the full-run panel (centred on a discarded event)
sde_rows = Any[]
for tgt in diag_targets_ky
    iθ = tgt == 0.0 ? findfirst(r -> r.theta == 0.0, rows) :
         argmin([r.theta == 0.0 ? Inf : abs(1000 * r.theta - tgt) for r in rows])
    θ, σ = rows[iθ].theta, rows[iθ].sigma
    ds = simulate(; model = :Davis2021, tspan = (-(diag_span + burnin), 0.0), θ = θ, σ = σ,
                  saveat = saveat, seed = 1000 + iθ, detect_reversals = true,
                  criterion = :Gammaitoni1998, h0 = h0,
                  maxiters = θ == 0.0 ? 100_000 : ceil(Int, 300 * (diag_span + burnin) / θ))
    t  = collect(Float64, ds.time); x = [v.value for v in ds.value]; Ts = ds.reversals
    ks = [k for k in 2:(length(Ts) - 1) if Ts[k - 1] >= -diag_span]
    evs = [(Ts[k], reversal_event(t, x, Ts[k - 1], Ts[k], Ts[k + 1], h_thr)) for k in ks]
    n_bad = count(e -> !isfinite(e[2].R), evs)
    θlab = "θ=$(round(1000 * θ, sigdigits = 3)) kyr"
    println("SDE check $θlab: $(length(evs)) transitions in $diag_span Ma, $n_bad discarded")

    good = findall(e -> isfinite(e[2].R), evs)
    bad  = findall(e -> !isfinite(e[2].R), evs)
    # full-run panel: diag_show Ma around the first discarded event (or the last diag_show Ma)
    c0 = isempty(bad) ? -diag_show / 2 : clamp(evs[bad[1]][1], -diag_span + diag_show / 2, -diag_show / 2)
    lo, hi = c0 - diag_show / 2, c0 + diag_show / 2
    evs_in = filter(e -> lo <= e[1] <= hi, evs)
    full = times_panel(t, x, Ts, evs_in, lo, hi; color = :black,
                       title = "$θlab: $(diag_show) Ma of a $(diag_span) Ma run " *
                               "($(length(evs)) transitions, $n_bad discarded in the run)")
    # close-ups: prefer kept events inside the shown window, then the discarded one
    good = vcat(filter(j -> lo <= evs[j][1] <= hi, good), filter(j -> !(lo <= evs[j][1] <= hi), good))
    pick = vcat(good[1:min(2, end)], bad[1:min(1, end)])
    length(pick) < 3 && (pick = vcat(pick, good[3:min(end, 2 + 3 - length(pick))]))
    zooms = map(pick) do j
        Tmid, ev = evs[j]
        lo = isfinite(ev.R) ? min(ev.TD, Tmid) - 0.03 : Tmid - 0.12
        hi = isfinite(ev.R) ? max(ev.TU, Tmid) + 0.03 : Tmid + 0.08
        why = isfinite(ev.R) ? "R=$(round(ev.R, digits = 2)), δ↓=$(round(1000 * ev.δD, digits = 1)), " *
                               "δ↑=$(round(1000 * ev.δU, digits = 1)) kyr" :
              "DISCARDED: " * (isnan(ev.TD) && isnan(ev.TU) ? "no T↓, no T↑" :
                               isnan(ev.TD) ? "no T↓" : "no T↑")
        times_panel(t, x, Ts, [(Tmid, ev)], lo, hi; color = :steelblue, markers = true,
                    title = why)
    end
    while length(zooms) < 3
        push!(zooms, plot(; framestyle = :none))
    end
    push!(sde_rows, full, zooms...)
end
sde_fig = plot(sde_rows...;
               layout = @layout([a{0.4w} b c d; e{0.4w} f g h; i{0.4w} j k l]),
               size = (2200, 1100), plot_title = "SDE — " * TIMES_LEGEND, plot_titlefontsize = 10)
savefig(sde_fig, joinpath(fig_dir, "sde_times_check.pdf"))

println("\nFigures saved to $fig_dir (asymmetry_summary, reversal_data_closeup, delta_distribution, sde_times_check)")
