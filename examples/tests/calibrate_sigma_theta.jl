# Run with multiple threads for speed (each θ's calibration runs independently):
#
#   julia -t auto --project=examples examples/tests/calibrate_sigma_theta.jl
#
# One-time (expensive) calibration: for each θ in the grid, numerically find the σ that
# makes the Davis2021 model's mean chron length match the reference model's — rather
# than the previous closed-form σ(θ) (which only matched Var[|X|], and only
# approximately preserved mean chron length, hence the θ_eff post-hoc relabeling). This
# script pays the iterative-bisection cost ONCE and writes a (θ, σ) lookup table to
# disk (examples/tests/data/sigma_calibration.txt); every test subdirectory reads it
# back from there, so every θ point already has a matching mean chron length by
# construction — no relabeling needed downstream, and no test-specific copy to keep in
# sync. Lives directly under examples/tests/ (not inside any one test subdirectory)
# since it's a shared resource, not specific to the chron-length clustering test alone.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Svalinn
using Statistics: mean
using Plots
using LaTeXStrings

default(fontfamily = "Computer Modern")

sim_age    = 10.0   # Ma, initial span for the reference run (grown until enough chrons)
sim_burnin = 1.0    # Ma

# Reversal-detection threshold for the :Gammaitoni1998 hysteresis criterion. Must match
# the one used by the tests (1_chron_length/main_run.jl uses h0 = 0.5).
h0 = 0.5

# Reference Davis2021 parameters, and the fixed relaxation rate γ (Davis2021's default;
# not swept, so it's a plain constant here).
θ₀ = 0.5 / 1000
σ₀ = 1 / (2θ₀)
γ  = 100.0   # Myr^-1

# Closed-form σ(θ) — used as the initial bracket center for the bisection below. It
# keeps the stationary variance of the linearized dynamics around a well,
# Var[x] ∝ σθ / (1 + γθ), equal to the reference value.
σ_of_θ_guess(θ) = σ₀ * (θ₀ / θ) * (γ * θ + 1) / (γ * θ₀ + 1)

# θ grid: 19 log-spaced points between 0.1 and 100 kyr, dropping the first one. A
# dedicated θ = 0.0 (white-noise limit) row is added separately further down; σ is not
# calibrated there (see below).
n_grid_calib = 19
θ_grid = (10 .^ range(log10(0.1), log10(100.0); length = n_grid_calib) ./ 1000)[2:end]
n_grid = length(θ_grid)

# n_chrons_calib sets the sampling noise floor on the measured mean chron length
# (relative error ~ 1/sqrt(n_chrons_calib) for an exponential-like duration
# distribution). tol must sit safely above that floor, or the bisection chases a target
# it can never reliably resolve and oscillates instead of converging.
n_chrons_calib = 10000    # ~1.4% sampling noise floor
tol            = 0.02    # accept within ±5% of the target mean chron length
max_iters      = 25
max_bracket_expansions = 6

"""
    simulate_mean_chron_length(; θ, σ, n_chrons, ...) -> (L, chrons_per_Ma)

Mean chron length `L` (Ma) of the first `n_chrons` normal chrons from a run grown until
it has that many, together with the average number of chrons per Ma over the run
(both polarities: twice the number of complete normal chrons per Ma, since normal and
reverse chrons alternate).

The run starts at `init_age` Ma and is doubled (from scratch, same seed) up to
`max_tries` times. If `σ` is small enough that reversals become too rare to reach
`n_chrons`, returns `(Inf, rate)` instead of erroring — a `σ` this small is
unambiguously "too small", and `Inf` lets `calibrate_sigma`'s bisection use that
directional information instead of crashing on it.

`L_cap`, if given, lets it bail out *early*: after each attempt, if the partial result
already has a mean chron length above `L_cap`, there's no need to keep doubling `age`
chasing a precise number for a σ that's already obviously too small.
"""
function simulate_mean_chron_length(; θ, σ, n_chrons, burnin, init_age = sim_age,
                                     growth = 2.0, max_tries = 10, seed = 1, L_cap = nothing)
    age = init_age
    local pd
    for attempt in 1:max_tries
        pd = simulate_chrons(; model = :Davis2021, tspan = (-age, 0.0), burnin = burnin,
                              θ = θ, σ = σ, seed = seed, h0 = h0)
        rate = 2 * length(pd.start) / age
        length(pd.start) >= n_chrons && return (mean(chron_lengths(pd)[1:n_chrons]), rate)

        if L_cap !== nothing && !isempty(pd.start) && mean(chron_lengths(pd)) > L_cap
            return (Inf, rate)
        end

        age *= growth
    end
    return (Inf, 2 * length(pd.start) / (age / growth))
end

# Span needed to collect n_chrons normal chrons with mean length L (×2 since only normal
# chrons count, ×1.3 margin). The solver's step size does not depend on θ, so neither
# does this.
span_for(n_chrons, L) = 1.3 * 2 * n_chrons * L

# Reference run. Its initial span is sized from a short pilot run (500 chrons), so it
# does not depend on how many doublings `simulate_mean_chron_length` allows.
L_pilot, _ = simulate_mean_chron_length(; θ = θ₀, σ = σ₀, n_chrons = 500,
                                         burnin = sim_burnin, seed = 1)
L_ref, rate_ref = simulate_mean_chron_length(; θ = θ₀, σ = σ₀, n_chrons = n_chrons_calib,
                                              burnin = sim_burnin, seed = 1,
                                              init_age = span_for(n_chrons_calib, L_pilot))
isfinite(L_ref) || error("Reference run did not reach $n_chrons_calib chrons; increase " *
                         "the initial span or max_tries in simulate_mean_chron_length.")
println("Reference Davis2021 (θ₀=$θ₀, σ₀=$σ₀): mean chron length L_ref = " *
        "$(round(L_ref, digits = 4)) Ma, $(round(rate_ref, digits = 2)) chrons/Ma")

# Initial span for every calibration probe: long enough to contain n_chrons_calib normal
# chrons at the target mean length, so almost every probe needs a single run.
init_age_calib = span_for(n_chrons_calib, L_ref)
println("Initial span per calibration probe: $(round(init_age_calib, digits = 1)) Ma")

"""
    calibrate_sigma(θ; L_target, seed) -> (σ, L, chrons_per_Ma, n_iters, converged)

Bisect (in log σ, since σ spans orders of magnitude) for the σ that makes this θ's
simulated mean chron length match `L_target`. Mean chron length is monotonically
*decreasing* in σ (more OU noise -> bigger kicks -> more frequent reversals -> shorter
chrons), so the bracket is expanded until it straddles the target before bisecting.
"""
progress_lock = ReentrantLock()
plog(msg) = lock(() -> println(msg), progress_lock)

function calibrate_sigma(θ::Float64; L_target::Float64, seed::Int = 1)
    # 5x the target is already unambiguously "too long, σ too small" for bisection's
    # purposes — no need to wait for a precise number past that.
    probe(σ) = simulate_mean_chron_length(; θ = θ, σ = σ, n_chrons = n_chrons_calib,
                                           burnin = sim_burnin, seed = seed,
                                           init_age = init_age_calib, L_cap = 5 * L_target)
    tag = "θ=$(round(θ, sigdigits = 4))"

    σ_guess = σ_of_θ_guess(θ)
    σ_lo, σ_hi = σ_guess / 3, σ_guess * 3
    (L_lo, r_lo), (L_hi, r_hi) = probe(σ_lo), probe(σ_hi)
    plog("$tag: initial bracket σ_lo=$(round(σ_lo, sigdigits=4)) (L=$(round(L_lo, digits=4))), " *
         "σ_hi=$(round(σ_hi, sigdigits=4)) (L=$(round(L_hi, digits=4))), target=$(round(L_target, digits=4))")

    n_expand = 0
    while L_lo < L_target && n_expand < max_bracket_expansions
        σ_lo /= 3
        L_lo, r_lo = probe(σ_lo)
        n_expand += 1
        plog("$tag: expanding σ_lo -> $(round(σ_lo, sigdigits=4)) (L=$(round(L_lo, digits=4)))")
    end
    n_expand = 0
    while L_hi > L_target && n_expand < max_bracket_expansions
        σ_hi *= 3
        L_hi, r_hi = probe(σ_hi)
        n_expand += 1
        plog("$tag: expanding σ_hi -> $(round(σ_hi, sigdigits=4)) (L=$(round(L_hi, digits=4)))")
    end

    if !(L_lo >= L_target >= L_hi)
        plog("$tag: WARNING could not bracket target (L_lo=$L_lo, L_hi=$L_hi, " *
             "target=$L_target) — returning closest bracket endpoint")
        return abs(L_lo - L_target) < abs(L_hi - L_target) ? (σ_lo, L_lo, r_lo, 0, false) :
                                                               (σ_hi, L_hi, r_hi, 0, false)
    end

    σ, L, r = σ_guess, NaN, NaN
    for iter in 1:max_iters
        σ = sqrt(σ_lo * σ_hi)   # geometric midpoint
        L, r = probe(σ)
        plog("$tag: iter $iter/$max_iters σ=$(round(σ, sigdigits=6)) L=$(round(L, digits=4)) " *
             "(target=$(round(L_target, digits=4)), rel.err=$(round(abs(L-L_target)/L_target, digits=4)))")
        if abs(L - L_target) / L_target <= tol
            plog("$tag: converged in $iter iters")
            return (σ, L, r, iter, true)
        end
        if L > L_target
            σ_lo = σ   # still too long -> need shorter-chron (larger σ) side
        else
            σ_hi = σ
        end
    end
    plog("$tag: did NOT converge within $max_iters iters (final rel.err=" *
         "$(round(abs(L-L_target)/L_target, digits=4)))")
    return (σ, L, r, max_iters, false)
end

results = Vector{NamedTuple}(undef, n_grid)
Threads.@threads for iθ in eachindex(θ_grid)
    θ = θ_grid[iθ]
    plog("θ=$(round(θ, sigdigits=4)): starting calibration ($iθ/$n_grid)")
    σ, L, r, n_iters, converged = calibrate_sigma(θ; L_target = L_ref, seed = 1000 + iθ)
    results[iθ] = (index = iθ, theta = θ, sigma = σ, mean_chron_length = L,
                   chrons_per_Ma = r, n_iters = n_iters, converged = converged)
    plog("θ=$θ -> σ=$(round(σ, sigdigits = 6)), L=$(round(L, digits = 4)) Ma " *
         "(target $(round(L_ref, digits = 4)) Ma), $(round(r, digits = 2)) chrons/Ma, " *
         "$n_iters iters, converged=$converged")
end

# --- θ=0.0 (white-noise limit). Here σ is interpreted as the limiting product θσ (see
# simulate_Davis2021): as θ → 0 with θσ held fixed, the OU-driven model converges to
# white noise of intensity ḡ²·2θσ. Every value of θσ therefore has its own white-noise
# limit, and the question is which family of models θ = 0 should be the endpoint of.
#
# Fixed θσ = 0.5 (the value used below, the package default): this is the limit of the
# family σ = 1/(2θ), which contains the reference (θ₀σ₀ = 0.5). But equal θσ does NOT
# mean equal effective noise. X only responds to noise at frequencies below ~γ, and the
# OU spectrum 2σθ/(1 + ω²θ²) is attenuated there by a factor ~1/(1 + γθ). At the
# reference γθ₀ = 0.05, so the reference model is driven by ~5% less effective noise
# than white noise with the same θσ. Since the reversal rate is exponentially sensitive
# to the noise intensity (Kramers: rate ∝ exp(-2ΔU/s²)), θσ = 0.5 at θ = 0 gives ~15%
# more chrons per Ma than the reference (mean chron length ~0.243 vs ~0.277 Ma).
#
# Calibrated family (fixed mean chron length, the rows above): σ(θ) follows the closed
# form σ₀(θ₀/θ)(1 + γθ)/(1 + γθ₀), so θσ → σ₀θ₀/(1 + γθ₀) = 0.5/1.05 ≈ 0.476 as θ → 0
# (the smallest calibrated θ already give θσ ≈ 0.483, 0.486). The white-noise limit of
# the calibrated models is therefore θσ ≈ 0.48, not 0.5.
#
# Both are valid white-noise models: θσ = 0.5 matches the reference *parameter*,
# θσ ≈ 0.48 matches the reference *mean chron length*. For tests that compare all θ at
# equal mean chron length, the consistent θ = 0 point is the calibrated one (θσ ≈ 0.48);
# with θσ = 0.5, differences at θ = 0 can come partly from the ~15% higher reversal
# rate rather than from the absence of memory alone.
#
# Here σ is kept at 0.5 (not tuned to L_ref); the row is simulated once to report its
# mean chron length and chrons per Ma (shown in panel (b) of the figure).
σ_white = 0.5   # must match simulate_Davis2021's θ=0 default (σ=nothing -> 0.5)
L_white, rate_white = simulate_mean_chron_length(; θ = 0.0, σ = σ_white,
                                                  n_chrons = n_chrons_calib,
                                                  burnin = sim_burnin, seed = 1000,
                                                  init_age = init_age_calib)
println("θ=0.0 (white noise, σ=$σ_white): mean chron length $(round(L_white, digits = 4)) Ma, " *
        "$(round(rate_white, digits = 2)) chrons/Ma")
white_result = (index = 0, theta = 0.0, sigma = σ_white, mean_chron_length = L_white,
                chrons_per_Ma = rate_white, n_iters = 0, converged = true)

# Columns 1–6 are read by index by the tests; ChronsPerMa is appended last.
out_dir = joinpath(@__DIR__, "data")
mkpath(out_dir)
open(joinpath(out_dir, "sigma_calibration.txt"), "w") do io
    println(io, "Index\tTheta\tSigma\tMeanChronLength\tNIters\tConverged\tChronsPerMa")
    for r in sort([white_result; results]; by = r -> r.index)
        println(io, "$(r.index)\t$(r.theta)\t$(r.sigma)\t$(r.mean_chron_length)\t" *
                     "$(r.n_iters)\t$(r.converged)\t$(r.chrons_per_Ma)")
    end
end

n_failed = count(!(r -> r.converged), results)
println("Done. Wrote sigma_calibration.txt to $out_dir " *
        "($(n_grid - n_failed)/$n_grid bisected θ>0 points converged within tol=$tol, " *
        "plus θ=0.0: converged=$(white_result.converged)).")

# ------------------------------------------------------------
# Figure: (a) calibrated σ(θ) with the closed-form approximation, (b) chrons per Ma.
# ------------------------------------------------------------
all_rows = sort([white_result; results]; by = r -> r.index)
θ_all         = [r.theta for r in all_rows]
σ_all         = [r.sigma for r in all_rows]
rate_all      = [r.chrons_per_Ma for r in all_rows]
converged_all = [r.converged for r in all_rows]

# Plain decimal tick labels instead of scientific notation.
plain_number(x) = x >= 1 ? string(isinteger(x) ? Int(x) : round(x; digits = 2)) :
                            string(round(x; digits = ceil(Int, -log10(x)) + 1))

θ_pos  = θ_all[θ_all .> 0]   # θ=0 can't sit on a log-x axis
θ_fine = 10 .^ range(log10(minimum(θ_pos)), log10(maximum(θ_pos)); length = 200)
θ_lims  = (minimum(θ_pos) / 1.5, maximum(θ_pos) * 1.5)
θ_ticks = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
pos_converged = (θ_all .> 0) .& converged_all
pos_failed    = (θ_all .> 0) .& .!converged_all

# (a) σ(θ)
fig_σ = plot(; xscale = :log10, yscale = :log10, minorgrid = true,
               xlims = θ_lims, xticks = θ_ticks,
               xformatter = plain_number, yformatter = plain_number,
               ylabel = L"\sigma", legend = :topright, title = "(a) calibrated noise level",
               titlefontsize = 10)
plot!(fig_σ, θ_fine, σ_of_θ_guess.(θ_fine);
      color = :gray, lw = 1, ls = :dash,
      label = L"\sigma_0 \frac{\theta_0}{\theta} \frac{1 + \gamma\theta}{1 + \gamma\theta_0}")
plot!(fig_σ, θ_all[pos_converged], σ_all[pos_converged];
      marker = :circle, color = :steelblue, lw = 1, label = "calibrated (converged)")
if any(pos_failed)
    scatter!(fig_σ, θ_all[pos_failed], σ_all[pos_failed];
             marker = :xcross, markersize = 7, markerstrokewidth = 2,
             color = :firebrick, label = "did not converge")
end
scatter!(fig_σ, [θ₀], [σ₀];
         marker = :star5, markersize = 10, color = :yellow, markerstrokecolor = :black,
         markerstrokewidth = 1, label = L"\mathrm{reference}\ (\theta_0, \sigma_0)")

# (b) chrons per Ma (both polarities)
fig_rate = plot(; xscale = :log10, minorgrid = true, xformatter = plain_number,
                  xlims = θ_lims, xticks = θ_ticks,
                  xlabel = L"\theta\ (\mathrm{Myr})", ylabel = "chrons per Ma",
                  legend = :topright, title = "(b) average number of chrons per Ma",
                  titlefontsize = 10)
hline!(fig_rate, [rate_ref]; color = :black, lw = 1, ls = :dash,
       label = latexstring("\\mathrm{reference}\\ (\\theta_0):\\ $(round(rate_ref, digits = 2))"))
hline!(fig_rate, [rate_white]; color = :darkorange, lw = 1, ls = :dot,
       label = latexstring("\\theta = 0\\ (\\sigma = $σ_white):\\ $(round(rate_white, digits = 2))"))
plot!(fig_rate, θ_all[pos_converged], rate_all[pos_converged];
      marker = :circle, color = :steelblue, lw = 1, label = "calibrated (converged)")
if any(pos_failed)
    scatter!(fig_rate, θ_all[pos_failed], rate_all[pos_failed];
             marker = :xcross, markersize = 7, markerstrokewidth = 2,
             color = :firebrick, label = "did not converge")
end

fig = plot(fig_σ, fig_rate; layout = grid(2, 1; heights = [0.6, 0.4]),
           size = (650, 750), link = :x, left_margin = 4Plots.mm)

figure_dir = joinpath(@__DIR__, "figures")
mkpath(figure_dir)
savefig(fig, joinpath(figure_dir, "sigma_calibration.pdf"))
println("Saved to $figure_dir/sigma_calibration.pdf")
display(fig)
