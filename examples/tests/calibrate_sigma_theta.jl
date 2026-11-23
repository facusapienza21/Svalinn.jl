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

# simulate_Davis2021 now defaults to solver=:exact for θ>0 (an exact-noise-increment
# integrator, validated against the old adaptive LambaEM solver in
# examples/tests/exact_ou_noise_test.jl and compare_current_solver_statistics.jl — the
# adaptive solver was found to overcount reversals by up to ~40x at small θ). `alg` below
# is only meaningful for solver=:adaptive (and for θ=0, which always uses the adaptive
# path regardless of `solver` — see src/models/Davis2021.jl); left as `nothing` since
# this script doesn't override the solver choice.
solver_alg = nothing

sim_age    = 10.0   # init_age at θ=θ₀; scaled by θ elsewhere (see init_age_for)
sim_burnin = 1.0   # Ma

# Reference Davis2021 parameters (used in test.jl), and the fixed relaxation rate γ
# (Davis2021's default; not swept, so it's a plain constant here).
θ₀ = 0.5 / 1000
σ₀ = 1 / (2θ₀)
γ  = 100.0   # Myr^-1

# Closed-form Var[|X|]-matching σ(θ) (see sweep_theta_calibrated.jl) — used only as the
# initial bracket center for the bisection below, since it's already a decent guess.
σ_of_θ_guess(θ) = σ₀ * (θ₀ / θ) * (γ * θ + 1) / (γ * θ₀ + 1)

# Since maxiters ~ (age/θ), a fixed init_age costs proportionally more solver steps the
# smaller θ is (small θ means many small timesteps per unit of simulated age). Scaling
# init_age with θ keeps the *first* attempt's step count roughly θ-independent instead
# of ballooning for the smallest θ in the grid; floored so it never gets pathologically
# tiny (which would just mean paying for more — still cheap — growth doublings).
init_age_for(θ) = max(sim_age * θ / θ₀, 0.01)

# Same θ grid sweep_theta_calibrated.jl uses — keep these in sync, since the sweep
# script looks up σ by θ value from this table. The smallest point of the raw log-spaced
# range (θ=0.0001) was found to be numerically unreliable even for the calibration probe
# itself — the default solver intermittently fails to resolve reversals at all there
# (see conversation notes). It's dropped from the bisection grid below and replaced by a
# dedicated θ=0.0 (white-noise limit) point instead, added separately further down —
# θ=0.0 has its own verified-safe solver defaults (see simulate_Davis2021) and needs no
# bisection at all, since σ isn't a free parameter to calibrate there (see below).
n_grid_calib = 19
θ_grid = (10 .^ range(log10(0.1), log10(100.0); length = n_grid_calib) ./ 1000)[2:end]
n_grid = length(θ_grid)

# Cheap probe size for calibration — we only need a reasonably precise *mean*, not the
# full H distribution, so this can be much smaller than the sweep script's n_chrons.
# n_chrons_calib sets the sampling noise floor on the measured mean chron length
# (relative error ~ 1/sqrt(n_chrons_calib) for an exponential-like duration
# distribution). tol must sit safely above that floor, or the bisection chases a target
# it can never reliably resolve and oscillates instead of converging — e.g. 300 chrons
# gives ~6% noise, so tol=0.02 was unachievable. This is a one-time script, so it's
# worth paying for a bigger probe sample to get a tighter, still-achievable tolerance.
n_chrons_calib = 2000    # ~2.2% sampling noise floor
tol            = 0.05    # accept within ±1% of the target mean chron length
max_iters      = 25
max_bracket_expansions = 6

"""
    simulate_mean_chron_length(; θ, σ, n_chrons, ...) -> Float64

Mean chron length from a run grown until it has `n_chrons` chrons. If `σ` is small
enough that reversals become too rare to reach `n_chrons` within `max_tries` growth
steps, returns `Inf` instead of erroring — a `σ` this small is unambiguously "too
small" (mean length way above any sane target), and `Inf` lets `calibrate_sigma`'s
bisection use that directional information instead of crashing on it.

`L_cap`, if given, lets it bail out *early*: after each attempt, if the partial result
already has a mean chron length above `L_cap`, there's no need to keep doubling `age`
(and `maxiters` with it) chasing a precise number for a σ that's already obviously too
small — `Inf` is exactly as informative to the bisection as the precise value would be.
"""
function simulate_mean_chron_length(; θ, σ, n_chrons, burnin, init_age = sim_age,
                                     growth = 2.0, max_tries = 10, seed = 1, L_cap = nothing)
    age = init_age
    local pd
    for attempt in 1:max_tries
        # θ=0 (white noise) needs no θ-scaled maxiters — simulate_Davis2021 already
        # auto-derives a safe fixed-step dt/maxiters for that case (see its docstring);
        # the formula below would divide by zero at θ=0, hence the guard.
        maxiters = θ == 0.0 ? 100_000 : ceil(Int, 300 * (age + burnin) / θ)
        pd = simulate_chrons(; model = :Davis2021, tspan = (-age, 0.0), burnin = burnin,
                              θ = θ, σ = σ, seed = seed, maxiters = maxiters, alg = solver_alg)
        length(pd.start) >= n_chrons && return mean(chron_lengths(pd)[1:n_chrons])

        if L_cap !== nothing && !isempty(pd.start) && mean(chron_lengths(pd)) > L_cap
            return Inf
        end

        age *= growth
    end
    return Inf
end

L_ref = simulate_mean_chron_length(; θ = θ₀, σ = σ₀, n_chrons = n_chrons_calib,
                                    burnin = sim_burnin, seed = 1)
println("Reference Davis2021 (θ₀=$θ₀, σ₀=$σ₀): mean chron length L_ref = " *
        "$(round(L_ref, digits = 4)) Ma")

"""
    calibrate_sigma(θ; L_target, seed) -> (σ, L, n_iters, converged)

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
                                           init_age = init_age_for(θ), L_cap = 5 * L_target)
    tag = "θ=$(round(θ, sigdigits = 4))"

    σ_guess = σ_of_θ_guess(θ)
    σ_lo, σ_hi = σ_guess / 3, σ_guess * 3
    L_lo, L_hi = probe(σ_lo), probe(σ_hi)
    plog("$tag: initial bracket σ_lo=$(round(σ_lo, sigdigits=4)) (L=$(round(L_lo, digits=4))), " *
         "σ_hi=$(round(σ_hi, sigdigits=4)) (L=$(round(L_hi, digits=4))), target=$(round(L_target, digits=4))")

    n_expand = 0
    while L_lo < L_target && n_expand < max_bracket_expansions
        σ_lo /= 3
        L_lo  = probe(σ_lo)
        n_expand += 1
        plog("$tag: expanding σ_lo -> $(round(σ_lo, sigdigits=4)) (L=$(round(L_lo, digits=4)))")
    end
    n_expand = 0
    while L_hi > L_target && n_expand < max_bracket_expansions
        σ_hi *= 3
        L_hi  = probe(σ_hi)
        n_expand += 1
        plog("$tag: expanding σ_hi -> $(round(σ_hi, sigdigits=4)) (L=$(round(L_hi, digits=4)))")
    end

    if !(L_lo >= L_target >= L_hi)
        plog("$tag: WARNING could not bracket target (L_lo=$L_lo, L_hi=$L_hi, " *
             "target=$L_target) — returning closest bracket endpoint")
        return abs(L_lo - L_target) < abs(L_hi - L_target) ? (σ_lo, L_lo, 0, false) :
                                                               (σ_hi, L_hi, 0, false)
    end

    σ, L = σ_guess, NaN
    for iter in 1:max_iters
        σ = sqrt(σ_lo * σ_hi)   # geometric midpoint
        L = probe(σ)
        plog("$tag: iter $iter/$max_iters σ=$(round(σ, sigdigits=6)) L=$(round(L, digits=4)) " *
             "(target=$(round(L_target, digits=4)), rel.err=$(round(abs(L-L_target)/L_target, digits=4)))")
        if abs(L - L_target) / L_target <= tol
            plog("$tag: converged in $iter iters")
            return (σ, L, iter, true)
        end
        if L > L_target
            σ_lo = σ   # still too long -> need shorter-chron (larger σ) side
        else
            σ_hi = σ
        end
    end
    plog("$tag: did NOT converge within $max_iters iters (final rel.err=" *
         "$(round(abs(L-L_target)/L_target, digits=4)))")
    return (σ, L, max_iters, false)
end

results = Vector{NamedTuple}(undef, n_grid)
Threads.@threads for iθ in eachindex(θ_grid)
    θ = θ_grid[iθ]
    plog("θ=$(round(θ, sigdigits=4)): starting calibration ($iθ/$n_grid)")
    σ, L, n_iters, converged = calibrate_sigma(θ; L_target = L_ref, seed = 1000 + iθ)
    results[iθ] = (index = iθ, theta = θ, sigma = σ, mean_chron_length = L,
                   n_iters = n_iters, converged = converged)
    plog("θ=$θ -> σ=$(round(σ, sigdigits = 6)), L=$(round(L, digits = 4)) Ma " *
         "(target $(round(L_ref, digits = 4)) Ma), $n_iters iters, converged=$converged")
end

# --- θ=0.0 (white-noise limit): no free σ to calibrate against L_ref here — σ is instead
# the limiting product θσ (see simulate_Davis2021), and the package's own default (0.5,
# passing σ=nothing) already IS the model's natural reference point, not something to be
# tuned to match L_ref. No simulation is run here at all — this script only defines the
# (θ, σ) table; sweep_theta_calibrated.jl is where θ=0.0 actually gets simulated, exactly
# like every other row.
σ_white = 0.5   # must match simulate_Davis2021's θ=0 default (σ=nothing -> 0.5)
white_result = (index = 0, theta = 0.0, sigma = σ_white, mean_chron_length = NaN,
                n_iters = 0, converged = true)

out_dir = joinpath(@__DIR__, "data")
mkpath(out_dir)
open(joinpath(out_dir, "sigma_calibration.txt"), "w") do io
    println(io, "Index\tTheta\tSigma\tMeanChronLength\tNIters\tConverged")
    for r in sort([white_result; results]; by = r -> r.index)
        println(io, "$(r.index)\t$(r.theta)\t$(r.sigma)\t$(r.mean_chron_length)\t" *
                     "$(r.n_iters)\t$(r.converged)")
    end
end

n_failed = count(!(r -> r.converged), results)
println("Done. Wrote sigma_calibration.txt to $out_dir " *
        "($(n_grid - n_failed)/$n_grid bisected θ>0 points converged within tol=$tol, " *
        "plus θ=0.0: converged=$(white_result.converged)).")

# ------------------------------------------------------------
# σ(θ) figure — lives here (not in any one test subdirectory) for the same reason the
# calibration table itself does: it's a property of the shared calibration, not specific
# to any one downstream test.
# ------------------------------------------------------------
all_rows = sort([white_result; results]; by = r -> r.index)
θ_all         = [r.theta for r in all_rows]
σ_all         = [r.sigma for r in all_rows]
converged_all = [r.converged for r in all_rows]

# Plain decimal tick labels instead of scientific notation.
plain_number(x) = x >= 1 ? string(isinteger(x) ? Int(x) : round(x; digits = 2)) :
                            string(round(x; digits = ceil(Int, -log10(x)) + 1))

fig_σ = plot(; xscale = :log10, yscale = :log10, minorgrid = true,
               xformatter = plain_number, yformatter = plain_number,
               xlabel = L"\theta\ (\mathrm{Myr})", ylabel = L"\sigma",
               legend = :topright, size = (600, 450))

θ_pos = θ_all[θ_all .> 0]   # θ=0 can't sit on a log-x axis
θ_fine = 10 .^ range(log10(minimum(θ_pos)), log10(maximum(θ_pos)); length = 200)
plot!(fig_σ, θ_fine, σ_of_θ_guess.(θ_fine);
      color = :gray, lw = 1, ls = :dash, label = "closed-form Var-matching guess")

pos_converged = (θ_all .> 0) .& converged_all
plot!(fig_σ, θ_all[pos_converged], σ_all[pos_converged];
      marker = :circle, color = :steelblue, lw = 1, label = "calibrated (converged)")

pos_failed = (θ_all .> 0) .& .!converged_all
if any(pos_failed)
    scatter!(fig_σ, θ_all[pos_failed], σ_all[pos_failed];
             marker = :xcross, markersize = 7, markerstrokewidth = 2,
             color = :firebrick, label = "did not converge")
end

scatter!(fig_σ, [θ₀], [σ₀];
         marker = :star5, markersize = 10, color = :yellow, markerstrokecolor = :black,
         markerstrokewidth = 1, label = L"\mathrm{reference}\ (\theta_0, \sigma_0)")

figure_dir = joinpath(@__DIR__, "figures")
mkpath(figure_dir)
savefig(fig_σ, joinpath(figure_dir, "sigma_calibration.pdf"))
println("Saved to $figure_dir/sigma_calibration.pdf")
display(fig_σ)
