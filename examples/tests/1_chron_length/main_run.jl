# Run with multiple threads to actually get the parallel speedup (each θ's replicates
# run via Threads.@threads) — it does nothing extra with the default single thread:
#
#   julia -t auto --project=examples examples/tests/1_chron_length/main_run.jl
#
# (`-t auto` uses all available cores; `-t N` for a specific count. In the VS Code Julia
# extension, set the "julia.NumThreads" setting instead of passing `-t` on the CLI.)

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Svalinn
using Statistics: mean

# A direct rigorous convergence test (fixed-step EM, dt shrunk from θ/20 down to θ/10000)
# showed LambaEM's default behavior (SciML's automatic algorithm choice here) is actually
# close to the converged answer at small θ — mean chron length converges to ~0.25-0.45 Ma,
# matching LambaEM, not the ~0.02-0.06 Ma that SKenCarp/ImplicitEM reported. Those two
# implicit-family algorithms agreeing with each other was mistaken earlier for validation;
# they instead appear to share a common failure mode (spurious extra reversals) when the
# fast OU forcing interacts with the nonlinear bistable drift. So: no algorithm override —
# stick with the default (LambaEM), matching calibrate_sigma_theta.jl.
solver_alg = nothing

# This script only runs the (expensive) SDE simulations and saves the resulting chrons
# to disk, one file per θ under data/simulations/, plus a master.txt index summarizing
# every θ point (raw θ, calibrated σ, mean chron length, etc.). See
# analyze_theta_calibrated.jl for reading these back in and producing the p-value /
# H-statistic plots — that split exists so replotting doesn't require rerunning the
# simulations.

sim_age = 20.0
sim_burnin = 1.0               # Ma

# Reference Davis2021 parameters (used in test.jl), and the fixed relaxation rate γ
# (Davis2021's default; not swept, so it's a plain constant here — recorded for
# provenance in params.txt, though it's no longer used directly in this script).
θ₀ = 0.5 / 1000
σ₀ = 1 / (2θ₀)
γ  = 100.0   # Myr^-1

# --- (θ, σ) grid: read from calibrate_sigma_theta.jl's numerically-calibrated table,
# instead of the closed-form Var[|X|]-matching σ(θ) used previously. That closed form
# only matched Var[|X|], not mean chron length, hence the θ_eff post-hoc relabeling
# used before — now σ is calibrated so mean chron length already matches the reference
# by construction for every θ, so θ_eff isn't needed at all. θ_grid/σ_grid are read
# directly off this table (not redefined here) so they can never drift out of sync with
# what was actually calibrated; rows that didn't converge are excluded, since their σ
# isn't trustworthy (see calibrate_sigma_theta.jl's own printed warnings for those).
sigma_calibration_file = joinpath(@__DIR__, "..", "data", "sigma_calibration.txt")
isfile(sigma_calibration_file) ||
    error("sigma_calibration.txt not found at $sigma_calibration_file — run " *
          "examples/tests/calibrate_sigma_theta.jl first.")

θ_grid = Float64[]
σ_grid = Float64[]
n_excluded = 0
open(sigma_calibration_file) do io
    readline(io)  # header
    for line in eachline(io)
        isempty(line) && continue
        c = split(line, '\t')
        if c[6] == "true"
            push!(θ_grid, parse(Float64, c[2]))
            push!(σ_grid, parse(Float64, c[3]))
        else
            global n_excluded += 1
        end
    end
end
n_grid = length(θ_grid)
println("Loaded $n_grid calibrated (θ, σ) points from sigma_calibration.txt " *
        "($n_excluded excluded, did not converge).")

# Total pooled H sample size per θ = n_replicates independent runs of n_chrons each
# (n_H_total = n_replicates * n_chrons is what actually determines KS-test power/noise
# — see the n_chrons/n_replicates discussion for why it's split this way rather than
# just cranking up n_chrons on a single run). n_chrons is fixed across the sweep (see
# grid_theta_sigma.jl for why this needs to be fixed rather than a fixed time span).
n_H_total    = 10000
n_replicates = 50
n_chrons     = n_H_total ÷ n_replicates

function simulate_n_chrons(; model, n_chrons, burnin, init_age = sim_age, growth = 2.0,
                            max_tries = 6, kwargs...)
    age = init_age
    local pd
    for attempt in 1:max_tries
        θ_cell   = kwargs[:θ]
        # θ=0 (white noise) needs no θ-scaled maxiters — simulate_Davis2021 already
        # auto-derives a safe fixed-step dt/maxiters for that case; the formula below
        # would divide by zero at θ=0, hence the guard (same fix as calibrate_sigma_theta.jl).
        maxiters = θ_cell == 0.0 ? 100_000 : ceil(Int, 300 * (age + burnin) / θ_cell)
        pd = simulate_chrons(; model = model, tspan = (-age, 0.0), burnin = burnin,
                              maxiters = maxiters, kwargs...)

        length(pd.start) >= n_chrons && return PolarityDataset(pd.name, pd.start[1:n_chrons],
                                                                 pd.stop[1:n_chrons], pd.polarity)
        age *= growth
    end
    error("simulate_n_chrons: only got $(length(pd.start))/$n_chrons chrons after " *
          "$max_tries tries (final age = $age Ma); raise init_age or growth.")
end

# Discard reversals that would create a chron shorter than this (Ma) — merges the short
# excursion into its neighbor instead of counting it as its own chron (see
# `filter_short_reversals`). 10,000 years = 0.01 Ma.
# min_chron_length = 10 / 1000
min_chron_length = 0.0 # We don't filter chrones for now.

# Reversal-detection threshold for the :Gammaitoni1998 hysteresis criterion: a chron
# starts once the field (normalized so equilibria sit at ±1) reaches +h0 (:normal) or
# -h0 (:reverse); touching the same threshold again while already in that state doesn't
# re-trigger. Previously this was left unset and silently fell back to `simulate`'s own
# package default (h0 = 0.5, see src/models/models.jl) — now explicit here instead.
h0 = 0.5

sim_dir = joinpath(@__DIR__, "data", "simulations")
mkpath(sim_dir)

# --- Reference run at (θ₀, σ₀) ---
ref_chrons = simulate_n_chrons(; model = :Davis2021, n_chrons = n_chrons, burnin = sim_burnin,
                                θ = θ₀, σ = σ₀, seed = 1, min_length = min_chron_length, h0 = h0,
                                alg = solver_alg)
L_ref = mean(chron_lengths(ref_chrons))
println("Reference Davis2021 (θ₀=$θ₀, σ₀=$σ₀): mean chron length = $(round(L_ref, digits = 3)) Ma")

open(joinpath(sim_dir, "reference.txt"), "w") do io
    println(io, "Start\tEnd")
    for (s, e) in zip(ref_chrons.start, ref_chrons.stop)
        println(io, "$s\t$e")
    end
end

# --- θ grid × replicates: one file per θ, plus a master index/summary row each ---
ndigits_grid = ndigits(n_grid)
master_rows  = Vector{NamedTuple}(undef, n_grid)

progress_lock = ReentrantLock()

for (iθ, θ) in enumerate(θ_grid)
    σ = σ_grid[iθ]
    filename = "theta_$(lpad(iθ, ndigits_grid, '0')).txt"
    println("θ $iθ/$n_grid (θ=$θ, σ=$σ): starting $n_replicates replicates...")

    # Parallel compute phase: each replicate is fully independent, and (since Julia 1.7)
    # Random.default_rng() is task-local, so Threads.@threads's per-iteration tasks each
    # get their own RNG stream — seeding inside one thread's iteration can't race with
    # another's. Writes go into a pre-sized per-replicate slot (never a shared
    # container), so this part needs no locking. Progress printing is the one shared
    # resource here, guarded by `progress_lock` so concurrent prints don't interleave.
    done       = Threads.Atomic{Int}(0)
    print_step = max(1, n_replicates ÷ 10)   # ~10 progress lines per θ
    replicate_results = Vector{Union{Nothing, NamedTuple}}(nothing, n_replicates)
    Threads.@threads for rep in 1:n_replicates
        sim_chrons = simulate_n_chrons(; model = :Davis2021, n_chrons = n_chrons,
                                        burnin = sim_burnin, θ = θ, σ = σ,
                                        seed = 1000 * iθ + rep,
                                        min_length = min_chron_length, h0 = h0,
                                        alg = solver_alg)
        replicate_results[rep] = (start = sim_chrons.start, stop = sim_chrons.stop,
                                   L = mean(chron_lengths(sim_chrons)))

        n_done = Threads.atomic_add!(done, 1) + 1
        if n_done % print_step == 0 || n_done == n_replicates
            lock(progress_lock) do
                println("  θ $iθ/$n_grid: $n_done/$n_replicates replicates done")
            end
        end
    end

    # Serial write phase: file I/O from multiple threads at once would interleave lines,
    # so writing happens only after all threads above have finished.
    L_sims  = Float64[]
    n_saved = 0
    open(joinpath(sim_dir, filename), "w") do io
        println(io, "Start\tEnd\tSimulation\tChron")
        for rep in 1:n_replicates
            r = replicate_results[rep]
            r === nothing && continue
            for (j, (s, e)) in enumerate(zip(r.start, r.stop))
                println(io, "$s\t$e\t$rep\t$j")
            end
            push!(L_sims, r.L)
            n_saved += 1
        end
    end

    L_sim = isempty(L_sims) ? NaN : mean(L_sims)

    master_rows[iθ] = (index = iθ, file = filename, theta = θ, sigma_calib = σ,
                        mean_chron_length = L_sim,
                        n_replicates_valid = n_saved, n_replicates_total = n_replicates,
                        n_chrons_per_replicate = n_chrons)

    println("θ = $θ, σ_calib = $σ -> mean chron length = $(round(L_sim, digits = 3)) Ma " *
            "(target $(round(L_ref, digits = 3)) Ma), " *
            "saved $n_saved/$n_replicates valid replicates -> $filename")
end

open(joinpath(sim_dir, "master.txt"), "w") do io
    println(io, "Index\tFile\tTheta\tSigmaCalib\tMeanChronLength\t" *
                 "NReplicatesValid\tNReplicatesTotal\tNChronsPerReplicate")
    for r in master_rows
        println(io, "$(r.index)\t$(r.file)\t$(r.theta)\t$(r.sigma_calib)\t" *
                     "$(r.mean_chron_length)\t$(r.n_replicates_valid)\t$(r.n_replicates_total)\t" *
                     "$(r.n_chrons_per_replicate)")
    end
end

# Small human-readable record of the parameters used for this run — L_ref is here (not
# in master.txt) since it's a single scalar shared by every row, not per-θ.
open(joinpath(sim_dir, "params.txt"), "w") do io
    println(io, "θ₀ = $θ₀")
    println(io, "σ₀ = $σ₀")
    println(io, "γ = $γ")
    println(io, "solver_alg = $(solver_alg === nothing ? "default (LambaEM)" : typeof(solver_alg))")
    println(io, "L_ref = $L_ref")
    println(io, "sigma_calibration_file = $sigma_calibration_file")
    println(io, "n_grid = $n_grid")
    println(io, "θ_grid range (Myr) = [$(first(θ_grid)), $(last(θ_grid))]")
    println(io, "n_H_total = $n_H_total")
    println(io, "n_replicates = $n_replicates")
    println(io, "n_chrons = $n_chrons")
    println(io, "min_chron_length = $min_chron_length")
    println(io, "h0 = $h0")
    println(io, "sim_burnin = $sim_burnin")
end

println("Done. Wrote reference.txt, theta_*.txt ($n_grid files), master.txt, params.txt to $sim_dir")
