"""
    simulate_Davis2021(; γ, X̄, ḡ, θ, σ, tspan, saveat, X0, z0, seed, solver)
        -> SynthesisDataset{SignedPaleointensity}

Simulate the Davis et al. (2021) stochastic bistable dipole model with colored
(Ornstein–Uhlenbeck) noise.  Returns a `SynthesisDataset{SignedPaleointensity}`
named `:Davis2021` whose values are ``X(t)/\\bar{X}`` (normalized so equilibria
sit at ±1).

The augmented Itô system is:

```math
dX = \\left[-\\gamma \\frac{X}{\\bar{X}}\\left(|X| - \\bar{X}\\right) + \\bar{g}\\,\\zeta\\right] dt
d\\zeta = -\\frac{\\zeta}{\\theta}\\, dt + \\sqrt{\\frac{2\\sigma}{\\theta}}\\, dW
```

where ``X`` has stable equilibria at ``\\pm\\bar{X}``, ``\\zeta`` is an OU process
with stationary variance ``\\sigma`` and correlation time ``\\theta``, and
``\\bar{g} = \\sqrt{2D}`` if ``D`` is the diffusivity.

For `θ = 0.0`, the ``\\zeta`` state is dropped entirely and `X` instead solves the
directly white-noise-driven SDE

```math
dX = \\left[-\\gamma \\frac{X}{\\bar{X}}\\left(|X| - \\bar{X}\\right)\\right] dt
     + \\bar{g}\\sqrt{2\\sigma}\\, dW,
```

i.e. the small-correlation-time limit of the system above, with `σ` here playing the
role of the limiting product ``\\theta\\sigma`` (see the `σ` argument below).

# Arguments
- `γ`      : relaxation rate (Myr⁻¹; default `2.3`)
- `X̄`      : equilibrium dipole magnitude (default `1.0`, normalized)
- `ḡ`      : OU–dipole coupling ``\\bar{g} = \\sqrt{2D}`` (default `0.5`)
- `θ`      : OU correlation time in Myr (default `0.02`). `θ = 0.0` selects the
                white-noise limit instead of the augmented OU system — see below.
- `σ`      : stationary variance of the OU process (default `1.0`). When `θ = 0.0`,
                `σ` is instead interpreted as the limiting product ``\\theta\\sigma``
                (the finite-θ stationary variance itself is not defined at θ=0); it
                defaults to `0.5` in that case, matching the package's `θσ = 0.5`
                white-noise-correspondence convention (e.g. `θ₀=0.5/1000, σ₀=1/(2θ₀)`
                used as the reference model elsewhere satisfies exactly `θ₀σ₀ = 0.5`).
- `tspan`  : time span in Ma, negative = past (default `(-2.0, 0.0)`)
- `saveat` : output step in Ma (default `0.001`, i.e. 1 kyr)
- `X0`     : initial ``X``; defaults to ``+\\bar{X}``
- `z0`     : initial ``\\zeta``; defaults to a draw from ``\\mathcal{N}(0,\\sigma)``
- `seed`   : integer random seed (`nothing` = no seeding)
- `detect_reversals` : if `true`, detect polarity reversals during integration and
                   store their times (Ma) in the returned dataset's `reversals` field
- `criterion` : reversal criterion, see `reversal_callback` (default `:zero_crossing`)
- `h0`        : threshold for `criterion = :Gammaitoni1998`, in the same normalized
                units as ``X/\\bar{X}``
- `solver`    : `:exact` (default, θ>0 only) or `:adaptive`. See below.
- `alg`, `dt`, `abstol`, `reltol`, `maxiters` : only used by `solver = :adaptive` (and
                always by the `θ = 0.0` white-noise path, which is unaffected by
                `solver`) — ignored otherwise. See below for their meaning.

# `solver`: `:exact` vs `:adaptive`

For `θ > 0`, two integrators are available:

- **`:exact` (default).** Rather than handing the stiff augmented `(X,ζ)` system to a
  general adaptive SDE solver, this integrates `X` with a Strang-split scheme whose
  noise sub-step is an EXACT joint Gaussian sample of `(∫ζ ds, ζ_{t+h})` over each step
  (the OU process's transition kernel is Gaussian at any step size, not just small `h`)
  — so the step size `h` is chosen purely from the DRIFT's local accuracy needs and is
  fully decoupled from `θ`, unlike an adaptive solver whose step must also resolve
  ζ's own fast dynamics (eigenvalue `-1/θ`). Validated in
  `examples/tests/exact_ou_noise_test.jl` (variance/strong-convergence/stationary-
  density/θ-independent-reversal-rate checks) and found to disagree sharply with
  `:adaptive` at small θ in `examples/tests/compare_current_solver_statistics.jl`: at
  this file's own physical defaults, the two solvers agree at the physical θ=0.5 kyr
  (reversal-rate ratio 1.27), but `:adaptive`'s rate is ~2x too high at θ=0.01 kyr, ~6x
  at θ=0.001 kyr, and ~42x at θ=0.0001 kyr, while `:exact` stays flat (within ~15%)
  across that whole range — matching the analytically-expected θ-independence. `alg`,
  `dt`, `abstol`, `reltol`, `maxiters` don't apply to this path (the step size is
  chosen internally; see the source for the exact bounds used).
- **`:adaptive`.** The original implementation: builds an `SDEProblem` for the full
  `(X,ζ)` system and calls `solve` (StochasticDiffEq.jl), defaulting to SciML's
  automatic algorithm selection (currently `LambaEM`) unless `alg`/`dt`/`abstol`/
  `reltol` override it. Kept available for comparison and as a fallback.

For `θ = 0.0` specifically (always via the equivalent of `:adaptive`'s machinery,
regardless of `solver`): SciML's automatic algorithm selection was checked against an
independent analytic mean-first-passage-time calculation (two methods: nested
quadrature and a finite-difference boundary-value solve, both giving the same answer)
and found to be systematically biased — about 24% too many reversals at default
tolerance — while fixed-step `EM()` with a small `dt` converges to the correct
(analytically-confirmed) answer. So unless the caller explicitly passes `alg` and/or
`dt`, `θ = 0.0` automatically uses `alg = EM()` with `dt = 1/(1000γ)` (1000 steps per
the deterministic relaxation time `1/γ`) instead of the (here, wrong) automatic choice.
"""
function simulate_Davis2021(;
    γ      :: Float64 = 100.0, # Myr^{-1} = 0.1 kyr{-1}
    X̄      :: Float64 = 5.23e22,
    ḡ      :: Float64 = sqrt(680 * 1e44),
    θ      :: Float64 = 0.5 / 1000, # Myr
    σ      :: Union{Float64, Nothing} = nothing,
    tspan              = (-2.0, 0.0),
    saveat             = 0.001,
    X0     :: Union{Nothing, Real} = nothing,
    z0     :: Union{Nothing, Real} = nothing,
    seed   :: Union{Nothing, Integer} = nothing,
    detect_reversals :: Bool = false,
    criterion :: Symbol = :Gammaitoni1998,
    h0 = nothing,
    solver :: Symbol = :exact,
    maxiters :: Int = 100_000,
    alg = nothing,
    dt = nothing,
    abstol = nothing,
    reltol = nothing,
)
    white_noise = θ == 0.0

    σ = if isnothing(σ)
        white_noise ? 0.5 : 1 / (2 * θ)
    else
        σ
    end

    θ >= 0.0 || error("θ must be nonnegative.")
    σ > 0.0 || error("σ must be strictly positive.")

    if white_noise
        return _simulate_Davis2021_adaptive(; γ = γ, X̄ = X̄, ḡ = ḡ, θ = θ, σ = σ,
            tspan = tspan, saveat = saveat, X0 = X0, z0 = z0, seed = seed,
            detect_reversals = detect_reversals, criterion = criterion, h0 = h0,
            maxiters = maxiters, alg = alg, dt = dt, abstol = abstol, reltol = reltol)
    elseif solver === :exact
        return _simulate_Davis2021_exact(; γ = γ, X̄ = X̄, ḡ = ḡ, θ = θ, σ = σ,
            tspan = tspan, saveat = saveat, X0 = X0, z0 = z0, seed = seed,
            detect_reversals = detect_reversals, criterion = criterion, h0 = h0)
    elseif solver === :adaptive
        return _simulate_Davis2021_adaptive(; γ = γ, X̄ = X̄, ḡ = ḡ, θ = θ, σ = σ,
            tspan = tspan, saveat = saveat, X0 = X0, z0 = z0, seed = seed,
            detect_reversals = detect_reversals, criterion = criterion, h0 = h0,
            maxiters = maxiters, alg = alg, dt = dt, abstol = abstol, reltol = reltol)
    else
        throw(ArgumentError("Unknown solver :$solver. Supported: :exact, :adaptive"))
    end
end

# ======================================================================================
# Internal utilities
# ======================================================================================

function _davis2021_drift!(du, u, p, _)
    X, ζ = u[1], u[2]
    du[1] = -p.γ * (X / p.X̄) * (abs(X) - p.X̄) + p.ḡ * ζ
    du[2] = -ζ / p.θ
    return nothing
end

function _davis2021_diffusion!(G, _, p, _)
    G[1, 1] = 0.0
    G[2, 1] = sqrt(2.0 * p.σ / p.θ)
    return nothing
end

# θ = 0 (white-noise limit): as θ → 0 with the product θσ held fixed, the OU-driven term
# ḡζ dt converges (its time integral converges to a scaled Wiener process — the standard
# small-correlation-time limit of an OU process) to a direct white-noise term
# ḡ√(2σ) dW on X itself, with no separate ζ state. Here `σ` is reinterpreted as that
# limiting product θσ (not the finite-θ stationary variance) — see `simulate_Davis2021`.
function _davis2021_drift_white!(du, u, p, _)
    X = u[1]
    du[1] = -p.γ * (X / p.X̄) * (abs(X) - p.X̄)
    return nothing
end

function _davis2021_diffusion_white!(G, _, p, _)
    # G[1] = p.ḡ * sqrt(2.0 * p.σ)
    G[1] = p.ḡ * sqrt(2.0 * p.σ)
    return nothing
end

# ======================================================================================
# solver = :exact — exact-noise-increment integrator for the augmented (X,ζ) system,
# θ > 0 only. Replaces handing (X,ζ) to a general adaptive SDE solver (LambaEM, see
# `_simulate_Davis2021_adaptive` below) with a scheme built specifically for this
# structure: ζ is an Ornstein–Uhlenbeck process, so over any step of length h, the pair
# (I_h, ζ_{t+h}) — where I_h = ∫_t^{t+h} ζ_s ds is what actually moves X — is EXACTLY
# jointly Gaussian, sampled here via its closed-form moments rather than by numerically
# resolving ζ's own fast dynamics (eigenvalue -1/θ), which is what made the adaptive
# solver's step size (and hence cost/accuracy) blow up as θ→0.
#
# Validated in examples/tests/exact_ou_noise_test.jl (variance sanity checks, strong
# convergence, stationary-density recovery, θ-independent reversal-rate convergence to
# the analytic Kramers rate) and examples/tests/compare_current_solver_statistics.jl
# (direct head-to-head against `solver=:adaptive` at this file's own physical defaults:
# the two agree at the physical θ=0.5 kyr — ratio 1.27 — but the adaptive solver's
# reversal rate is 2x too high at θ=0.01 kyr, 6x at θ=0.001 kyr, and 42x at θ=0.0001 kyr,
# while this exact scheme stays flat to within ~15% across that entire range).
# ======================================================================================

"""Drift v(x) = -γ(x/X̄)(|x|-X̄), stable equilibria at ±X̄."""
_davis2021_exact_v(x, γ, X̄) = -γ * (x / X̄) * (abs(x) - X̄)

"""
    _davis2021_exact_drift_flow(x, h, γ, X̄) -> x(t+h)

EXACT solution of dx/dt = v(x) alone over an interval of length h — for x>0 (and, by
v's odd symmetry, y=|x| when x<0), this is exactly the logistic ODE dy/dt=γy(1-y/X̄),
with closed form y(t) = X̄·y0·e^{γt} / (X̄+y0(e^{γt}-1)). Used for the Strang-split
drift half-steps below; being exact removes drift-flow discretization error entirely.
"""
function _davis2021_exact_drift_flow(x::Float64, h::Float64, γ::Float64, X̄::Float64)
    x == 0.0 && return 0.0
    s  = sign(x)
    y0 = abs(x)
    E  = exp(γ * h)
    return s * (X̄ * y0 * E) / (X̄ + y0 * (E - 1))
end

"""
    _davis2021_exact_ou_bracket(h, θ) -> Float64

The BRACKET term in Var(I_h) = 2σθ·BRACKET, computed via the closed form for
u=h/θ ≥ 1e-3, or the leading asymptotic series (catastrophic-cancellation-safe) for
u < 1e-3: BRACKET = θ·(u³/3 - u⁴/4 + 7u⁵/60).
"""
function _davis2021_exact_ou_bracket(h::Float64, θ::Float64)
    u = h / θ
    if u < 1e-3
        return θ * (u^3 / 3 - u^4 / 4 + 7 * u^5 / 60)
    else
        E1 = exp(-h / θ)
        E2 = E1 * E1
        return h - 2θ * (1 - E1) + (θ / 2) * (1 - E2)
    end
end

"""
    _davis2021_exact_ou_step(ζ, h, θ, σ) -> (I_h, ζ_next)

Exact joint sample of (I_h, ζ_{t+h}) given ζ_t=ζ, for the OU process with stationary
variance σ and correlation time θ (`d ζ = -ζ/θ dt + √(2σ/θ) dW`). `I_h = ∫_t^{t+h} ζ_s ds`.
"""
function _davis2021_exact_ou_step(ζ::Float64, h::Float64, θ::Float64, σ::Float64)
    E1 = exp(-h / θ)
    E2 = E1 * E1

    μ_I   = θ * ζ * (1 - E1)
    μ_ζ   = ζ * E1
    varI  = 2σ * θ * _davis2021_exact_ou_bracket(h, θ)
    varζ  = σ * (1 - E2)
    covIζ = σ * θ * (1 - E1)^2

    z1, z2 = randn(), randn()
    sqrtVarI = sqrt(varI)
    I_h = μ_I + sqrtVarI * z1
    resid_var = max(varζ - covIζ^2 / varI, 0.0)   # clamp tiny negative roundoff, no NaN
    ζ_next = μ_ζ + (covIζ / sqrtVarI) * z1 + sqrt(resid_var) * z2

    return I_h, ζ_next
end

"""
    _davis2021_exact_interp(xs, ys, xq) -> Vector{Float64}

Linear interpolation of (xs,ys) onto query points xq (clamped at the ends), used to
resample the fine internal trajectory onto the caller's requested `saveat` grid.
"""
function _davis2021_exact_interp(xs::Vector{Float64}, ys::Vector{Float64}, xq::AbstractVector)
    out = Vector{Float64}(undef, length(xq))
    for (i, x) in enumerate(xq)
        if x <= xs[1]
            out[i] = ys[1]
        elseif x >= xs[end]
            out[i] = ys[end]
        else
            j = searchsortedlast(xs, x)
            x0, x1 = xs[j], xs[j + 1]
            y0, y1 = ys[j], ys[j + 1]
            out[i] = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        end
    end
    return out
end

"""
    _davis2021_exact_reversals(t, x; criterion, h0) -> Vector{Float64}

Detect reversals on an already-computed, fine (t, x=X/X̄) trajectory, replicating
`reversal_callback`'s `:zero_crossing`/`:Gammaitoni1998` semantics via linear
interpolation between consecutive fine-grid samples for the crossing time (equivalent
to `ContinuousCallback`'s root-finding on continuous dense output, since the fine step
is small by construction — see `_simulate_Davis2021_exact`).
"""
function _davis2021_exact_reversals(t::Vector{Float64}, x::Vector{Float64};
                                     criterion::Symbol, h0 = nothing)
    reversals = Float64[]
    n = length(t)
    if criterion === :zero_crossing
        for i in 2:n
            if (x[i - 1] > 0) != (x[i] > 0)
                frac = x[i - 1] / (x[i - 1] - x[i])
                push!(reversals, t[i - 1] + frac * (t[i] - t[i - 1]))
            end
        end
    elseif criterion === :Gammaitoni1998
        h0 === nothing && throw(ArgumentError("criterion=:Gammaitoni1998 requires h0"))
        h0 > 0 || throw(ArgumentError("h0 must be positive, got $h0"))
        state = x[1] >= 0 ? :normal : :reverse
        for i in 2:n
            if state === :reverse && x[i - 1] < h0 && x[i] >= h0
                frac = (h0 - x[i - 1]) / (x[i] - x[i - 1])
                push!(reversals, t[i - 1] + frac * (t[i] - t[i - 1]))
                state = :normal
            elseif state === :normal && x[i - 1] > -h0 && x[i] <= -h0
                frac = (-h0 - x[i - 1]) / (x[i] - x[i - 1])
                push!(reversals, t[i - 1] + frac * (t[i] - t[i - 1]))
                state = :reverse
            end
        end
    else
        throw(ArgumentError("Unknown criterion :$criterion. Supported: :zero_crossing, :Gammaitoni1998"))
    end
    return reversals
end

function _simulate_Davis2021_exact(; γ, X̄, ḡ, θ, σ, tspan, saveat, X0, z0, seed,
                                     detect_reversals, criterion, h0)
    seed !== nothing && Random.seed!(seed)

    σ0_sq = 2σ * θ
    # Conservative step-size bounds (see examples/tests/exact_ou_noise_test.jl §5):
    # γ_eff=5γ covers excursions up to ~3X̄ (v'(x) grows away from the well), and the
    # second bound keeps the per-step noise displacement small vs the scale on which
    # v' varies. Neither bound depends on θ — that decoupling is the whole point.
    γ_eff = 5γ
    h = min(0.05 / γ_eff, 0.01 * X̄^2 / σ0_sq)

    T = tspan[2] - tspan[1]
    nsteps = max(ceil(Int, T / h), 1)
    h_actual = T / nsteps   # so the fine grid lands exactly on tspan[2]

    X0_val = X0 === nothing ? X̄ : Float64(X0)
    ζ = z0 === nothing ? sqrt(σ) * randn() : Float64(z0)
    X = X0_val

    t_fine = Vector{Float64}(undef, nsteps + 1)
    X_fine = Vector{Float64}(undef, nsteps + 1)
    t_fine[1] = tspan[1]
    X_fine[1] = X

    for n in 1:nsteps
        I_h, ζ_next = _davis2021_exact_ou_step(ζ, h_actual, θ, σ)
        Xhalf = _davis2021_exact_drift_flow(X, h_actual / 2, γ, X̄)
        Xss   = Xhalf + ḡ * I_h
        X     = _davis2021_exact_drift_flow(Xss, h_actual / 2, γ, X̄)
        ζ = ζ_next
        t_fine[n + 1] = tspan[1] + n * h_actual
        X_fine[n + 1] = X
    end

    max_excursion = maximum(abs, X_fine) / X̄
    if max_excursion > 3.0
        @warn "simulate_Davis2021 (solver=:exact): trajectory reached |X|/X̄ = " *
              "$(round(max_excursion, digits = 2)), beyond the 3x envelope assumed when " *
              "picking the step size h — results may be under-resolved."
    end

    x_norm_fine = X_fine ./ X̄
    reversals = detect_reversals ?
        _davis2021_exact_reversals(t_fine, x_norm_fine; criterion = criterion, h0 = h0) :
        Float64[]

    saveat_vec = saveat isa AbstractVector ?
        collect(Float64, saveat) :
        collect(range(tspan[1], tspan[2]; step = saveat))
    X_out = _davis2021_exact_interp(t_fine, x_norm_fine, saveat_vec)

    return SynthesisDataset(:Davis2021, saveat_vec, SignedPaleointensity.(X_out), reversals)
end

# ======================================================================================
# solver = :adaptive — the original implementation: hands the augmented (X,ζ) system to
# a general SciML adaptive SDE solver (LambaEM by default). Kept as an option (not the
# default for θ>0 any more — see `simulate_Davis2021`) since it's still what θ=0.0 uses,
# and for direct comparison against `solver=:exact`.
# ======================================================================================
function _simulate_Davis2021_adaptive(; γ, X̄, ḡ, θ, σ, tspan, saveat, X0, z0, seed,
                                        detect_reversals, criterion, h0,
                                        maxiters, alg, dt, abstol, reltol)
    white_noise = θ == 0.0

    # At θ=0, SciML's automatic algorithm selection was checked against an independent
    # analytic mean-first-passage-time calculation and found to systematically
    # over-count reversals (~24% too short a mean chron length at default tolerance);
    # fixed-step EM() with a small dt converges to the correct answer instead. So unless
    # the caller explicitly overrides alg/dt, use that verified-safe combination here —
    # 1000 steps per the deterministic relaxation time 1/γ, matching what was validated.
    if white_noise
        alg === nothing && (alg = EM())
        dt  === nothing && (dt = 1 / (1000 * γ))
    end

    seed !== nothing && Random.seed!(seed)

    # Fixed-step dt (from the θ=0 auto-default above, or a user-supplied dt for any
    # algorithm) needs enough steps for the whole span, regardless of the maxiters
    # default/value passed in — raise it if needed, never lower a caller's larger value.
    if dt !== nothing
        maxiters = max(maxiters, ceil(Int, (tspan[2] - tspan[1]) / dt) + 10)
    end

    p = (γ = γ, X̄ = X̄, ḡ = ḡ, θ = θ, σ = σ)

    saveat_vec      = saveat isa AbstractVector ?
        saveat :
        collect(range(tspan[1], tspan[2]; step = saveat))
    tspan_internal  = (0.0, tspan[2] - tspan[1])
    saveat_internal = saveat_vec .- tspan[1]

    X0_val = X0 === nothing ? X̄ : Float64(X0)
    u0     = if white_noise
        Float64[X0_val]
    else
        z0_val = z0 === nothing ? sqrt(σ) * randn() : Float64(z0)
        Float64[X0_val, z0_val]
    end

    cb, reversals_internal, last_t = detect_reversals ?
        reversal_callback(u -> u[1] / X̄, u0, tspan_internal[1]; criterion = criterion, h0 = h0) :
        (nothing, Float64[], Ref(0.0))

    prob = if white_noise
        SDEProblem(_davis2021_drift_white!, _davis2021_diffusion_white!, u0, tspan_internal, p)
    else
        SDEProblem(_davis2021_drift!, _davis2021_diffusion!, u0, tspan_internal, p;
                  noise_rate_prototype = zeros(2, 1))
    end

    solve_kwargs = Dict{Symbol, Any}(:saveat => saveat_internal, :callback => cb,
                                      :maxiters => maxiters)
    dt     !== nothing && (solve_kwargs[:dt] = dt)
    abstol !== nothing && (solve_kwargs[:abstol] = abstol)
    reltol !== nothing && (solve_kwargs[:reltol] = reltol)

    sol  = alg === nothing ?
        solve(prob; solve_kwargs...) :
        solve(prob, alg; solve_kwargs...)

    n = length(sol.u)
    if n < length(saveat_vec)
        @warn "simulate_Davis2021: solver returned $n / $(length(saveat_vec)) points (retcode = $(sol.retcode))"
    end

    X_vals = [u[1] / X̄ for u in sol.u]

    # invert tspan_internal = (0, tspan[2] - tspan[1])
    to_ma     = t -> t + tspan[1]
    reversals = reversals_internal .+ tspan[1]
    time_out, X_out = _patched_series(saveat_vec, X_vals, n, detect_reversals, last_t, to_ma)
    return SynthesisDataset(:Davis2021, time_out, SignedPaleointensity.(X_out), reversals)
end
