"""
    simulate_Davis2021(; γ, X̄, ḡ, θ, σ, tspan, saveat, X0, z0, seed,
                         detect_reversals, criterion, h0)
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
- `γ`      : relaxation rate in Myr⁻¹ (default `100.0`)
- `X̄`      : equilibrium dipole moment (default `5.23e22`)
- `ḡ`      : OU–dipole coupling ``\\bar{g} = \\sqrt{2D}`` (default `√(6.8e46)`)
- `θ`      : OU correlation time in Myr (default `0.5e-3`, i.e. 0.5 kyr). `θ = 0.0`
             selects the white-noise limit.
- `σ`      : stationary variance of the OU process (default `1/(2θ)`, so that
             ``\\theta\\sigma = 0.5``). When `θ = 0.0`, `σ` is instead interpreted as
             the limiting product ``\\theta\\sigma`` and defaults to `0.5`, so both
             defaults give the same white-noise-equivalent intensity.
- `tspan`  : time span in Ma, negative = past (default `(-2.0, 0.0)`)
- `saveat` : output step in Ma (default `0.001`, i.e. 1 kyr), or a vector of times
- `X0`     : initial ``X``; defaults to ``+\\bar{X}``
- `z0`     : initial ``\\zeta``; defaults to a draw from ``\\mathcal{N}(0,\\sigma)``
             (ignored when `θ = 0.0`)
- `seed`   : integer random seed (`nothing` = no seeding)
- `detect_reversals` : if `true`, detect polarity reversals on the internal fine grid
                   and store their times (Ma) in the returned dataset's `reversals` field
- `criterion` : `:zero_crossing` or `:Gammaitoni1998` (default; hysteresis with
                threshold `h0`, see `reversal_callback`)
- `h0`        : threshold for `criterion = :Gammaitoni1998`, in units of ``X/\\bar{X}``.
                No default here; the `simulate` dispatcher supplies `h0 = 0.5`.
- `dt`        : internal integration step in Ma (default `0.01/γ`, i.e. 1% of the
                determinisstic relaxation time). This is a maximum: each interval between
                consecutive `saveat` times is split into equal steps no longer than `dt`.

# Integrator

`X` is advanced with a Strang splitting: an exact half-step of the drift flow (closed-form
logistic solution), a full noise step, and another exact drift half-step. The noise step
is exact at any step size:

- `θ > 0`: the pair ``(\\int_t^{t+dt} \\zeta\\,ds,\\ \\zeta_{t+dt})`` is sampled from its
  exact joint Gaussian transition (Gillespie, 1996), so `dt` does not need to resolve `θ`.
  Exact integration of the OU process is possible for any time step, and direct
  numerical integration of the augmented `(X, ζ)` system should be avoided: it produces
  artificial spikes (spurious reversals) in the limit `θ → 0`.
- `θ = 0`: the increment is ``\\bar{g}\\sqrt{2\\sigma\\, dt}\\, \\mathcal{N}(0,1)``.

The only discretization error is the splitting error between drift and noise, controlled
by `γ·dt`. It does not depend on `θ`: the variance of the noise kick per step is at most
its white-noise value ``\\bar{g}^2\\, 2\\sigma\\theta\\, dt`` for any `θ`.

# References
- Gillespie, D. T. (1996). Exact numerical simulation of the Ornstein–Uhlenbeck process
  and its integral. *Physical Review E*, 54(2), 2084–2091.
  https://doi.org/10.1103/PhysRevE.54.2084
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
    dt :: Union{Nothing, Real} = nothing,
)
    θ >= 0.0 || error("θ must be nonnegative.")
    dt = isnothing(dt) ? 0.01 / γ : Float64(dt)
    dt > 0.0 || error("dt must be strictly positive.")

    σ = if isnothing(σ)
        θ == 0.0 ? 0.5 : 1 / (2 * θ)
    else
        σ
    end
    σ > 0.0 || error("σ must be strictly positive.")

    return _simulate_Davis2021_exact(; γ = γ, X̄ = X̄, ḡ = ḡ, θ = θ, σ = σ,
        tspan = tspan, saveat = saveat, X0 = X0, z0 = z0, seed = seed,
        detect_reversals = detect_reversals, criterion = criterion, h0 = h0, dt = dt)
end

# ======================================================================================
# Exact-noise-increment integrator. ζ is an Ornstein–Uhlenbeck process, so over any step
# of length dt the pair (I_dt, ζ_{t+dt}) — where I_dt = ∫_t^{t+dt} ζ_s ds is what actually
# moves X — is EXACTLY jointly Gaussian, sampled here via its closed-form moments rather
# than by numerically resolving ζ's own fast dynamics (eigenvalue -1/θ). For θ = 0 the
# noise increment is simply Gaussian with variance 2σ·dt (σ = θσ product).
# ======================================================================================

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
    _ou_bracket(u) -> Float64

Dimensionless bracket in the variance of the OU integral (see `_exact_ou_step`),

        Σ₂₂ = cθ³ ⋅ _ou_bracket(δt / θ),   _ou_bracket(u) = u - 2(1 - e^{-u}) + 1/2 (1 - e^{-2u}).

For small u the terms of the closed form are O(1) but cancel down to O(u³), so in floating
point the closed form loses all precision (and can turn negative). For u < 1e-3 the Taylor
series u³/3 - u⁴/4 + 7u⁵/60 is used instead; the omitted terms are O(u⁶) < 1e-18.
"""
function _ou_bracket(u::Float64)
    if u < 1e-3
        return u^3 / 3 - u^4 / 4 + 7 * u^5 / 60
    else
        α = exp(-u)
        return u - 2 * (1 - α) + (1 - α^2) / 2
    end
end

"""
    _exact_ou_step(ζ::Float64, δt::Float64, θ::Float64, c::Float64) -> (ζ_next, ΔI)

Exact update over a step `δt` of an OU process ζ_t described by the SDE

        dζ_t = - ζ_t / θ dt + √c dW_t

together with its integral I_t = ∫₀ᵗ ζ_s ds (Gillespie, 1996). For any δt > 0, the pair
(ζ_{t+δt}, I_{t+δt}) conditional on (ζ_t, I_t) is normal N(μ, Σ) with α = exp(- δt / θ) and

        μ = [ ζ_t ⋅ α ,  I_t + θ ⋅ ζ_t ⋅ (1 - α) ]
        Σ = [ (cθ / 2) ⋅ (1 - α^2)       (cθ^2 / 2) ⋅ (1 - α)^2
              (cθ^2 / 2) ⋅ (1 - α)^2     cθ^3 [ δt / θ - 2(1 - α) + 1/2 (1 - α^2) ] ]

The updated (ζ_{t+δt}, I_{t+δt}) is then generated as

        (ζ_{t+δt}, I_{t+δt}) = μ + L ⋅ n

where n = (n₁, n₂) are two independent standard normals and L is the Cholesky factor of
Σ (Σ = L Lᵀ):

        L = [ √Σ₁₁               0
              Σ₁₂ / √Σ₁₁    √(Σ₂₂ - Σ₁₂² / Σ₁₁) ]

That is, ζ_{t+δt} is drawn from its marginal distribution and I_{t+δt} from its
distribution conditional on ζ_{t+δt}; the shared n₁ carries their correlation.

Σ₂₂ is computed as cθ³ ⋅ `_ou_bracket(δt / θ)`, which avoids catastrophic cancellation
in the closed form when δt ≪ θ.

Returns ζ_{t+δt} and the increment ΔI = I_{t+δt} - I_t (independent of I_t).

# References
- Gillespie, D. T. (1996). Exact numerical simulation of the Ornstein–Uhlenbeck process
  and its integral. *Physical Review E*, 54(2), 2084–2091.
  https://doi.org/10.1103/PhysRevE.54.2084
"""
function _exact_ou_step(ζ::Float64, δt::Float64, θ::Float64, c::Float64)
    α = exp(-δt / θ)

    # Mean of (ζ_{t+δt}, ΔI)
    μ_ζ = ζ * α
    μ_I = θ * ζ * (1 - α)

    # Covariance Σ
    Σ11 = (c * θ / 2) * (1 - α^2)
    Σ12 = (c * θ^2 / 2) * (1 - α)^2
    Σ22 = c * θ^3 * _ou_bracket(δt / θ)

    # Cholesky factor L of Σ
    L11 = sqrt(Σ11)
    L21 = Σ12 / L11
    L22 = sqrt(max(Σ22 - L21^2, 0.0))   # clamp tiny negative roundoff

    n1, n2 = randn(), randn()
    ζ_next = μ_ζ + L11 * n1
    ΔI     = μ_I + L21 * n1 + L22 * n2

    return ζ_next, ΔI
end


"""
    _davis2021_time_stops(tspan, saveat_vec, dt) -> Vector{Float64}

Integration grid from `tspan[1]` to `tspan[2]` that contains every time in `saveat_vec`
EXACTLY. Each interval between consecutive nodes `[tspan[1]; saveat_vec; tspan[2]]` is
split into `ceil(Δ/dt)` equal sub-steps, so no step is longer than `dt`. Both the drift
flow and the noise increment are exact for any step length, so the sub-steps do not need
to be uniform across intervals.
"""
function _davis2021_time_stops(tspan, saveat_vec::Vector{Float64}, dt::Float64)
    nodes = unique([Float64(tspan[1]); saveat_vec; Float64(tspan[2])])
    t_stops = [nodes[1]]
    for k in 2:length(nodes)
        a, b = nodes[k - 1], nodes[k]
        n = ceil(Int, (b - a) / dt)
        for j in 1:(n - 1)
            push!(t_stops, a + j * (b - a) / n)
        end
        push!(t_stops, b)          # land exactly on the node
    end
    return t_stops
end

"""
    _davis2021_reversal_step!(reversals, state, t0, x0, t1, x1; criterion, h0) -> state

Check the step `(t0, x0) → (t1, x1)` of the normalized trajectory `x = X/X̄` for a
reversal, push its time onto `reversals` if found, and return the updated polarity
`state` (`:normal`/`:reverse`, only used by `:Gammaitoni1998`). The crossing time is
placed by linear interpolation within the step. Semantics match `reversal_callback`:

- `:zero_crossing`: every sign change of `x`.
- `:Gammaitoni1998`: hysteresis — `:normal` once `x` rises through `+h0`, `:reverse` once
  it falls through `-h0`.
"""
function _davis2021_reversal_step!(reversals::Vector{Float64}, state::Symbol,
                                   t0::Float64, x0::Float64, t1::Float64, x1::Float64;
                                   criterion::Symbol, h0)
    if criterion === :zero_crossing
        if (x0 > 0) != (x1 > 0)
            push!(reversals, t0 + x0 / (x0 - x1) * (t1 - t0))
        end
    elseif state === :reverse && x0 < h0 && x1 >= h0
        push!(reversals, t0 + (h0 - x0) / (x1 - x0) * (t1 - t0))
        state = :normal
    elseif state === :normal && x0 > -h0 && x1 <= -h0
        push!(reversals, t0 + (-h0 - x0) / (x1 - x0) * (t1 - t0))
        state = :reverse
    end
    return state
end

function _simulate_Davis2021_exact(; γ, X̄, ḡ, θ, σ, tspan, saveat, X0, z0, seed,
                                     detect_reversals, criterion, h0, dt)
    if detect_reversals
        criterion ∈ (:zero_crossing, :Gammaitoni1998) || throw(ArgumentError(
            "Unknown criterion :$criterion. Supported: :zero_crossing, :Gammaitoni1998"))
        if criterion === :Gammaitoni1998
            h0 === nothing && throw(ArgumentError("criterion=:Gammaitoni1998 requires h0"))
            h0 > 0 || throw(ArgumentError("h0 must be positive, got $h0"))
        end
    end

    # Output times: must be sorted and inside tspan
    saveat_vec = saveat isa AbstractVector ?
        collect(Float64, saveat) :
        collect(range(tspan[1], tspan[2]; step = saveat))
    issorted(saveat_vec) || throw(ArgumentError("saveat must be sorted"))
    (isempty(saveat_vec) || (saveat_vec[1] >= tspan[1] && saveat_vec[end] <= tspan[2])) ||
        throw(ArgumentError("saveat must lie within tspan = $tspan"))
    saveat_vec = unique(saveat_vec)

    # Integration grid: steps of at most dt, landing exactly on every saveat time
    t_stops = _davis2021_time_stops(tspan, saveat_vec, dt)

    seed !== nothing && Random.seed!(seed)
    white_noise = θ == 0.0
    # OU forcing dζ = -ζ/θ dt + √(2σ/θ) dW, i.e. diffusion constant c = 2σ/θ in `_exact_ou_step`
    c = white_noise ? 0.0 : 2σ / θ

    # Initial state: X (dipole) and ζ (OU forcing; unused in the white-noise limit)
    X = X0 === nothing ? X̄ : Float64(X0)
    ζ = white_noise ? 0.0 : (z0 === nothing ? sqrt(σ) * randn() : Float64(z0))

    X_out     = Vector{Float64}(undef, length(saveat_vec))   # X/X̄ at saveat times
    i_save    = 1                                            # next saveat index to fill
    reversals = Float64[]
    state     = X >= 0 ? :normal : :reverse

    if i_save <= length(saveat_vec) && saveat_vec[i_save] == t_stops[1]
        X_out[i_save] = X / X̄
        i_save += 1
    end

    for k in 2:length(t_stops)
        t0, t1 = t_stops[k - 1], t_stops[k]
        Δt = t1 - t0
        x0 = X / X̄

        # noise increment I = ∫_{t0}^{t1} ζ ds (exact for any Δt)
        if white_noise
            I = sqrt(2σ * Δt) * randn()       # σ = θσ product at θ = 0
        else
            ζ, I = _exact_ou_step(ζ, Δt, θ, c)
        end
        # Strang splitting: exact drift half-step, noise kick, exact drift half-step
        Xhalf = _davis2021_exact_drift_flow(X, Δt / 2, γ, X̄)
        Xss   = Xhalf + ḡ * I
        X     = _davis2021_exact_drift_flow(Xss, Δt / 2, γ, X̄)

        if detect_reversals
            state = _davis2021_reversal_step!(reversals, state, t0, x0, t1, X / X̄;
                                              criterion = criterion, h0 = h0)
        end
        if i_save <= length(saveat_vec) && saveat_vec[i_save] == t1
            X_out[i_save] = X / X̄
            i_save += 1
        end
    end
    @assert i_save == length(saveat_vec) + 1 "not every saveat time was reached"

    return SynthesisDataset(:Davis2021, saveat_vec, SignedPaleointensity.(X_out), reversals)
end
