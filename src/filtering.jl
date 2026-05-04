"""
Particle filter (sequential Monte Carlo) for the Davis2021 model.

Estimates the log-likelihood log p(Y_{1:T} | θ) for a given sequence of
observations Y and OU correlation time θ, with all other model parameters fixed.
"""

using StochasticDiffEq
using StatsBase: sample, Weights

# Private SDE functions — same equations as simulate_Davis2021 in models.jl
function _davis2021_drift!(du, u, p, _)
    X, ζ = u[1], u[2]
    du[1] = -p.γ * (X / p.X̄) * (abs(X) - p.X̄) + p.ḡ * ζ
    du[2] = -ζ / p.θ
end

function _davis2021_diffusion!(G, _, p, _)
    G[1, 1] = 0.0
    G[2, 1] = sqrt(2.0 * p.σ_ou / p.θ)
end

"""
    particle_filter_loglikelihood(
        observations, t, θ;
        N, σ_obs, γ, X̄, ḡ, σ_ou, rng
    ) -> (log_Z, t, n_unique)

Bootstrap particle filter to estimate the marginal log-likelihood
`log p(Y_{1:T} | θ)` under the Davis2021 SDE model.

The hidden state is `[X, ζ]` — dipole and OU noise process.  The observation
model is `Y_k = X_k/X̄ + ε_k` with `εk ~ N(0, σ_obs²)`.  Each particle is
propagated with the SDE solver from `StochasticDiffEq`.

Only `θ` (OU correlation time) is treated as a free parameter; all other
model parameters are fixed to their Davis2021 defaults.

# Arguments
- `observations` : observed (noisy) normalized dipole values Y = X/X̄ at each time step
- `t`            : corresponding time vector (Ma), must be sorted in increasing order
- `θ`            : OU correlation time (Myr) — the parameter being inferred

# Keyword arguments
- `N`     : number of particles (default `500`)
- `σ_obs` : observation noise standard deviation (default `0.05`)
- `γ`     : relaxation rate — fixed (default `74.0` Myr⁻¹)
- `X̄`     : equilibrium dipole magnitude — fixed (default `5.23e22` A m²)
- `ḡ`     : OU–dipole coupling — fixed (default `sqrt(109 * 10e44)`)
- `σ_ou`  : stationary variance of OU; defaults to `1/√(2θ)` (Davis2021 default)
- `rng`   : random number generator (default `Random.default_rng()`)

# Returns
A tuple `(log_Z, t_track, n_unique)` where:
- `log_Z`    : estimated `log p(Y_{1:T} | θ)` (`-Inf` if the filter degenerates)
- `t_track`  : time vector (same as `t`)
- `n_unique` : number of unique particles after resampling at each step — tracks
               particle depletion (drops toward 1 when the filter collapses)
"""
function particle_filter_loglikelihood(
    observations :: AbstractVector{<:Real},
    t            :: AbstractVector{<:Real},
    θ            :: Float64;
    N            :: Int                     = 500,
    σ_obs        :: Float64                 = 0.05,
    γ            :: Float64                 = 74.0,
    X̄            :: Float64                 = 5.23e22,
    ḡ            :: Float64                 = sqrt(109 * 10e44),
    σ_ou         :: Union{Float64,Nothing}  = nothing,
    rng          :: AbstractRNG             = Random.default_rng(),
)
    T = length(observations)
    T == length(t) || throw(ArgumentError("observations and t must have the same length"))
    issorted(t)    || throw(ArgumentError("t must be sorted in increasing order"))
    θ > 0.0        || return (-Inf, t, zeros(Int, length(t)))

    σ_ou_val = σ_ou === nothing ? 1.0 / sqrt(2.0 * θ) : σ_ou
    p_sde    = (γ = γ, X̄ = X̄, ḡ = ḡ, θ = θ, σ_ou = σ_ou_val)

    # Base SDEProblem — tspan and u0 are overridden per particle/step via remake
    base_prob = SDEProblem(
        _davis2021_drift!, _davis2021_diffusion!,
        zeros(2), (0.0, 1.0), p_sde;
        noise_rate_prototype = zeros(2, 1),
    )

    # ---- initialise particles -----------------------------------------------
    # X₀ = X̄  (positive equilibrium),  ζ₀ ~ N(0, σ_ou)
    X_particles = fill(X̄, N)
    ζ_particles = sqrt(σ_ou_val) .* randn(rng, N)

    log_Z    = 0.0          # accumulated log normalising constant
    n_unique = zeros(Int, T) # unique particles after resampling at each step

    for k in 1:T
        dt      = k == 1 ? (t[2] - t[1]) : (t[k] - t[k-1])
        t_start = k == 1 ? t[1] - dt      : t[k-1]
        t_end   = t[k]

        # ---- propagate all particles with the SDE solver ---------------------
        for i in 1:N
            prob_i = remake(base_prob;
                u0    = [X_particles[i], ζ_particles[i]],
                tspan = (t_start, t_end),
            )
            sol_i = solve(prob_i; saveat = [t_end])
            X_particles[i] = sol_i.u[end][1]
            ζ_particles[i] = sol_i.u[end][2]
        end

        # ---- observation weights: Y_k = X_k/X̄ + N(0, σ_obs²) ---------------
        log_weights = [-0.5 * ((observations[k] - X_particles[i] / X̄) / σ_obs)^2 for i in 1:N]
        log_weights .-= log(σ_obs) + 0.5 * log(2π)

        # Log-sum-exp trick: subtract max before exponentiating to avoid underflow.
        # Without it, all exp(log_w) can flush to 0 in Float64 (e.g. if log_w ~ -1000).
        # The max cancels out exactly:
        #   log( mean(exp(lw)) ) = log( mean(exp(lw - max_lw)) ) + max_lw
        # so log_Z is unaffected — this is purely a numerical stabilisation.
        max_lw  = maximum(log_weights)
        weights = exp.(log_weights .- max_lw)
        Z_k     = mean(weights)
        log_Z  += log(Z_k) + max_lw

        isfinite(log_Z) || return (-Inf, t, n_unique)

        # ---- systematic resampling ------------------------------------------
        weights ./= sum(weights)
        idx          = sample(rng, 1:N, Weights(weights), N)
        n_unique[k]  = length(unique(idx))
        X_particles  = X_particles[idx]
        ζ_particles  = ζ_particles[idx]
    end

    return log_Z, t, n_unique
end
