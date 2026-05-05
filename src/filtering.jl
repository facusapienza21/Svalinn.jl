# Particle filter (sequential Monte Carlo) for the Davis2021 model.
# Uses `_davis2021_drift!` and `_davis2021_diffusion!` defined in models/Davis2021.jl.

"""
    resample_multinomial(weights, N, rng) -> Vector{Int}

Draw N ancestor indices by multinomial resampling: each index is drawn
independently from the categorical distribution defined by `weights`.
Higher variance than systematic resampling but statistically unbiased.
"""
function resample_multinomial(weights::AbstractVector{<:Real}, N::Int, rng::AbstractRNG)
    return sample(rng, 1:N, Weights(weights), N)
end

"""
    resample_systematic(weights, N, rng) -> Vector{Int}

Draw N ancestor indices by systematic resampling: a single uniform draw
u ~ U(0, 1/N) is made, then N evenly-spaced points u, u+1/N, ..., u+(N-1)/N
are mapped through the CDF of `weights`. Lower variance than multinomial
resampling while remaining unbiased.
"""
function resample_systematic(weights::AbstractVector{<:Real}, N::Int, rng::AbstractRNG)
    cw  = cumsum(weights)
    u   = (rand(rng) .+ (0:N-1)) ./ N
    idx = Vector{Int}(undef, N)
    j   = 1
    for i in 1:N
        while j < N && cw[j] < u[i]
            j += 1
        end
        idx[i] = j
    end
    return idx
end

"""
    particle_filter_loglikelihood(observations, t, θ; kwargs...) -> ParticleFilterResult

Bootstrap particle filter estimating the marginal log-likelihood
`log p(Y_{1:T} | θ, γ, X̄, ḡ, σ_ou)` under the Davis2021 SDE model.

The hidden state is `[X, ζ]` — dipole and OU noise process. The observation
model is `Y_k = X_k/X̄ + ε_k` with `ε_k ~ N(0, σ_obs²)`. Each particle is
propagated with the SDE solver from `StochasticDiffEq`. Any model parameter
can be treated as the free parameter by varying it across calls.

# Arguments
- `observations` : observed (noisy) normalized dipole values Y = X/X̄, length T
- `t`            : time vector (Ma), must be sorted in increasing order, length T
- `θ`            : OU correlation time (Myr); default `0.5/1000`

# Keyword arguments
- `N`           : number of particles (default `500`)
- `σ_obs`       : observation noise std — must match how `observations` were generated (default `0.05`)
- `γ`           : relaxation rate (Myr⁻¹; default `74.0`)
- `X̄`           : equilibrium dipole magnitude (A m²; default `5.23e22`)
- `ḡ`           : OU–dipole coupling (default `sqrt(109 * 10e44)`)
- `σ_ou`        : stationary variance of the OU process; defaults to `1/√(2θ)`
- `rng`         : random number generator (default `Random.default_rng()`)
- `ESS_threshold` : resample when ESS drops below this value; default `N/2`
- `resample`    : resampling strategy — `:systematic` (default) or `:multinomial`

# Returns
A `ParticleFilterResult` with fields `log_Z`, `t`, `n_unique`, `X_traj`, `ζ_traj`.
"""
function particle_filter_loglikelihood(
    observations :: AbstractVector{<:Real},
    t            :: AbstractVector{<:Real},
    θ            :: Float64                 = 0.5 / 1000;
    N            :: Int                     = 500,
    σ_obs        :: Float64                 = 0.05,
    γ            :: Float64                 = 74.0,
    X̄            :: Float64                 = 5.23e22,
    ḡ            :: Float64                 = sqrt(109 * 10e44),
    σ_ou         :: Union{Float64,Nothing}  = nothing,
    rng          :: AbstractRNG             = Random.default_rng(),
    ESS_threshold  :: Float64               = Inf,
    resample     :: Symbol                  = :systematic,
)
    T = length(observations)
    T == length(t) || throw(ArgumentError("observations and t must have the same length"))
    issorted(t)    || throw(ArgumentError("t must be sorted in increasing order"))
    resample ∈ (:systematic, :multinomial) || throw(ArgumentError("resample must be :systematic or :multinomial"))

    resample_fn = resample === :systematic ? resample_systematic : resample_multinomial

    _empty = ParticleFilterResult(-Inf, collect(Float64, t), zeros(Int, T),
                                  zeros(N, T), zeros(N, T))
    θ > 0.0 || return _empty

    σ_ou_val = σ_ou === nothing ? 1.0 / sqrt(2.0 * θ) : σ_ou
    # Note: parameter named σ to match _davis2021_drift!/diffusion! convention in models.jl
    p_sde = (γ = γ, X̄ = X̄, ḡ = ḡ, θ = θ, σ = σ_ou_val)

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

    log_Z    = 0.0
    n_unique = zeros(Int, T)
    X_traj   = zeros(N, T)
    ζ_traj   = zeros(N, T)

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

        isfinite(log_Z) || return ParticleFilterResult(-Inf, collect(Float64, t), n_unique, X_traj, ζ_traj)

        # ---- resampling (only when ESS drops below threshold) ---------------
        # ESS = 1 / Σwᵢ² measures particle diversity; drops toward 1 when
        # a single particle dominates. Resampling when ESS < N/2 avoids
        # unnecessary resampling (which introduces variance) when weights are uniform.
        weights ./= sum(weights)
        ESS = 1.0 / sum(w^2 for w in weights)
        if ESS < ESS_threshold
            idx         = resample_fn(weights, N, rng)
            n_unique[k] = length(unique(idx))
            X_particles = X_particles[idx]
            ζ_particles = ζ_particles[idx]
        else
            n_unique[k] = N  # no resampling — all particles still distinct
        end

        # ---- store trajectories ---------------------------------------------
        X_traj[:, k] = X_particles
        ζ_traj[:, k] = ζ_particles
    end

    return ParticleFilterResult(log_Z, collect(Float64, t), n_unique, X_traj, ζ_traj)
end
