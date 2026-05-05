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

"""
    simulate_Davis2021(; γ, X̄, ḡ, θ, σ, tspan, saveat, X0, z0, seed)
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

# Arguments
- `γ`      : relaxation rate (Myr⁻¹; default `2.3`)
- `X̄`      : equilibrium dipole magnitude (default `1.0`, normalized)
- `ḡ`      : OU–dipole coupling ``\\bar{g} = \\sqrt{2D}`` (default `0.5`)
- `θ`      : OU correlation time in Myr (default `0.02`)
- `σ`      : stationary variance of the OU process (default `1.0`)
- `tspan`  : time span in Ma, negative = past (default `(-2.0, 0.0)`)
- `saveat` : output step in Ma (default `0.001`, i.e. 1 kyr)
- `X0`     : initial ``X``; defaults to ``+\\bar{X}``
- `z0`     : initial ``\\zeta``; defaults to a draw from ``\\mathcal{N}(0,\\sigma)``
- `seed`   : integer random seed (`nothing` = no seeding)
"""
function simulate_Davis2021(;
    γ      :: Float64 = 74.0, # Myr
    X̄      :: Float64 = 5.23e22,
    ḡ      :: Float64 = sqrt(109 * 10e44),
    θ      :: Float64 = 0.5 / 1000, # Myr
    σ      :: Union{Float64, Nothing} = nothing,
    tspan              = (-2.0, 0.0),
    saveat             = 0.001,
    X0     :: Union{Nothing, Real} = nothing,
    z0     :: Union{Nothing, Real} = nothing,
    seed   :: Union{Nothing, Integer} = nothing,
)
    σ = if isnothing(σ)
        1 / sqrt(2 * θ)
    else
        σ
    end

    θ > 0.0 || error("θ must be strictly positive.")
    σ > 0.0 || error("σ must be strictly positive.")
    seed !== nothing && Random.seed!(seed)

    p = (γ = γ, X̄ = X̄, ḡ = ḡ, θ = θ, σ = σ)

    saveat_vec      = saveat isa AbstractVector ?
        saveat :
        collect(range(tspan[1], tspan[2]; step = saveat))
    tspan_internal  = (0.0, tspan[2] - tspan[1])
    saveat_internal = saveat_vec .- tspan[1]

    X0_val = X0 === nothing ? X̄ : Float64(X0)
    z0_val = z0 === nothing ? sqrt(σ) * randn() : Float64(z0)
    u0     = Float64[X0_val, z0_val]

    prob = SDEProblem(_davis2021_drift!, _davis2021_diffusion!, u0, tspan_internal, p;
                      noise_rate_prototype = zeros(2, 1))
    sol  = solve(prob; saveat = saveat_internal)

    n = length(sol.u)
    if n < length(saveat_vec)
        @warn "simulate_Davis2021: solver returned $n / $(length(saveat_vec)) points"
    end

    X_vals = [u[1] / X̄ for u in sol.u]
    return SynthesisDataset(:Davis2021, saveat_vec[1:n], SignedPaleointensity.(X_vals))
end