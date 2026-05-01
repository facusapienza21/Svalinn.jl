using DifferentialEquations
using StochasticDiffEq
using Random

"""
    simulate(; model=:G12, kwargs...) -> SynthesisDataset

Dispatch to the appropriate model simulation based on `model` and return the time
vector `t` and the dipole component `D`. Currently supported:

- `:G12` : Gissinger (2012) three-component geodynamo model (calls `simulate_G12`)

All other keyword arguments are forwarded to the underlying model function.
`saveat` accepts either a scalar (uniform spacing) or a vector of specific time points.
"""
function simulate(;
    model::Symbol = :G12,
    kwargs...,
    )

    if model === :G12
        return simulate_G12(; kwargs...)
    elseif model === :P09
        return simulate_P09(; kwargs...)
    elseif model === :DW
        return simulate_DW(; kwargs...)
    elseif model === :Davis2021
        return simulate_Davis2021(; kwargs...)
    # elseif model === :DW2D
    #     return simulate_DW2D(; kwargs...)
    else
        throw(ArgumentError("Unknown model :$model. Supported models: :G12, :P09, :DW"))
    end
end

"""
    simulate_G12(; μ, ν, Γ, tspan, saveat, u0) -> SynthesisDataset{SignedPaleointensity}

Simulate the Gissinger (2012) three-component geodynamo model:

    dQ/dt =  μ·Q − V·D
    dD/dt = −ν·D + V·Q
    dV/dt =  Γ − V + Q·D

where Q and D are the quadrupole and dipole magnetic field components and V is
the velocity. One model time unit corresponds to 4 kyr.

D is rescaled by `1/√2` for comparison with Sint-2000. Returns a
`SynthesisDataset{SignedPaleointensity}` named `:G12`.

If `u0` is not provided, the simulation starts near the positive-polarity fixed point
with a small perturbation.

# Arguments
- `μ`, `ν`, `Γ` : model parameters (default: standard reversing regime)
- `tspan`        : integration time span in Ma, negative = past (default `(-2.0, 0.0)`)
- `saveat`       : output times in Ma — either a `Float64` step size (converted to a
                   `LinRange` spanning `tspan`) or a vector of specific time points
                   (default `0.001`, i.e. 1 kyr)
- `u0`           : initial condition `[Q₀, D₀, V₀]`; if `nothing`, uses the fixed point
"""
function simulate_G12(;
    μ::Float64 = 0.119,
    ν::Float64 = 0.1,
    Γ::Float64 = 0.9,
    tspan      = (-2.0, 0.0),
    saveat     = 0.001,
    u0         = nothing,
    )

    function G12!(du, u, p, t)
        Q, D, V = u
        μ, ν, Γ = p
        du[1] =  μ * Q - V * D
        du[2] = -ν * D + V * Q
        du[3] =  Γ - V + Q * D
    end

    if isnothing(u0)
        Qeq =  -sqrt((Γ + sqrt(μ * ν)) * sqrt(ν / μ))
        Deq =   sqrt((Γ + sqrt(μ * ν)) * sqrt(μ / ν))
        Veq = -(μ * ν)^0.5
        u0  = [Qeq + 5e-3, Deq + 5e-3, Veq + 1e-2]
    end

    saveat_vec = saveat isa AbstractVector ?
        saveat :
        collect(range(tspan[1], tspan[2]; step = saveat))

    tspan_simulation = 250 .* (tspan .- tspan[2])
    saveat_simulation = 250 .* (saveat_vec .- tspan[2])

    prob = ODEProblem(G12!, u0, tspan_simulation, (μ, ν, Γ))
    sol  = solve(
        prob,
        Tsit5();
        reltol   = 1e-12,
        abstol   = 1e-12,
        saveat   = saveat_simulation
        )

    D = Array(sol)[2, :] ./ √2
    return SynthesisDataset(:G12, saveat_vec, SignedPaleointensity.(D))
end

"""
    simulate_P09(; a1, a0, sigma, x_offset, R, tspan, saveat, u0) -> SynthesisDataset{SignedPaleointensity}

Simulate the Petrelis et al. (2009) stochastic phase model:

    du = (a0 + a1·sin(2u)) dt + σ dW

The observed dipole is reconstructed as:

    D(t) = R · cos(u(t) + x_offset)

One time unit is 1 Myr. Returns a `SynthesisDataset{SignedPaleointensity}` named `:P09`.

# Arguments
- `a1`       : nonlinear coupling (default `-185.0`)
- `a0`       : linear drift; default `0.9 * a1`
- `sigma`    : noise amplitude; default `0.2 * √|a1|`
- `x_offset` : phase offset for dipole reconstruction (default `0.3`)
- `R`        : amplitude for dipole reconstruction (default `1.3`)
- `tspan`    : time span in Ma, negative = past (default `(-2.0, 0.0)`)
- `saveat`   : output step in Ma (default `0.001`, i.e. 1 kyr)
- `u0`       : initial phase (default `0.8`)
"""
function simulate_P09(;
    a1::Float64       = -185.0,
    a0::Float64       = 0.9 * (-185.0),
    sigma::Float64    = 0.2 * sqrt(185.0),
    x_offset::Float64 = 0.3,
    R::Float64        = 1.3,
    tspan             = (-2.0, 0.0),
    saveat            = 0.001,
    u0::Float64       = 0.8,
    )

    p = (a0 = a0, a1 = a1, sigma = sigma)

    f_P09(u, p, _) = p.a0 + p.a1 * sin(2u)
    g_P09(_, p, _) = p.sigma

    saveat_vec      = saveat isa AbstractVector ?
        saveat :
        collect(range(tspan[1], tspan[2]; step = saveat))
    tspan_internal  = (0.0, tspan[2] - tspan[1])
    saveat_internal = saveat_vec .- tspan[1]

    prob = SDEProblem(f_P09, g_P09, u0, tspan_internal, p)
    sol  = solve(prob; dt = 1e-4, saveat = saveat_internal)

    u_vals = sol.u
    D      = R .* cos.(u_vals .+ x_offset)
    return SynthesisDataset(:P09, saveat_vec, SignedPaleointensity.(D))
end

"""
    simulate_DW(; α, β, σ, tspan, saveat, u0) -> SynthesisDataset{SignedPaleointensity}

Simulate the Buffett, Ziegler & Constable (2013) stochastic double-well model:

    dx = (-α·x³ + β·x) dt + σ dW

derived from the double-well potential V(x) = (α/4)x⁴ − (β/2)x².
The two stable equilibria are at x* = ±√(β/α).

Parameters are calibrated from PADM2M (Ziegler et al. 2011) as reported in
Buffett et al. (2013). Time unit is Myr; dipole unit is ZAm².
The two stable equilibria are at x* = ±√(β/α) ≈ ±1.44 ZAm².

# Arguments
- `α`      : quartic potential coefficient (default `0.0423` Myr⁻¹ (ZAm²)⁻²)
- `β`      : quadratic potential coefficient (default `0.0882` Myr⁻¹)
- `σ`      : noise amplitude (default `0.735` ZAm² Myr⁻¹/²)
- `tspan`  : time span in Ma (= Myr), negative = past (default `(-2.0, 0.0)`)
- `saveat` : output step in Ma (default `0.001`, i.e. 1 kyr)
- `u0`     : initial dipole moment in ZAm²; defaults to `+√(β/α)`
"""
function simulate_DW(;
    γ::Float64      = 0.1, # kyr^{-1}
    x̄::Float64      = 5.23e22, # A m^2
    σ::Float64      = 0.34e44,
    tspan           = (-2.0, 0.0),
    saveat          = 0.001,
    )

    saveat_vec = saveat isa AbstractVector ?
        saveat :
        collect(range(tspan[1], tspan[2]; step = saveat))

    # Parameters are in kyr; tspan/saveat are in Ma → convert to kyr
    tspan_kyr  = (0.0, (tspan[2] - tspan[1]) * 1000.0)
    saveat_kyr = (saveat_vec .- tspan[1]) .* 1000.0

    # Work in normalized units y = x / x̄ ∈ [-1, 1] at equilibria
    # dy = f̃(y) dt + σ̃ dW
    # f̃(y) = γ·y·(y - 1) for y ≥ 0,  γ·y·(y + 1) for y < 0
    # σ̃   = √(2σ) / x̄
    σ_norm = (2σ)^0.5 / x̄
    p  = (γ = γ, σ_norm = σ_norm)
    u0 = 1.0   # start at positive equilibrium (y = +1)

    function f_DW(y, p, _)
        y >= 0.0 ? p.γ * y * (1.0 - y) : p.γ * y * (y + 1.0)
    end
    g_DW(_, p, _) = p.σ_norm

    prob = SDEProblem(f_DW, g_DW, u0, tspan_kyr, p)
    sol  = solve(prob, saveat = saveat_kyr)

    n = length(sol.u)
    if n < length(saveat_vec)
        @warn "simulate_DW: solver aborted at $(sol.t[end]/1000) Ma (got $n / $(length(saveat_vec)) points)"
    end

    return SynthesisDataset(:DW, saveat_vec[1:n], SignedPaleointensity.(sol.u))
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
    γ      :: Float64 = 100.0,
    X̄      :: Float64 = 5.23e22,
    ḡ      :: Float64 = sqrt(2.0 * 340.0e44),
    θ      :: Float64 = 0.1,
    σ      :: Float64 = 500.0,
    tspan              = (-2.0, 0.0),
    saveat             = 0.001,
    X0     :: Union{Nothing, Real} = nothing,
    z0     :: Union{Nothing, Real} = nothing,
    seed   :: Union{Nothing, Integer} = nothing,
)
    θ > 0.0 || error("θ must be strictly positive.")
    σ > 0.0 || error("σ must be strictly positive.")
    seed !== nothing && Random.seed!(seed)

    function drift!(du, u, p, _)
        X, ζ = u[1], u[2]
        du[1] = -p.γ * (X / p.X̄) * (abs(X) - p.X̄) + p.ḡ * ζ
        du[2] = -ζ / p.θ
        return nothing
    end

    function diffusion!(G, _, p, _)
        G[1, 1] = 0.0
        G[2, 1] = sqrt(2.0 * p.σ / p.θ)
        return nothing
    end

    p = (γ = γ, X̄ = X̄, ḡ = ḡ, θ = θ, σ = σ)

    saveat_vec      = saveat isa AbstractVector ?
        saveat :
        collect(range(tspan[1], tspan[2]; step = saveat))
    tspan_internal  = (0.0, tspan[2] - tspan[1])
    saveat_internal = saveat_vec .- tspan[1]

    X0_val = X0 === nothing ? X̄ : Float64(X0)
    z0_val = z0 === nothing ? sqrt(σ) * randn() : Float64(z0)
    u0     = Float64[X0_val, z0_val]

    prob = SDEProblem(drift!, diffusion!, u0, tspan_internal, p;
                      noise_rate_prototype = zeros(2, 1))
    sol  = solve(prob; saveat = saveat_internal)

    n = length(sol.u)
    if n < length(saveat_vec)
        @warn "simulate_Davis2021: solver returned $n / $(length(saveat_vec)) points"
    end

    X_vals = [u[1] / X̄ for u in sol.u]
    return SynthesisDataset(:Davis2021, saveat_vec[1:n], SignedPaleointensity.(X_vals))
end
