using DifferentialEquations

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
    else
        throw(ArgumentError("Unknown model :$model. Supported models: :G12"))
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
