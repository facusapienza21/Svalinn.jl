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
- `detect_reversals` : if `true`, detect polarity reversals during integration and
                   store their times (Ma) in the returned dataset's `reversals` field
- `criterion`    : reversal criterion, see `reversal_callback` (default `:zero_crossing`)
- `h0`           : threshold for `criterion = :Gammaitoni1998`, in the same units as
                   the returned `D/√2` values
"""
function simulate_G12(;
    μ::Float64 = 0.119,
    ν::Float64 = 0.1,
    Γ::Float64 = 0.9,
    tspan      = (-2.0, 0.0),
    saveat     = 0.001,
    u0         = nothing,
    detect_reversals::Bool = false,
    criterion::Symbol = :zero_crossing,
    h0 = nothing,
    maxiters::Int = 100_000,
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

    cb, reversals_internal, last_t = detect_reversals ?
        reversal_callback(u -> u[2] / √2, u0, tspan_simulation[1]; criterion = criterion, h0 = h0) :
        (nothing, Float64[], Ref(0.0))

    prob = ODEProblem(G12!, u0, tspan_simulation, (μ, ν, Γ))
    sol  = solve(
        prob,
        Tsit5();
        reltol   = 1e-12,
        abstol   = 1e-12,
        saveat   = saveat_simulation,
        callback = cb,
        maxiters = maxiters,
        )

    D = Array(sol)[2, :] ./ √2
    n = length(D)
    if n < length(saveat_vec)
        @warn "simulate_G12: solver returned $n / $(length(saveat_vec)) points (retcode = $(sol.retcode))"
    end

    # invert tspan_simulation = 250 .* (tspan .- tspan[2])
    to_ma       = t -> t / 250 + tspan[2]
    reversals   = reversals_internal ./ 250 .+ tspan[2]
    time_out, D_out = _patched_series(saveat_vec, D, n, detect_reversals, last_t, to_ma)
    return SynthesisDataset(:G12, time_out, SignedPaleointensity.(D_out), reversals)
end

