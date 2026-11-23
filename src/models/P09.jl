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
- `detect_reversals` : if `true`, detect polarity reversals during integration and
                   store their times (Ma) in the returned dataset's `reversals` field
- `criterion` : reversal criterion, see `reversal_callback` (default `:zero_crossing`)
- `h0`        : threshold for `criterion = :Gammaitoni1998`, in the same units as `D`
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
    detect_reversals::Bool = false,
    criterion::Symbol = :Gammaitoni1998,
    h0 = nothing,
    maxiters::Int = 100_000,
    )

    p = (a0 = a0, a1 = a1, sigma = sigma)

    f_P09(u, p, _) = p.a0 + p.a1 * sin(2u)
    g_P09(_, p, _) = p.sigma

    saveat_vec      = saveat isa AbstractVector ?
        saveat :
        collect(range(tspan[1], tspan[2]; step = saveat))
    tspan_internal  = (0.0, tspan[2] - tspan[1])
    saveat_internal = saveat_vec .- tspan[1]

    cb, reversals_internal, last_t = detect_reversals ?
        reversal_callback(u -> R * cos(u + x_offset), u0, tspan_internal[1]; criterion = criterion, h0 = h0) :
        (nothing, Float64[], Ref(0.0))

    prob = SDEProblem(f_P09, g_P09, u0, tspan_internal, p)
    sol  = solve(prob; dt = 1e-4, saveat = saveat_internal, callback = cb, maxiters = maxiters)

    u_vals = sol.u
    D      = R .* cos.(u_vals .+ x_offset)
    n = length(D)
    if n < length(saveat_vec)
        @warn "simulate_P09: solver returned $n / $(length(saveat_vec)) points (retcode = $(sol.retcode))"
    end

    # invert tspan_internal = (0, tspan[2] - tspan[1])
    to_ma     = t -> t + tspan[1]
    reversals = reversals_internal .+ tspan[1]
    time_out, D_out = _patched_series(saveat_vec, D, n, detect_reversals, last_t, to_ma)
    return SynthesisDataset(:P09, time_out, SignedPaleointensity.(D_out), reversals)
end