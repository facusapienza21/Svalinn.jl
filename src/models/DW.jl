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
- `detect_reversals` : if `true`, detect polarity reversals during integration and
                   store their times (Ma) in the returned dataset's `reversals` field
- `criterion` : reversal criterion, see `reversal_callback` (default `:zero_crossing`)
- `h0`        : threshold for `criterion = :Gammaitoni1998`, in normalized units
                (`x / x̄`, same as the returned values)
"""
function simulate_DW(;
    γ::Float64      = 0.1, # kyr^{-1}
    x̄::Float64      = 5.23e22, # A m^2
    σ::Float64      = 0.34e44,
    tspan           = (-2.0, 0.0),
    saveat          = 0.001,
    detect_reversals::Bool = false,
    criterion::Symbol = :Gammaitoni1998,
    h0 = nothing,
    maxiters::Int = 100_000,
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

    cb, reversals_internal, last_t = detect_reversals ?
        reversal_callback(identity, u0, tspan_kyr[1]; criterion = criterion, h0 = h0) :
        (nothing, Float64[], Ref(0.0))

    prob = SDEProblem(f_DW, g_DW, u0, tspan_kyr, p)
    sol  = solve(prob, saveat = saveat_kyr, callback = cb, maxiters = maxiters)

    n = length(sol.u)
    # invert saveat_kyr = (saveat_vec .- tspan[1]) .* 1000.0
    to_ma = t -> t / 1000.0 + tspan[1]
    if n < length(saveat_vec)
        aborted_at = detect_reversals ? to_ma(last_t[]) : sol.t[end] / 1000 + tspan[1]
        @warn "simulate_DW: solver aborted at $aborted_at Ma (got $n / $(length(saveat_vec)) points, retcode = $(sol.retcode))"
    end

    reversals = reversals_internal ./ 1000.0 .+ tspan[1]
    time_out, u_out = _patched_series(saveat_vec, sol.u, n, detect_reversals, last_t, to_ma)
    return SynthesisDataset(:DW, time_out, SignedPaleointensity.(u_out), reversals)
end


