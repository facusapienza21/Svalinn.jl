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


