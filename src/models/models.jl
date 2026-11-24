"""
    simulate(; model=:G12, kwargs...) -> SynthesisDataset

Dispatch to the appropriate model simulation based on `model` and return the time
vector `t` and the dipole component `D`. Currently supported:

- `:G12` : Gissinger (2012) three-component geodynamo model (calls `simulate_G12`)

All other keyword arguments are forwarded to the underlying model function.
`saveat` accepts either a scalar (uniform spacing) or a vector of specific time points.
`h0` is the default threshold used by `criterion = :Gammaitoni1998` when
`detect_reversals = true` (see `reversal_callback`). `maxiters` (default `100_000`,
forwarded via `kwargs`, not used by `:Davis2021`) is the ODE/SDE solver's iteration
budget — a long `tspan` can exhaust it and abort the solve early (`retcode != :Success`,
reported via `@warn`); raise `maxiters` if that happens.
"""
function simulate(;
    model::Symbol = :G12,
    h0 = 0.5,
    kwargs...,
    )

    if model === :G12
        return simulate_G12(; h0 = h0, kwargs...)
    elseif model === :P09
        return simulate_P09(; h0 = h0, kwargs...)
    elseif model === :DW
        return simulate_DW(; h0 = h0, kwargs...)
    elseif model === :Davis2021
        return simulate_Davis2021(; h0 = h0, kwargs...)
    else
        throw(ArgumentError("Unknown model :$model. Supported models: :G12, :P09, :DW"))
    end
end