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