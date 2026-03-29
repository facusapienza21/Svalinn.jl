using ComplexityMeasures

"""
    ordinal_entropy(x::AbstractVector; d::Int=5, τ::Int=1) -> Float64

Compute the normalized permutation entropy H ∈ [0, 1] of a uniformly-sampled
discrete time series `x`. Only the values of `x` are used — no time axis is needed.

- `d` : embedding dimension (pattern length), must be ≥ 2
- `τ` : embedding delay

H = 0 implies a perfectly regular (periodic) signal.
H = 1 implies a maximally random signal.
"""
function ordinal_entropy(x::AbstractVector; d::Int=5, τ::Int=1)
    if length(x) < factorial(d + 1)
        @warn "Time series length ($(length(x))) is shorter than (d+1)! = $(factorial(d+1)) for d=$d. Entropy estimate may be unreliable."
    end
    ordinal_patterns = OrdinalPatterns(; m=d, τ=τ)
    return information_normalized(Shannon(), ordinal_patterns, x)
end

"""
    ordinal_complexity(x::AbstractVector; d::Int=5, τ::Int=1) -> Float64

Compute the statistical complexity of a uniformly-sampled discrete time series
`x` using the Jensen-Shannon divergence from the uniform ordinal distribution.
Only the values of `x` are used — no time axis is needed.

- `d` : embedding dimension (pattern length)
- `τ` : embedding delay
"""
function ordinal_complexity(x::AbstractVector; d::Int=5, τ::Int=1)
    min_length = factorial(d + 1)
    if length(x) < min_length
        @warn "Time series length ($(length(x))) is shorter than (d+1)! = $min_length for d=$d. Complexity estimate may be unreliable."
    end
    error("Not implemented yet")
end
