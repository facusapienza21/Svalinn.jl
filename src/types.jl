"""
    AbstractData

Abstract supertype for all data types in GeoReversal.
"""
abstract type AbstractData end

"""
    AbstractPaleointensity

Abstract supertype for all paleointensity measurement types.
"""
abstract type AbstractPaleointensity end

struct AbsolutePaleointensity <: AbstractPaleointensity end
struct RelativePaleointensity <: AbstractPaleointensity end
struct SignedPaleointensity   <: AbstractPaleointensity end

"""
    SyntheticData <: AbstractData

Synthetic dataset with a time series and a vector of signed paleointensity values.
"""
struct SyntheticData <: AbstractData
    time  :: Vector{Float64}
    value :: Vector{SignedPaleointensity}
end

"""
    RawData{T<:AbstractPaleointensity} <: AbstractData

Raw dataset with a time series and a vector of paleointensity values.
The type parameter T specifies the kind of paleointensity measurement:
AbsolutePaleointensity, RelativePaleointensity, or SignedPaleointensity.
"""
struct RawData{T<:AbstractPaleointensity} <: AbstractData
    time  :: Vector{Float64}
    value :: Vector{T}
end
