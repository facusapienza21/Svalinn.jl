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

"""
    AbsolutePaleointensity

A single absolute paleointensity measurement (always non-negative).

# Fields
- `value :: Float64` — absolute dipole moment value (≥ 0)
"""
struct AbsolutePaleointensity <: AbstractPaleointensity
    value :: Float64
    function AbsolutePaleointensity(value::Float64)
        @assert value ≥ 0 "AbsolutePaleointensity value must be non-negative, got $value"
        new(value)
    end
end

"""
    RelativePaleointensity

A single relative paleointensity measurement, normalised against a reference record.

# Fields
- `value :: Float64` — normalised paleointensity value
"""
struct RelativePaleointensity <: AbstractPaleointensity
    value :: Float64
end

"""
    SignedPaleointensity

A single signed paleointensity measurement (positive = normal polarity, negative = reversed).
"""
struct SignedPaleointensity <: AbstractPaleointensity
    value :: Float64
end

"""
    SynthesisDataset{T<:AbstractPaleointensity} <: AbstractData

Synthesis dataset with a time series and a vector of paleointensity values of type `T`.
Usually this corresponds to the PADM2M or Sint2000 datasets. The default `T` is
`SignedPaleointensity`; use `to_absolute` to obtain a `SynthesisDataset{AbsolutePaleointensity}`.

# Fields
- `name  :: Symbol`        — dataset identifier (e.g. `:PADM2M`, `:Sint2000`)
- `time  :: Vector{Float64}` — time array in Ma
- `value :: Vector{T}`     — paleointensity values
"""
struct SynthesisDataset{T<:AbstractPaleointensity} <: AbstractData
    name  :: Symbol
    time  :: Vector{Float64}
    value :: Vector{T}
end

"""
    RawData{T<:AbstractPaleointensity} <: AbstractData

Raw dataset with a time series and a vector of paleointensity values.
The type parameter `T` encodes the kind of measurement at the type level.

# Fields
- `name  :: Symbol`    — dataset identifier
- `time  :: Vector{Float64}` — time array in Ma
- `value :: Vector{T}`       — paleointensity values
"""
struct RawData{T<:AbstractPaleointensity} <: AbstractData
    name  :: Symbol
    time  :: Vector{Float64}
    value :: Vector{T}
end

"""
    PolarityDataset <: AbstractData

A sequence of geomagnetic polarity chrons, each defined by a start and end age.
`start` and `stop` are the boundaries of the chrons in Ma, and `polarity` indicates
whether these intervals correspond to normal or reversed polarity.

# Fields
- `start    :: Vector{Float64}` — start ages of the chrons (Ma)
- `stop     :: Vector{Float64}` — end ages of the chrons (Ma)
- `polarity :: Symbol`          — `:normal` (default) or `:reverse`
"""
struct PolarityDataset <: AbstractData
    name     :: Symbol
    start    :: Vector{Float64}
    stop     :: Vector{Float64}
    polarity :: Symbol

    function PolarityDataset(name, start, stop, polarity=:normal)
        @assert polarity ∈ (:normal, :reverse) "polarity must be :normal or :reverse"
        @assert length(start) == length(stop)  "start and stop must have the same length"
        new(name, start, stop, polarity)
    end
end
