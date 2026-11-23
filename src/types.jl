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
- `name      :: Symbol`          — dataset identifier (e.g. `:PADM2M`, `:Sint2000`)
- `time      :: Vector{Float64}` — time array in Ma
- `value     :: Vector{T}`       — paleointensity values
- `reversals :: Vector{Float64}` — times (Ma) of detected polarity reversals, empty unless
                                    the dataset was produced with reversal detection enabled
                                    (see `simulate(...; detect_reversals=true)`)
"""
struct SynthesisDataset{T<:AbstractPaleointensity} <: AbstractData
    name      :: Symbol
    time      :: Vector{Float64}
    value     :: Vector{T}
    reversals :: Vector{Float64}
end

SynthesisDataset(name::Symbol, time::Vector{Float64}, value::Vector{T}) where {T<:AbstractPaleointensity} =
    SynthesisDataset(name, time, value, Float64[])

"""
    RawData{T<:AbstractPaleointensity} <: AbstractData

Raw dataset with a time series and a vector of paleointensity values.
The type parameter `T` encodes the kind of measurement at the type level.

# Fields
- `name         :: Symbol`          — dataset identifier (e.g. `Symbol("ODP 983")`)
- `time         :: Vector{Float64}` — time array in Ma
- `value        :: Vector{T}`       — paleointensity values
- `rpi_mean     :: Float64`         — mean paleointensity value (constant across the record)
- `lat          :: Float64`         — latitude of the sediment core (degrees)
- `lon          :: Float64`         — longitude of the sediment core (degrees)
- `sample_id    :: Int`             — numeric sample identifier
- `scale_factor :: Float64`         — normalisation scale factor
"""
struct RawData{T<:AbstractPaleointensity} <: AbstractData
    name         :: Symbol
    time         :: Vector{Float64}
    value        :: Vector{T}
    rpi_mean     :: Float64
    lat          :: Float64
    lon          :: Float64
    sample_id    :: Int
    scale_factor :: Float64

    function RawData{T}(name, time, value, rpi_mean, lat, lon, sample_id, scale_factor) where T
        @assert length(time) == length(value) "time and value must have the same length"
        new(name, time, value, rpi_mean, lat, lon, sample_id, scale_factor)
    end
end

"""
    RawDataCollection{T<:AbstractPaleointensity} <: AbstractData

A named collection of `RawData{T}` datasets of the same paleointensity type.
Useful for grouping multiple records that share a common measurement kind
(e.g. a set of relative paleointensity stacks).

# Fields
- `name     :: Symbol`          — identifier for the collection
- `datasets :: Vector{RawData{T}}` — individual records
"""
struct RawDataCollection{T<:AbstractPaleointensity} <: AbstractData
    name     :: Symbol
    datasets :: Vector{RawData{T}}
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

"""
    ParticleFilterResult

All outputs of a bootstrap particle filter run.

# Fields
- `log_Z`    : estimated marginal log-likelihood log p(Y_{1:T} | θ)
- `t`        : time vector (Ma), length T
- `n_unique` : number of unique ancestor particles after resampling at each step (length T);
               drops toward 1 when the filter collapses (particle depletion)
- `X_traj`   : normalized dipole trajectories, size N × T — state of each particle at each time
- `ζ_traj`   : OU noise trajectories, size N × T
"""
struct ParticleFilterResult
    log_Z    :: Float64
    t        :: Vector{Float64}
    n_unique :: Vector{Int}
    X_traj   :: Matrix{Float64}
    ζ_traj   :: Matrix{Float64}
end
