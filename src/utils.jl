using MAT
using DelimitedFiles
using CSV
using DataFrames
using LinearAlgebra

"""
    read_synthesis(filepath::String; name::Symbol) -> SynthesisDataset

Read a `.mat` file containing a signed dipole moment time series and return a `SynthesisDataset`.
"""
function read_synthesis(filepath::String; name::Symbol)
    raw = matread(filepath)
    d   = vec(raw["d"])
    # Convert time from ka to Ma:
    t   = vec(raw["t"]) ./ 1000
    return SynthesisDataset(name, t, SignedPaleointensity.(d))
end

"""
    to_absolute(d::SynthesisDataset{SignedPaleointensity}) -> SynthesisDataset{AbsolutePaleointensity}

Convert a signed synthesis dataset to an absolute one by taking the absolute value of each
paleointensity measurement. The time axis and name are preserved.
"""
function to_absolute(d::SynthesisDataset{SignedPaleointensity})
    return SynthesisDataset(d.name, d.time, AbsolutePaleointensity.(abs.(x.value for x in d.value)))
end

"""
    to_signed(ds::SynthesisDataset{AbsolutePaleointensity},
              polarity::PolarityDataset) -> SynthesisDataset{SignedPaleointensity}

Assign a sign to each sample in `ds` based on a polarity timescale. Each time point is
checked against the chrons in `polarity`: if it falls within any chron interval
`[start[i], stop[i]]`, the sign matches `polarity.polarity` (`:normal` → `+1`,
`:reverse` → `−1`). Points that fall outside all chrons keep sign `+1` (assumed normal)
and a warning is issued if any such points exist.

# Arguments
- `ds`      : absolute-intensity dataset to sign
- `polarity`: a `PolarityDataset` (e.g. from `read_polarity`) describing polarity chrons
"""
function to_signed(
    ds      :: SynthesisDataset{AbsolutePaleointensity},
    polarity:: PolarityDataset,
)
    chron_sign   =  polarity.polarity === :normal ? 1.0 : -1.0
    default_sign = -chron_sign   # gaps have opposite polarity

    signed_vals = map(ds.time, ds.value) do t, v
        # polarity chrons use positive ages (past = +); ds.time uses negative ages (past = −)
        t_pos   = -t
        matched = any(t0 <= t_pos <= t1 for (t0, t1) in zip(polarity.start, polarity.stop))
        SignedPaleointensity((matched ? chron_sign : default_sign) * v.value)
    end

    return SynthesisDataset(ds.name, ds.time, signed_vals)
end

"""
    read_rpi(filepath::String; name::Symbol, min_age::Float64=-Inf) -> RawDataCollection{AbsolutePaleointensity}

Read a CSV file in Ziegler2011 format and return a `RawDataCollection{AbsolutePaleointensity}`.
Each unique `sample_name` becomes a `RawData{AbsolutePaleointensity}`, with times stored as
negative values (past convention) and rows sorted by age.

- `min_age` : keep only rows with age ≥ `-min_age` Ma (positive, e.g. `2.0` to keep the last 2 Ma).
              Datasets that are entirely outside the range are skipped silently.

Samples containing any negative RPI values are skipped with a warning.
The collection is identified by `name`.
"""
function read_rpi(filepath::String; name::Symbol, min_age::Float64=Inf)
    @assert min_age > 0.0 "min_age must be positive (e.g. 2.0 to keep the last 2 Ma)."
    df = CSV.read(filepath, DataFrame)
    df.age = -Float64.(df.age)   # negate all ages upfront: past → negative
    datasets = RawData{AbsolutePaleointensity}[]
    for (key, grp) in pairs(groupby(df, :sample_name))
        _df = let g = DataFrame(grp)
            g = g[g.age .>= -min_age, :]
            g[sortperm(g.age), :]
        end

        isempty(_df) && continue

        if any(_df.RPI .< 0)
            n_neg = count(_df.RPI .< 0)
            @warn "Skipping sample \"$(key.sample_name)\": $n_neg negative RPI value(s) found (AbsolutePaleointensity requires non-negative values)"
            continue
        end

        @assert allequal(_df.RPI_mean) "RPI_mean is not constant for sample $(key.sample_name)"
        push!(datasets, RawData{AbsolutePaleointensity}(
            Symbol(key.sample_name),
            Float64.(_df.age),
            AbsolutePaleointensity.(_df.RPI),
            Float64(first(_df.RPI_mean)),
            Float64(first(_df.lat)),
            Float64(first(_df.lon)),
            Int(first(_df.sample_id)),
            Float64(first(_df.scale_factor))
        ))
    end
    return RawDataCollection(name, datasets)
end

"""
    resample(ds::RawData{T}, Δt::Float64) -> SynthesisDataset{T}

Resample a non-uniformly sampled `RawData{T}` onto a uniform time grid with spacing `Δt` (in Ma)
using linear interpolation. The grid spans from the first to the last time point of `ds`.
Returns a `SynthesisDataset{T}` with the same name.

- `Δt` : grid spacing in Ma (e.g. `0.001` for 1 kyr)
"""
function resample(ds::RawData{T}, Δt::Float64) where {T<:AbstractPaleointensity}
    t_raw = ds.time
    v_raw = [x.value for x in ds.value]

    t_grid = collect(range(first(t_raw), last(t_raw); step=Δt))
    v_grid = _linear_interp(t_raw, v_raw, t_grid)

    return SynthesisDataset(ds.name, t_grid, T.(v_grid))
end

"""
    resample(ds::RawData{T}, t_target::AbstractVector{Float64}) -> SynthesisDataset{T}

Resample a non-uniformly sampled `RawData{T}` onto the provided `t_target` grid using
linear interpolation. Only grid points that fall within `[first(ds.time), last(ds.time)]`
are retained (points outside the record's time range are dropped).
Returns a `SynthesisDataset{T}` with the same name.

Use this overload to project multiple records onto a shared common grid.
"""
function resample(ds::RawData{T}, t_target::AbstractVector{Float64}) where {T<:AbstractPaleointensity}
    t_raw  = ds.time
    v_raw  = [x.value for x in ds.value]

    # Keep only target points within the record's own time span
    t_grid = filter(t -> first(t_raw) <= t <= last(t_raw), t_target)
    v_grid = _linear_interp(t_raw, v_raw, t_grid)

    return SynthesisDataset(ds.name, t_grid, T.(v_grid))
end

# Linear interpolation of v_old (sampled at t_old) onto t_new.
# t_old must be sorted. Values outside the range are clamped to the boundary.
function _linear_interp(t_old::Vector{Float64}, v_old::Vector{Float64}, t_new::Vector{Float64})
    v_new = similar(t_new)
    for (i, t) in enumerate(t_new)
        j = searchsortedfirst(t_old, t)
        if j == 1
            v_new[i] = v_old[1]
        elseif j > length(t_old)
            v_new[i] = v_old[end]
        else
            α = (t - t_old[j-1]) / (t_old[j] - t_old[j-1])
            v_new[i] = v_old[j-1] + α * (v_old[j] - v_old[j-1])
        end
    end
    return v_new
end

"""
    add_noise(ds::SynthesisDataset{T}; σ, λ, kernel, ϵ_reg) -> SynthesisDataset{T}

Add spatially-correlated Gaussian noise to a `SynthesisDataset` by sampling from a
Gaussian process with covariance matrix K[i,j] = σ² · kernel(t[i], t[j]).

The noise is drawn via Cholesky decomposition of K and added to the existing values.
A small regularisation term `ϵ_reg · I` is added to K for numerical stability.

When `λ = 0`, the kernel is degenerate (pure white noise): K is taken as σ² · I directly,
skipping the full covariance build and Cholesky decomposition.

# Arguments
- `σ`      : standard deviation of the noise (scales all kernel values by σ²)
- `λ`      : correlation length for the default exponential (OU) kernel; set to `0` for
             uncorrelated white noise. Ignored when a custom `kernel` is provided.
- `kernel` : a function `(ti, tj) -> Float64` defining the covariance structure.
             Defaults to an exponential (Ornstein-Uhlenbeck) kernel parameterised by `λ`.
- `ϵ_reg`  : diagonal regularisation added to K before Cholesky (default `1e-20`)

# Examples
```julia
# White noise (λ = 0)
noisy = add_noise(ds; σ=0.5, λ=0.0)

# Exponential kernel with λ = 0.1 Ma
noisy = add_noise(ds; σ=0.5, λ=0.1)

# Custom squared-exponential kernel
noisy = add_noise(ds; σ=0.5, kernel=(ti, tj) -> exp(-((ti - tj)^2) / (2 * 0.05^2)))
```
"""
function add_noise(ds::SynthesisDataset{T};
    σ,
    λ      = nothing,
    kernel = isnothing(λ) ? error("provide either a kernel or λ") :
                            (ti, tj) -> exp(-abs(ti - tj) / λ),
    ϵ_reg  = 1e-20,
    ) where {T<:AbstractPaleointensity}
    t = ds.time
    n = length(t)

    # White noise shortcut: K = σ²·I, no Cholesky needed
    if !isnothing(λ) && λ == 0.0
        noise = σ .* randn(n)
    else
        # Build full covariance matrix and sample via Cholesky
        K = σ^2 .* [kernel(t[i], t[j]) for i in 1:n, j in 1:n]
        # Add a small regularization
        K += ϵ_reg * I
        noise = cholesky(Symmetric(K)).L * randn(n)
    end

    values = [x.value for x in ds.value] .+ noise

    return SynthesisDataset(ds.name, t, T.(values))
end

"""
    read_polarity(filepath::String; name::Symbol, polarity::Symbol=:normal, min_age::Float64=Inf)
        -> PolarityDataset

Read a tab-separated `.txt` file in GPTS format and return a `PolarityDataset`.
`min_age` (Ma), if given, keeps only chrons starting at or before that age (e.g.
`min_age=2.0` keeps the last 2 Ma of the record).
"""
function read_polarity(filepath::String; name::Symbol, polarity::Symbol=:normal, min_age::Float64=Inf)
    @assert polarity ∈ (:normal, :reverse) "polarity must be :normal or :reverse"
    @assert min_age > 0.0 "min_age must be positive (e.g. 2.0 to keep the last 2 Ma)."
    data, _ = readdlm(filepath, '\t', Any; header=true)
    valid   = [isa(data[i, 1], Number) && isa(data[i, 2], Number) for i in axes(data, 1)]
    n_dropped = count(.!valid)
    if n_dropped > 0
        @warn "Dropped $n_dropped row(s) with missing or non-numeric values in \"$(basename(filepath))\""
    end
    start_ages = Float64.(data[valid, 1])
    stop_ages  = Float64.(data[valid, 2])

    keep = start_ages .<= min_age
    return PolarityDataset(name, start_ages[keep], stop_ages[keep], polarity)
end
