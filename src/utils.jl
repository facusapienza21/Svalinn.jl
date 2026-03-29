using MAT
using DelimitedFiles

"""
    read_synthesis(filepath::String; name::Symbol) -> SynthesisDataset

Read a `.mat` file containing a signed dipole moment time series and return a `SynthesisDataset`.
"""
function read_synthesis(filepath::String; name::Symbol)
    raw = matread(filepath)
    d   = vec(raw["d"])
    t   = vec(raw["t"]) ./ 1000   # ka → Ma
    return SynthesisDataset(name, t, SignedPaleointensity.(d))
end

"""
    read_polarity(filepath::String; name::Symbol, polarity::Symbol=:normal) -> PolarityDataset

Read a tab-separated `.txt` file in GPTS format and return a `PolarityDataset`.
"""
function read_polarity(filepath::String; name::Symbol, polarity::Symbol=:normal)
    @assert polarity ∈ (:normal, :reverse) "polarity must be :normal or :reverse"
    data, _ = readdlm(filepath, '\t', Any; header=true)
    valid      = [isa(data[i, 1], Number) && isa(data[i, 2], Number) for i in axes(data, 1)]
    start_ages = Float64.(data[valid, 1])
    stop_ages  = Float64.(data[valid, 2])
    return PolarityDataset(name, start_ages, stop_ages, polarity)
end
