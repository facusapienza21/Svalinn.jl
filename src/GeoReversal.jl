module GeoReversal

using Infiltrator

export AbstractData, AbstractPaleointensity
export AbsolutePaleointensity, RelativePaleointensity, SignedPaleointensity
export SynthesisDataset, RawData, RawDataCollection, PolarityDataset
export read_synthesis, read_polarity, read_rpi, to_absolute, to_signed, resample, add_noise
export ordinal_entropy, ordinal_complexity, ordinal_entropy_complexity
export plot_CH_plane, plot_noise_curve!
export simulate

include("types.jl")
include("utils.jl")
include("show.jl")
include("ordinal_patterns.jl")
include("models.jl")

end
