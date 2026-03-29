module GeoReversal

using Infiltrator

export AbstractData, AbstractPaleointensity
export AbsolutePaleointensity, RelativePaleointensity, SignedPaleointensity
export SynthesisDataset, RawData, PolarityDataset
export read_synthesis, read_polarity, to_absolute
export ordinal_entropy, ordinal_complexity, ordinal_entropy_complexity
export plot_CH_plane

include("types.jl")
include("utils.jl")
include("show.jl")
include("ordinal_patterns.jl")

end
