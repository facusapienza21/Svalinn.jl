module GeoReversal

export AbstractData, AbstractPaleointensity
export AbsolutePaleointensity, RelativePaleointensity, SignedPaleointensity
export SynthesisDataset, RawData, PolarityDataset
export read_synthesis, read_polarity

include("types.jl")
include("utils.jl")
include("show.jl")

end
