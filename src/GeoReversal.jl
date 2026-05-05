module GeoReversal

using Infiltrator
using DifferentialEquations
using StochasticDiffEq
using Random
using StatsBase: sample, Weights

export AbstractData, AbstractPaleointensity
export AbsolutePaleointensity, RelativePaleointensity, SignedPaleointensity
export SynthesisDataset, RawData, RawDataCollection, PolarityDataset
export ParticleFilterResult
export read_synthesis, read_polarity, read_rpi, to_absolute, to_signed, resample, add_noise
export ordinal_entropy, ordinal_complexity, ordinal_entropy_complexity
export plot_CH_plane, plot_noise_curve!
export simulate
export ami, ami_curve, best_tau, acf, best_tau_acf
export particle_filter_loglikelihood, resample_multinomial, resample_systematic

include("types.jl")
include("utils.jl")
include("show.jl")
include("ordinal_patterns.jl")
include("models/Davis2021.jl")
include("models/G12.jl")
include("models/P09.jl")
include("models/DW.jl")
include("models/models.jl")
include("embedding_delay.jl")
include("filtering.jl")

end
