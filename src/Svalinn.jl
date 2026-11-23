module Svalinn

using Infiltrator
using DifferentialEquations
using StochasticDiffEq
using Random
using Statistics
using StatsBase: sample, Weights
using Distributions: Uniform
using HypothesisTests: ExactOneSampleKSTest, ApproximateOneSampleKSTest,
                        ApproximateTwoSampleKSTest
import HypothesisTests: pvalue

# Data types
export AbstractData, AbstractPaleointensity
export AbsolutePaleointensity, RelativePaleointensity, SignedPaleointensity
export SynthesisDataset, RawData, RawDataCollection, PolarityDataset
export ParticleFilterResult

# Data I/O and processing
export read_synthesis, read_polarity, read_rpi, to_absolute, to_signed, resample, add_noise

# Models
export simulate

# Reversals and chrons
export reversal_callback, chrons, simulate_chrons, chron_lengths, filter_short_reversals

# Statistics and hypothesis tests
export H_statistic, poisson_ks_test, plot_H_cdf, chrons_ks_test, ChronsKSTest, pvalue

# Filtering
export particle_filter_loglikelihood, resample_multinomial, resample_systematic

# ---------------------------------------------------------------------------------------
# Data: types, readers/converters for paleomagnetic records, and pretty printing
# ---------------------------------------------------------------------------------------
include("types.jl")      # paleointensity types and dataset containers
include("utils.jl")      # read_* readers, to_absolute/to_signed, resample, add_noise
include("show.jl")       # Base.show methods for the dataset types

# ---------------------------------------------------------------------------------------
# Reversals: detection during integration and conversion of reversal times into chrons
# (loaded before the models, which use `reversal_callback` and `_patched_series`)
# ---------------------------------------------------------------------------------------
include("reversals.jl")

# ---------------------------------------------------------------------------------------
# Models: stochastic geodynamo models and the `simulate(; model = ...)` dispatcher
# ---------------------------------------------------------------------------------------
include("models/Davis2021.jl")   # Davis et al. (2021): bistable dipole + OU (colored) noise
include("models/G12.jl")         # Gissinger (2012): three-component geodynamo model
include("models/P09.jl")         # Petrelis et al. (2009): stochastic phase model
include("models/DW.jl")          # double-well potential with white noise
include("models/models.jl")      # simulate(; model = :G12 | :P09 | :DW | :Davis2021)

# ---------------------------------------------------------------------------------------
# Statistics: chron statistics and hypothesis tests
# ---------------------------------------------------------------------------------------
include("stats/utils.jl")            # H_statistic
include("stats/htest/poisson.jl")    # poisson_ks_test, plot_H_cdf: are reversals Poisson?
include("stats/htest/KS_chrons.jl")  # chrons_ks_test: data vs simulated chrons (2-sample KS)

# ---------------------------------------------------------------------------------------
# Filtering: particle filter for model likelihoods on observed records
# ---------------------------------------------------------------------------------------
include("filtering.jl")

end
