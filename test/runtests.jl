using GeoReversal
using ComplexityMeasures
using Test
using Random

const data_dir = joinpath(@__DIR__, "data")

include("test_types.jl")
include("test_ordinal.jl")

@testset verbose=true "GeoReversal.jl" begin

    @testset verbose=true "Data processing" begin
        @testset verbose=true "Read synthesis" begin test_read_synthesis(data_dir) end
        @testset verbose=true "Read polarity"  begin test_read_polarity(data_dir)  end
    end

    @testset verbose=true "Ordinal Patterns Metrics" begin
        test_ordinal_entropy_random()
        test_ordinal_entropy_sorted()
        test_ordinal_entropy_logistic()
    end

end
