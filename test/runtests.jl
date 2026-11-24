using Svalinn
using Test
using Random

const data_dir = joinpath(@__DIR__, "data")

include("test_types.jl")
include("test_Davis2021.jl")

@testset verbose=true "Svalinn.jl" begin

    @testset verbose=true "Data processing" begin
        @testset verbose=true "Read synthesis" begin test_read_synthesis(data_dir) end
        @testset verbose=true "Read polarity"  begin test_read_polarity(data_dir)  end
    end

    @testset verbose=true "Davis2021 model" begin test_Davis2021() end

end
