using Svalinn
using Test
using Random

const data_dir = joinpath(@__DIR__, "data")

include("test_types.jl")

@testset verbose=true "Svalinn.jl" begin

    @testset verbose=true "Data processing" begin
        @testset verbose=true "Read synthesis" begin test_read_synthesis(data_dir) end
        @testset verbose=true "Read polarity"  begin test_read_polarity(data_dir)  end
    end

end
