function test_read_synthesis(data_dir)
    filepath = joinpath(data_dir, "PADM2Mstar.mat")
    padm2m   = read_synthesis(filepath; name=:PADM2M)

    @test padm2m isa SynthesisDataset
    @test padm2m.name == :PADM2M
    @test length(padm2m.time) == length(padm2m.value)
    @test all(x -> x isa SignedPaleointensity, padm2m.value)
end

function test_read_polarity(data_dir)
    filepath = joinpath(data_dir, "GPTS_CK95.txt")
    gpts     = read_polarity(filepath; name=:GPTS_CK95)

    @test gpts isa PolarityDataset
    @test gpts.name == :GPTS_CK95
    @test gpts.polarity == :normal
    @test length(gpts.start) == length(gpts.stop)
    @test all(gpts.stop .> gpts.start)
end
