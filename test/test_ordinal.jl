function test_ordinal_entropy_random()
    Random.seed!(42)
    x = randn(100_000)
    @test ordinal_entropy(x; d=5, τ=1) ≈ 1.0 atol=0.01
end

function test_ordinal_entropy_sorted()
    x_sorted = collect(1.0:100_000.0)
    @test ordinal_entropy(x_sorted; d=5, τ=1) ≈ 0.0 atol=0.01
end

function test_ordinal_entropy_logistic()
    r = 4.0
    x_logistic = let x = zeros(10_000); x[begin] = 0.4
        for i in eachindex(x)[2:end]; x[i] = r * x[i-1] * (1 - x[i-1]) end; x
    end
    o        = OrdinalPatterns(; m=5, τ=1)
    probs    = probabilities(o, x_logistic)
    H_max    = log(factorial(5))
    H_manual = -sum(p * log(p) for p in probs if p > 0) / H_max
    @test ordinal_entropy(x_logistic; d=5, τ=1) ≈ H_manual atol=1e-10
end
