# Tests for the Davis et al. (2021) model (src/models/Davis2021.jl).
#
# Everything here calls the package's own functions and compares against independent
# references: exact Gaussian moments (BigFloat), the analytic white-noise stationary
# density, and the analytic mean first-passage time (mean chron length).
#
# The physics tests use normalized units (X̄ = 1, γ = 1, ḡ = 1/2, θσ = 0.5), so the
# white-noise intensity is s² = ḡ²·2θσ = 1/4 and the barrier is 2ΔU/s² = 4/3: low enough
# that a run of 3·10⁴ relaxation times contains a few thousand chrons.

using Statistics

const S = Svalinn

# ---------------------------------------------------------------------------------------
# Analytic references for the white-noise limit  dx = v(x) dt + s dW,
# v(x) = -γ x(|x| - 1),  U(x) = γ(|x|³/3 - x²/2),  stationary density ∝ exp(-2U/s²).
# ---------------------------------------------------------------------------------------
_U(x, γ) = γ * (abs(x)^3 / 3 - x^2 / 2)

"Stationary density p(x) ∝ exp(-2U(x)/s²) on a grid, normalized."
function _stationary_density(γ, s2; L = 4.0, n = 200_001)
    x  = collect(range(-L, L; length = n))
    w  = exp.(-2 .* _U.(x, γ) ./ s2)
    return x, w ./ (sum(w) * (x[2] - x[1]))
end

"""
Mean time to reach -h0 starting at +h0 (reflecting at +∞): the mean chron length under
the :Gammaitoni1998 hysteresis criterion,
    T = (2/s²) ∫_{-h0}^{h0} e^{Φ(y)} ∫_y^∞ e^{-Φ(z)} dz dy,   Φ = 2U/s².
"""
function _analytic_mean_chron(γ, s2, h0; L = 4.0, n = 400_001)
    z  = collect(range(-h0, L; length = n))
    dz = z[2] - z[1]
    Φ  = 2 .* _U.(z, γ) ./ s2
    f  = exp.(-Φ)
    inner = zeros(n)                          # ∫_y^L e^{-Φ}, trapezoid from the right
    for i in n-1:-1:1
        inner[i] = inner[i+1] + 0.5 * dz * (f[i] + f[i+1])
    end
    g = exp.(Φ) .* inner
    m = findlast(<=(h0), z)
    return (2 / s2) * dz * (sum(g[1:m]) - 0.5 * (g[1] + g[m]))
end

_vals(ds) = [Float64(v.value) for v in ds.value]

function test_Davis2021()

    # ===================================================================================
    # Exact OU step (Gillespie 1996)
    # ===================================================================================
    @testset "_ou_bracket: closed form and series vs BigFloat" begin
        setprecision(256) do
            for u in 10 .^ range(-8, 2; length = 501)
                ub  = big(u)
                ref = ub - 2 * (1 - exp(-ub)) + (1 - exp(-2ub)) / 2
                @test S._ou_bracket(u) ≈ Float64(ref) rtol = 1e-6
                @test S._ou_bracket(u) > 0
            end
        end
    end

    @testset "_exact_ou_step: moments match the exact Gaussian transition" begin
        Random.seed!(1)
        θ, c, ζ0 = 0.2, 17.0, 0.8
        N = 200_000
        for δt in (1e-3, 0.02, 0.2, 2.0)             # δt/θ = 0.005, 0.1, 1, 10
            α   = exp(-δt / θ)
            μζ  = ζ0 * α
            μI  = θ * ζ0 * (1 - α)
            Σ11 = (c * θ / 2) * (1 - α^2)
            Σ12 = (c * θ^2 / 2) * (1 - α)^2
            Σ22 = c * θ^3 * (δt / θ - 2 * (1 - α) + (1 - α^2) / 2)

            draws = [S._exact_ou_step(ζ0, δt, θ, c) for _ in 1:N]
            ζ, I  = first.(draws), last.(draws)

            @test abs(mean(ζ) - μζ) < 5 * sqrt(Σ11 / N)
            @test abs(mean(I) - μI) < 5 * sqrt(Σ22 / N)
            @test var(ζ) ≈ Σ11 rtol = 0.03
            @test var(I) ≈ Σ22 rtol = 0.03
            @test cov(ζ, I) / sqrt(Σ11 * Σ22) ≈ Σ12 / sqrt(Σ11 * Σ22) atol = 0.02
        end
    end

    @testset "_exact_ou_step: white-noise limit θ → 0 at fixed cθ²" begin
        # With c = σ0²/θ², Var(ΔI) → σ0²·δt and ζ decorrelates completely (ΔI ~ W increment)
        Random.seed!(2)
        σ0², δt, N = 1.0, 0.01, 100_000
        for θ in (1e-4, 1e-6)
            c = σ0² / θ^2
            ζ0 = sqrt(c * θ / 2) * randn()           # stationary start
            I  = [last(S._exact_ou_step(ζ0, δt, θ, c)) for _ in 1:N]
            @test var(I) ≈ σ0² * δt rtol = 0.03
            @test all(isfinite, I)
        end
        @test all(isfinite, S._exact_ou_step(0.3, 1e-14, 0.5, 4.0))   # tiny δt/θ: no NaN
    end

    # ===================================================================================
    # Building blocks
    # ===================================================================================
    @testset "Drift flow is the exact flow of v" begin
        γ, X̄ = 1.3, 2.0
        v(x) = -γ * (x / X̄) * (abs(x) - X̄)
        flow(x, h) = S._davis2021_exact_drift_flow(x, h, γ, X̄)
        for x in (-5.0, -2.0, -0.3, 0.0, 1e-8, 0.7, 2.0, 6.0)
            @test flow(x, 0.0) ≈ x
            @test flow(flow(x, 0.3), 0.5) ≈ flow(x, 0.8) rtol = 1e-12 atol = 1e-14  # semigroup
            ε = 1e-6
            @test (flow(x, ε) - flow(x, -ε)) / 2ε ≈ v(x) rtol = 1e-6 atol = 1e-10    # d/dh = v
            @test flow(-x, 0.4) ≈ -flow(x, 0.4)                                       # odd
            x != 0 && @test sign(flow(x, 10.0)) == sign(x)                            # no crossing
        end
        @test flow(0.1, 50.0) ≈ X̄ rtol = 1e-10
    end

    @testset "Time grid lands exactly on saveat" begin
        sv = collect(-1.0:0.1:0.0)
        ts = S._davis2021_time_stops((-1.0, 0.0), sv, 0.03)
        @test ts[1] == -1.0 && ts[end] == 0.0
        @test all(s -> s in ts, sv)
        @test maximum(diff(ts)) <= 0.03 + 1e-12
        @test issorted(ts)
        ts2 = S._davis2021_time_stops((-1.0, 0.0), [-0.55, -0.123], 0.1)
        @test (-0.55 in ts2) && (-0.123 in ts2)
        @test maximum(diff(ts2)) <= 0.1 + 1e-12
    end

    @testset "Reversal detection on a synthetic path" begin
        t = collect(range(0, 4; length = 4001))
        x = sin.(π .* t)                           # zeros at 1, 2, 3
        x[1500:1600] .= 0.1 .* sin.(50 .* t[1500:1600])   # small wiggles near 0
        detect(criterion; h0 = nothing) = begin
            rev, state = Float64[], :normal
            for i in 2:length(t)
                state = S._davis2021_reversal_step!(rev, state, t[i-1], x[i-1], t[i], x[i];
                                                    criterion = criterion, h0 = h0)
            end
            rev
        end
        @test length(detect(:zero_crossing)) > 3                  # wiggles count
        rg = detect(:Gammaitoni1998; h0 = 0.5)
        @test rg ≈ [1 + 1/6, 2 + 1/6, 3 + 1/6] atol = 1e-3       # sin(πt) = ∓0.5
    end

    @testset "API: reproducibility, output grid, input checks" begin
        kw = (; γ = 1.0, X̄ = 3.0, ḡ = 1.0, θ = 0.05, tspan = (-10.0, 0.0), saveat = 0.1)
        a = simulate(; model = :Davis2021, kw..., seed = 7)
        b = simulate(; model = :Davis2021, kw..., seed = 7)
        c = simulate(; model = :Davis2021, kw..., seed = 8)
        @test _vals(a) == _vals(b)
        @test _vals(a) != _vals(c)
        @test a.time ≈ collect(-10.0:0.1:0.0)
        @test length(a.time) == length(a.value)
        @test _vals(a)[1] == 1.0                              # X0 = X̄ → normalized 1
        d = simulate(; model = :Davis2021, kw..., saveat = [-7.0, -3.3, -0.2], seed = 7)
        @test d.time == [-7.0, -3.3, -0.2]
        @test_throws ArgumentError simulate(; model = :Davis2021, kw..., saveat = [-20.0, 0.0])
        @test_throws ArgumentError simulate(; model = :Davis2021, kw..., saveat = [0.0, -1.0])
        @test_throws ArgumentError simulate(; model = :Davis2021, kw..., detect_reversals = true,
                                             criterion = :foo)
    end

    # ===================================================================================
    # Physics: white-noise limit (θγ ≪ 1, θσ = 0.5 ⇒ s² = ḡ²) must reproduce the analytic
    # stationary density and mean chron length, for θ = 0 and for small θ > 0.
    # ===================================================================================
    γ, ḡ, h0, dt = 1.0, 1 / 2, 0.5, 1e-3
    s2 = ḡ^2
    xs, ps = _stationary_density(γ, s2)
    Ex2_ref   = sum(xs .^ 2 .* ps) * (xs[2] - xs[1])
    chron_ref = _analytic_mean_chron(γ, s2, h0)

    run(θ; seed, T = 30_000.0) = simulate(; model = :Davis2021, γ = γ, X̄ = 1.0, ḡ = ḡ,
        θ = θ, σ = θ == 0 ? 0.5 : 0.5 / θ, dt = dt, tspan = (-T, 0.0), saveat = 0.05,
        seed = seed, detect_reversals = true, criterion = :Gammaitoni1998, h0 = h0)

    @testset "Physics: stationary density and mean chron length (θ = $θ)" for θ in (0.0, 1e-3)
        ds = run(θ; seed = 11)
        x  = _vals(ds)
        c  = diff(ds.reversals)
        se = std(c) / sqrt(length(c))
        @info "Davis2021 physics (θ = $θ)" n_chrons = length(c) mean_chron = mean(c) analytic = chron_ref z = (mean(c) - chron_ref) / se E_x2 = mean(abs2, x) E_x2_ref = Ex2_ref

        # stationary density: second moment and L1 distance of the histogram of |x|. The
        # density is symmetric, so folding removes the (slowly averaging) imbalance in
        # time spent in each well and tests the shape of the density.
        @test mean(abs2, x) ≈ Ex2_ref rtol = 0.03
        edges = range(0.0, 2.5; length = 26)
        ax = abs.(x)
        h  = [count(v -> edges[k] <= v < edges[k+1], ax) for k in 1:25] ./ (length(ax) * step(edges))
        hp = [2 * sum(ps[(xs .>= edges[k]) .& (xs .< edges[k+1])]) * (xs[2] - xs[1]) / step(edges)
              for k in 1:25]
        @test sum(abs.(h .- hp)) * step(edges) < 0.03

        # mean chron length vs analytic MFPT
        @test length(c) > 300
        @test abs(mean(c) - chron_ref) < 4se
    end

    @testset "Physics: mean chron length is independent of θ for θγ ≪ 1" begin
        means = map((0.0, 1e-4, 1e-2)) do θ
            c = diff(run(θ; seed = 21).reversals)
            (mean(c), std(c) / sqrt(length(c)))
        end
        for (m, se) in means[2:end]
            @test abs(m - means[1][1]) < 4 * sqrt(se^2 + means[1][2]^2)
        end
    end
end
