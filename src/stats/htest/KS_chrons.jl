# Complementary CDF of the asymptotic Kolmogorov distribution, P(K >= lambda). Ported
# verbatim from PIML-reversals' `AnalisisPoissonaidad.jl` (`kolmogorov_survival`).
function _kolmogorov_survival(lambda::Real)
    lambda_value = Float64(lambda)

    if lambda_value <= 0.0
        return 1.0
    end

    # For small lambda, the usual alternating series converges poorly.
    # This alternative expression computes the CDF, and we return 1 - CDF.
    if lambda_value < 1.18
        s = 0.0
        for k in 1:10_000
            term = exp(-((2 * k - 1)^2 * pi^2) / (8.0 * lambda_value^2))
            s += term
            term < 1.0e-14 && break
        end
        cdf = sqrt(2.0 * pi) / lambda_value * s
        return clamp(1.0 - cdf, 0.0, 1.0)
    else
        s = 0.0
        for k in 1:10_000
            signk = isodd(k) ? 1.0 : -1.0
            term = 2.0 * signk * exp(-2.0 * k^2 * lambda_value^2)
            s += term
            abs(term) < 1.0e-14 && break
        end
        return clamp(s, 0.0, 1.0)
    end
end

"""
    ChronsKSTest

Result of [`chrons_ks_test`](@ref): the two-sample KS statistic `δ` and sample sizes
`n_x`, `n_y` (from `HypothesisTests.ApproximateTwoSampleKSTest`) together with the
`correction` used to convert `δ` into a p-value. Call `pvalue` on it as usual.
"""
struct ChronsKSTest
    n_x        :: Int
    n_y        :: Int
    δ          :: Float64
    correction :: Symbol
end

"""
    pvalue(x::ChronsKSTest)

p-value for a [`ChronsKSTest`](@ref), computed from the asymptotic Kolmogorov
distribution at `λ = f(n_eff) * x.δ`, where `n_eff = n_x*n_y/(n_x+n_y)` and `f` depends
on `x.correction`:

- `:stephens`: `f(n_eff) = √n_eff + 0.12 + 0.11/√n_eff`, the Stephens (1970) finite-sample
  correction (a.k.a. the *Numerical Recipes* `probks` formula) — matches the convention
  used in the PIML-reversals Poisson analysis. More accurate than the plain asymptotic
  at the small-to-moderate sample sizes typical here.
- `:traditional`: `f(n_eff) = √n_eff`, the plain (uncorrected) asymptotic used by
  `HypothesisTests.ApproximateTwoSampleKSTest`.
"""
function pvalue(x::ChronsKSTest)
    n_eff = x.n_x * x.n_y / (x.n_x + x.n_y)
    λ = if x.correction === :stephens
        (sqrt(n_eff) + 0.12 + 0.11 / sqrt(n_eff)) * x.δ
    elseif x.correction === :traditional
        sqrt(n_eff) * x.δ
    else
        throw(ArgumentError("correction must be :stephens or :traditional, got $(x.correction)"))
    end
    return _kolmogorov_survival(λ)
end

"""
    chrons_ks_test(H_data::AbstractVector{<:Real}, H_simulated::AbstractVector{<:Real};
                   correction::Symbol = :stephens)
    chrons_ks_test(data::PolarityDataset, simulated::PolarityDataset; correction::Symbol = :stephens)

Two-sample Kolmogorov–Smirnov test comparing the Carbone et al. (2005) `H_statistic`
of a `data` chron record against that of a `simulated` one (e.g. from
`simulate_chrons`). Unlike `poisson_ks_test`, this does not assume either side is
Poisson — it compares the two empirical `H` distributions directly, so `simulated` can
be made arbitrarily large/precise to approximate the model's true `H` distribution as
closely as possible.

The `PolarityDataset` method just computes `H_statistic` on each side and delegates to
the vector method. Calling the vector method directly lets you pool `H_statistic`
results from several *independent* replicate simulations (different seeds) into one
`H_simulated` before testing — since consecutive `H` values within a single run are
correlated (both by `H_statistic`'s overlapping-window construction and by the model's
own noise correlation time), pooling across independent runs increases the *effective*
sample size in a way that simply lengthening one run does not.

The KS statistic `δ` itself is computed via `HypothesisTests.ApproximateTwoSampleKSTest`;
`correction` selects how the p-value is derived from it (see
[`pvalue(::ChronsKSTest)`](@ref)):
- `:stephens` (default): Stephens (1970) finite-sample-corrected asymptotic p-value.
  Matches the convention used in the PIML-reversals Poisson analysis.
- `:traditional`: the plain `HypothesisTests.jl` asymptotic p-value (no correction).

Returns a [`ChronsKSTest`](@ref). A small p-value rejects the hypothesis that `data`
and `simulated` share the same `H` distribution.
"""
function chrons_ks_test(H_data::AbstractVector{<:Real}, H_simulated::AbstractVector{<:Real};
                         correction::Symbol = :stephens)
    correction ∈ (:stephens, :traditional) ||
        throw(ArgumentError("correction must be :stephens or :traditional, got $correction"))
    ks = ApproximateTwoSampleKSTest(H_data, H_simulated)
    return ChronsKSTest(ks.n_x, ks.n_y, ks.δ, correction)
end

function chrons_ks_test(data::PolarityDataset, simulated::PolarityDataset; correction::Symbol = :stephens)
    return chrons_ks_test(H_statistic(data), H_statistic(simulated); correction = correction)
end
