"""
    reversal_callback(dipole::Function, u0, t0; criterion=:zero_crossing, h0=nothing)
        -> (callback, reversals, last_t)

Build a callback that detects polarity reversals of the scalar dipole observable
`dipole(u)` reconstructed from the integrator state `u`, and records the
(solver-internal) time of each reversal into `reversals` as it happens. `u0`/`t0` are
the initial state/time, used to infer the starting polarity and to seed `last_t`. The
same callback works for both `ODEProblem` and `SDEProblem` integrators, since it only
depends on `u` and `t`.

`last_t` is a `Ref{Float64}` updated on every accepted integrator step to the current
`integrator.t`, so `last_t[]` always holds the true last time the integrator reached —
including when the solve is aborted early (e.g. `retcode = :MaxIters` or `:Unstable`).
This matters because `sol.t[end]` is *not* reliable in that case: SciMLBase's
`save_end` only appends the final point on ordinary termination, so on an aborted
solve `sol.t` can be stuck at `t0` even though the callback (and hence `reversals`)
correctly tracked the integration far past it.

Reported times are in the solver's internal time units — callers are responsible
for converting back to Ma using the same transform applied to `saveat`.

# Criteria
- `:zero_crossing` (default): a reversal is recorded every time `dipole(u)` crosses
  zero. `dipole` only needs to be correct up to a positive scaling factor, since only
  its root matters.
- `:Gammaitoni1998`: hysteresis (Schmitt-trigger) rule with threshold `h0 > 0`. The
  field is `:normal` once `dipole(u)` rises *through* `+h0` (an up-crossing) and
  `:reverse` once it falls *through* `-h0` (a down-crossing) — direction matters, not
  just touching the level; touching either threshold while already in the corresponding
  state is also not a reversal. Requires `h0`.
"""
function reversal_callback(dipole::Function, u0, t0; criterion::Symbol=:Gammaitoni1998, h0=nothing)
    reversals = Float64[]
    last_t    = Ref(Float64(t0))
    track_condition(u, t, integrator) = true
    track_affect!(integrator) = (last_t[] = integrator.t)
    track_cb = DiscreteCallback(track_condition, track_affect!; save_positions = (false, false))

    if criterion === :zero_crossing
        condition(u, t, integrator) = dipole(u)
        affect!(integrator) = push!(reversals, integrator.t)
        cb = ContinuousCallback(condition, affect!; save_positions = (false, false))
        return CallbackSet(cb, track_cb), reversals, last_t

    elseif criterion === :Gammaitoni1998
        h0 === nothing && throw(ArgumentError("criterion=:Gammaitoni1998 requires h0"))
        h0 > 0 || throw(ArgumentError("h0 must be positive, got $h0"))

        state = Ref(dipole(u0) >= 0 ? :normal : :reverse)

        condition_up(u, t, integrator)   = dipole(u) - h0
        condition_down(u, t, integrator) = dipole(u) + h0

        function affect_up!(integrator)
            if state[] !== :normal
                state[] = :normal
                push!(reversals, integrator.t)
            end
        end
        function affect_down!(integrator)
            if state[] !== :reverse
                state[] = :reverse
                push!(reversals, integrator.t)
            end
        end

        # Restrict each callback to its physically meaningful direction only — an
        # up-crossing for +h0 (dipole(u) rising through it), a down-crossing for -h0
        # (falling through it). Without this, ContinuousCallback's 2-arg form defaults
        # affect_neg! = affect!, so e.g. cb_up would also fire when dipole(u) *retreats*
        # back down through +h0 (harmless here only because the state guard happens to
        # absorb it, but it's not the documented/intended definition — matches the
        # dX/dt-signed criterion originally drafted for this rule).
        cb_up   = ContinuousCallback(condition_up,   affect_up!,   nothing;     save_positions = (false, false))
        cb_down = ContinuousCallback(condition_down, nothing,      affect_down!; save_positions = (false, false))
        return CallbackSet(cb_up, cb_down, track_cb), reversals, last_t

    else
        throw(ArgumentError("Unknown criterion :$criterion. Supported: :zero_crossing, :Gammaitoni1998"))
    end
end

"""
    _patched_series(saveat_vec, values, n, detect_reversals, last_t, to_ma)
        -> (time_vec, value_vec)

Build the time/value vectors for the first `n` points of `saveat_vec`/`values`, but
correct the record's end time to `to_ma(last_t[])` when `detect_reversals` is set.
`last_t` (from `reversal_callback`) holds the solver's true last-reached internal
time, which is more accurate than `saveat_vec[n]` whenever the solve was aborted early
(see `reversal_callback` for why `sol.t`/`saveat` alone can't be trusted in that case).

When `n == 1` (only the initial point was saved), the corrected end time is appended
as a second point rather than overwriting the only point — overwriting it would erase
the record's start time, since index 1 and index `end` coincide.
"""
function _patched_series(saveat_vec, values, n::Int, detect_reversals::Bool, last_t, to_ma::Function)
    time_vec  = collect(Float64, saveat_vec[1:n])
    value_vec = values[1:n]
    if detect_reversals && n > 0
        corrected_end = to_ma(last_t[])
        if n == 1
            time_vec  = [time_vec[1], corrected_end]
            value_vec = [value_vec[1], value_vec[1]]
        else
            time_vec[end] = corrected_end
        end
    end
    return time_vec, value_vec
end

"""
    filter_short_reversals(reversals, t0, t_end; min_length=0.0) -> (times, indices)

Iteratively eliminate the shortest chron. `t0`/`t_end` (the record's fixed start/end
times) together with `reversals` define a sequence of chrons (the intervals between
consecutive boundaries); at each step, find the chron with the smallest duration, and if
it is shorter than `min_length` (Ma), drop the reversal separating it from a neighbor —
merging the two into one longer chron — then repeat on the updated boundary list. Stops
once the shortest remaining chron is at least `min_length`, or only one chron remains.
`min_length <= 0` is a no-op.

The chron being eliminated merges *backward* into its left-hand neighbor (drop its left
boundary), except when it is the first chron in the record — there the left boundary is
the fixed `t0`, so it merges *forward* instead (drop its right boundary). `t_end` is
never dropped either.

Merging backward (not forward) is required for correctness, not just a convention: any
interior chron's two immediate neighbors always have the *same* polarity as each other
(alternation forces `..., P, ~P, P, ...`), so eliminating the `~P` chron should let its
two `P` neighbors end up adjacent with matching polarity, so `chrons`'s coalescing step
can merge them into one continuous `P` chron — the physically correct result of treating
the eliminated chron as noise. Dropping the *left* edge achieves this (the merged segment
inherits the left neighbor's — correct — polarity via its original toggle index, and
still touches the untouched right neighbor, which has the same polarity, at their shared
edge). Dropping the *right* edge instead (as this used to do) makes the merged segment
inherit the eliminated chron's own (wrong) polarity, silently mislabeling the right
neighbor's true duration.

Returns the kept reversal `times` together with their 1-based `indices` into the
original `reversals` vector. The indices matter because every element of `reversals`
(kept or dropped) is a genuine polarity toggle: the polarity of the segment starting at
a kept reversal is determined by *how many original toggles preceded it* (odd/even), not
by its position among the survivors — dropping an odd number of reversals between two
kept ones still flips the polarity once, dropping an even number cancels out. `chrons`
uses `indices` for exactly this reason; naively alternating polarity by kept-list
position instead of original-list position silently mislabels every chron after the
first place an odd number of reversals gets dropped in a row.
"""
function filter_short_reversals(reversals::AbstractVector{<:Real}, t0::Real, t_end::Real;
                                 min_length::Real = 0.0)
    if min_length <= 0
        return collect(Float64, reversals), collect(1:length(reversals))
    end

    edges   = Float64[t0; reversals...; t_end]
    indices = Vector{Union{Int,Nothing}}(undef, length(edges))
    indices[1]   = nothing   # t0: fixed record boundary, not a real reversal
    indices[end] = nothing   # t_end: ditto
    indices[2:end-1] .= 1:length(reversals)

    while length(edges) > 2
        durations = diff(edges)
        i = argmin(durations)
        durations[i] >= min_length && break

        if i == 1                   # shortest chron's left edge is the fixed t0
            deleteat!(edges, i + 1) # -> merge forward: drop its right edge instead
            deleteat!(indices, i + 1)
        else
            deleteat!(edges, i)     # merge backward: drop the left edge
            deleteat!(indices, i)
        end
    end

    kept_times   = edges[2:end-1]
    kept_indices = Int[indices[k] for k in 2:length(indices)-1]
    return kept_times, kept_indices
end

"""
    chrons(ds::SynthesisDataset; polarity=:normal, min_length=0.0) -> PolarityDataset

Turn the `reversals` recorded on `ds` (see `simulate(...; detect_reversals=true)`)
into a `PolarityDataset` of the chrons matching `polarity`. The polarity of the
first chron is inferred from the sign of `ds.value[1]`; chrons alternate from there.

`min_length` (Ma) discards reversals that would create a chron shorter than it, merging
that excursion into its neighbor instead (see `filter_short_reversals`); default `0.0`
keeps every detected reversal. Each surviving segment's polarity is derived from its
*original* position in `ds.reversals` (see `filter_short_reversals`), not from its
position among the survivors, so filtering never desynchronizes the normal/reverse
labeling from the true alternating sequence.
"""
function chrons(ds::SynthesisDataset; polarity::Symbol=:normal, min_length::Real=0.0)
    @assert polarity ∈ (:normal, :reverse) "polarity must be :normal or :reverse"
    isempty(ds.reversals) && return PolarityDataset(ds.name, Float64[], Float64[], polarity)

    kept_times, kept_indices = filter_short_reversals(ds.reversals, ds.time[1], ds.time[end];
                                                        min_length = min_length)
    isempty(kept_times) && return PolarityDataset(ds.name, Float64[], Float64[], polarity)

    edges      = [ds.time[1]; kept_times; ds.time[end]]
    first_sign = sign(ds.value[1].value) >= 0 ? :normal : :reverse
    opposite   = first_sign == :normal ? :reverse : :normal

    start = Float64[]
    stop  = Float64[]
    for j in 1:(length(edges) - 1)
        # number of *original* toggles that occurred before this segment started
        toggles_before = j == 1 ? 0 : kept_indices[j - 1]
        seg_polarity   = iseven(toggles_before) ? first_sign : opposite
        if seg_polarity == polarity
            # Merging can leave two *consecutive* segments with the same computed
            # polarity (dropping an odd run of reversals shifts every later segment's
            # toggle parity by one) — when that happens to the polarity we're
            # extracting, coalesce them into one continuous chron instead of emitting
            # two entries that touch at a zero-length gap (which corrupts H_statistic,
            # forcing spurious H = 0 / H = 1 values).
            if !isempty(stop) && stop[end] == edges[j]
                stop[end] = edges[j + 1]
            else
                push!(start, edges[j])
                push!(stop, edges[j + 1])
            end
        end
    end
    return PolarityDataset(ds.name, start, stop, polarity)
end

"""
    simulate_chrons(; model, tspan=(-2.0, 0.0), burnin=0.0, polarity=:normal,
                     min_length=0.0, kwargs...) -> PolarityDataset

Run a single realization of `model` over `tspan` padded by `burnin` Ma at the start
(to let transient behavior settle before recording), and return the chrons matching
`polarity` that start at or after `tspan[1]` as a `PolarityDataset`. The last chron is
dropped if it is right-censored by `tspan[2]` (i.e. does not end in a detected
reversal). Use `chron_lengths` to get the durations (Ma) out of the result.

`min_length` (Ma) discards any detected reversal that would create a chron shorter than
it, merging that excursion into its neighbor instead of counting it as its own chron
(see `chrons`/`filter_short_reversals`); default `0.0` keeps every detected reversal.

Only the two `saveat` endpoints are actually saved: reversal-detection accuracy does
not depend on `saveat` since `reversal_callback` uses the solver's own continuous
interpolant, so this avoids storing the dense trajectory for a run whose values are
otherwise unused. `kwargs` are forwarded to `simulate` (e.g. `h0`, `criterion`,
model parameters); `detect_reversals` and `saveat` are set internally and must not be
passed.
"""
function simulate_chrons(; model::Symbol, tspan = (-2.0, 0.0), burnin::Float64 = 0.0,
                          polarity::Symbol = :normal, min_length::Real = 0.0, kwargs...)
    padded_tspan = (tspan[1] - burnin, tspan[2])
    ds = simulate(; model = model, tspan = padded_tspan,
                  saveat = [padded_tspan[1], padded_tspan[2]],
                  detect_reversals = true, kwargs...)

    all_chrons = chrons(ds; polarity = polarity, min_length = min_length)
    keep       = findall(>=(tspan[1]), all_chrons.start)
    if !isempty(all_chrons.stop) && all_chrons.stop[end] == ds.time[end]
        keep = filter(!=(length(all_chrons.start)), keep)
    end

    return PolarityDataset(ds.name, all_chrons.start[keep], all_chrons.stop[keep], polarity)
end

"""
    chron_lengths(pd::PolarityDataset) -> Vector{Float64}

Durations (Ma) of each chron in `pd`.
"""
chron_lengths(pd::PolarityDataset) = pd.stop .- pd.start
