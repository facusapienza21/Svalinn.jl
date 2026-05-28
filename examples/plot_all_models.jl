using GeoReversal
using Plots

# Simulate all four models over 10 Ma at 2 kyr resolution
tspan  = (-10.0, 0.0)   # Ma; negative = past
saveat = 0.002           # Ma (2 kyr)

models = [:G12, :P09, :DW, :Davis2021]

datasets = [
    simulate(; model = :G12,       tspan = tspan, saveat = saveat),
    simulate(; model = :P09,       tspan = tspan, saveat = saveat),
    simulate(; model = :DW,        tspan = tspan, saveat = saveat),
    simulate(; model = :Davis2021, tspan = tspan, saveat = saveat, seed = 47),
]

titles = [
    "G12  —  Gissinger (2012)",
    "P09  —  Petrelis et al. (2009)",
    "DW   —  Buffett, Ziegler & Constable (2013)",
    "Davis2021  —  Davis et al. (2021)",
]

ylabels = [
    "D / √2  (norm.)",
    "R cos(u + δ)  (norm.)",
    "x / x̄",
    "X / X̄",
]

subplots = map(1:4) do i
    ds   = datasets[i]
    vals = [v.value for v in ds.value]
    p = plot(
        ds.time, vals;
        title     = titles[i],
        titlefont = font(9),
        ylabel    = ylabels[i],
        legend    = false,
        lw        = 0.4,
        color     = :black,
        xticks    = (i == 4 ? :auto : nothing),
        xlabel    = (i == 4 ? "Time (Ma)" : ""),
    )
    hline!(p, [0.0]; color = :firebrick, lw = 0.6, ls = :dash)
    p
end

fig = plot(
    subplots...;
    layout      = (4, 1),
    size        = (900, 900),
    left_margin = 8Plots.mm,
    link        = :x,
)

mkpath(joinpath(@__DIR__, "figures"))
savefig(fig, joinpath(@__DIR__, "figures", "all_models_10Ma.png"))
display(fig)
