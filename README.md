# Svalinn

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://facusapienza21.github.io/Svalinn.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://facusapienza21.github.io/Svalinn.jl/dev/)
[![Build Status](https://github.com/facusapienza21/Svalinn.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/facusapienza21/Svalinn.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/facusapienza21/Svalinn.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/facusapienza21/Svalinn.jl)

Svalinn is a Julia package for testing stochastic and dynamical models of geomagnetic polarity reversals against the paleomagnetic record.
It simulates axial-dipole models, detects reversals and chrons in their trajectories, and compares them with observed polarity timescales and paleointensity records through statistical tests.

## What's in the package

- **Models** (`simulate(; model = ...)`):
  - `:Davis2021`: bistable dipole driven by colored (Ornstein–Uhlenbeck) noise with
    correlation time `θ`, after Davis & Buffett (2021). Integrated with an exact OU
    update (Gillespie, 1996) and a Strang splitting of the drift, so the step size does
    not depend on `θ`; `θ = 0` gives the white-noise limit.
  - `:DW`: stochastic double-well model (Buffett, Ziegler & Constable, 2013).
  - `:P09`: stochastic phase model (Pétrélis et al., 2009).
  - `:G12`: three-component deterministic geodynamo model (Gissinger, 2012).
- **Data**: readers for polarity timescales (e.g. CK95, GPTS 2012), paleointensity
  syntheses (PADM2M, Sint-2000) and relative paleointensity records.
- **Reversals and chrons**: reversal detection during integration (zero crossing or
  hysteresis with threshold `h0`), and conversion to chron sequences.
- **Statistics**: the `H` statistic and Poisson/KS tests for clustering of chrons.