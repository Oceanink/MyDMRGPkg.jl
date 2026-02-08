# MyDMRGPkg

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Oceanink.github.io/MyDMRGPkg.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Oceanink.github.io/MyDMRGPkg.jl/dev/)

<!-- [![Build Status](https://github.com/Oceanink/MyDMRGPkg.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/Oceanink/MyDMRGPkg.jl/actions/workflows/CI.yml?query=branch%3Amaster) -->

[![Coverage](https://codecov.io/gh/Oceanink/MyDMRGPkg.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Oceanink/MyDMRGPkg.jl)

`MyDMRGPkg` is a Julia package for Density Matrix Renormalization Group (DMRG) on the spin-1/2 Heisenberg chain.
It provides MPS/MPO data structures, and both one-site and two-site DMRG solvers.

## Features

- MPS and MPO tensor-network structures (`MPS`, `MPO`)
- Heisenberg-chain Hamiltonian as an MPO (`heisen_chain_MPO`) for `"OBC"` or `"PBC"`
- Bethe-Ansatz reference ground-state energy (`heisen_chain_Bethe`)
- One-site DMRG (`DMRG_loop!`, `DMRG_converge!`)
- Two-site DMRG with truncation diagnostics (`DMRG_loop_2site!`)
- Canonicalization and checks (`r2l_LQ!`, `is_left_canonical`, `is_right_canonical`)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Oceanink/MyDMRGPkg.jl")
```

## Quick Start (One-Site DMRG)

```julia
using MyDMRGPkg

N = 40
d = 2
D = 20
BC = "PBC"

mps = MPS{Float64}(N, d, D)
r2l_LQ!(mps)  # prepare right-canonical initial state

mpo = heisen_chain_MPO(N, BC)
energies = DMRG_loop!(mps, mpo, 2, 1e-12)

E_bethe = heisen_chain_Bethe(N, BC)
rel_err = abs((energies[end] - E_bethe) / E_bethe)
println("Final energy: ", energies[end])
println("Relative error vs Bethe: ", rel_err)
```

## Quick Start (Two-Site DMRG)

```julia
using MyDMRGPkg

N = 40
d = 2
D = 20
BC = "PBC"

mps = MPS{Float64}(N, d, D)
r2l_LQ!(mps)

mpo = heisen_chain_MPO(N, BC)
energies, trunc_errors = DMRG_loop_2site!(mps, mpo, 2, 1e-12)

println("Final energy: ", energies[end])
println("Last truncation error: ", trunc_errors[end])
```

## Running Tests

From the package root:

```julia
using Pkg
Pkg.test()
```

The tests include both one-site and two-site DMRG checks and generate plots under `test/output/`.

## Notes

- Call `r2l_LQ!` before DMRG sweeps to canonicalize the initial MPS.
- `DMRG_loop!` returns energy values per local update.
- `DMRG_loop_2site!` returns `(energies, trunc_errors)`.

## License

This project is distributed under the terms of the `LICENSE` file in this repository.
