# MyDMRGPkg

<!-- [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Oceanink.github.io/MyDMRGPkg.jl/dev/) -->

[![Build Status](https://github.com/Oceanink/MyDMRGPkg.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Oceanink/MyDMRGPkg.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/Oceanink/MyDMRGPkg.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Oceanink/MyDMRGPkg.jl)

`MyDMRGPkg` is a Julia package for Density Matrix Renormalization Group (DMRG) calculations on spin-1/2 quantum spin chains. It provides MPS/MPO data structures and both one-site and two-site DMRG solvers.

## Features

- **MPS/MPO structures**: `MPS{T}` and `MPO{T}` tensor network types with GPU support (`CuMPS{T}`, `CuMPO{T}`)
- **Supported Hamiltonians**:
  - Heisenberg chain (`heisen_chain_MPO`) with OBC/PBC
  - XXZ chain (`xxz_chain_MPO`) with anisotropy parameter Δ
  - Haldane-Shastry model (`haldane_shastry_MPO`) with long-range interactions
  - Transverse Ising model (`transverse_ising_MPO`, `long_range_transverse_ising_MPO`)
- **Bethe ansatz reference**: `heisen_chain_Bethe()` for ground-state energy comparison
- **One-site DMRG**: `DMRG_loop!()` (fixed sweeps) and `DMRG_converge!()` (adaptive convergence)
- **Two-site DMRG**: `DMRG_loop_2site!()` with truncation error tracking
- **GPU Two-site DMRG**: `DMRG_loop_2site_cuda!()` with linearmap eigensolver

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Oceanink/MyDMRGPkg.jl")
```

## Quick Start (GPU Two-Site DMRG)

```julia
using MyDMRGPkg
using MyDMRGPkg:cu, cpu
using CUDA, cuTENSOR, LinearMaps

N = 60
d = 2
D = 100

# Create MPS and MPO on CPU
mps = MPS{Float64}(N, d, D)
r2l_LQ!(mps)
mpo = xxz_chain_MPO(N, -0.5, "PBC")

# Transfer to GPU
mps_gpu = cu(mps)
mpo_gpu = cu(mpo)

# Run GPU DMRG
energies, trunc_errors = DMRG_loop_2site!(mps_gpu, mpo_gpu, max_loops, 1e-12; store_all=true, show_progress=true, show_vram=true)

# Transfer back to CPU if needed
mps_cpu = cpu(mps_gpu)
println("Final energy: ", energies[end])
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
energies, trunc_errors = DMRG_loop_2site!(mps, mpo, 2, 1e-12; store_all=true, show_progress=true)

println("Final energy: ", energies[end])
println("Last truncation error: ", trunc_errors[end])
```

## XXZ Chain Example

```julia
using MyDMRGPkg

N = 40
Δ = 1.0  # anisotropy parameter (Δ=1 gives Heisenberg limit)
BC = "PBC"

mps = MPS{Float64}(N, 2, 20)
r2l_LQ!(mps)

mpo = xxz_chain_MPO(N, Δ, BC)
energies = DMRG_loop!(mps, mpo, 2, 1e-12)
println("Ground state energy: ", energies[end])
```

## Running Tests

From the package root:

```julia
using Pkg
Pkg.test()
```

Tests include both one-site and two-site DMRG checks and generate plots under `test/output/`.

## Notes

- Always call `r2l_LQ!(mps)` before DMRG sweeps to canonicalize the initial MPS
- `DMRG_loop!` returns energy values per local update
- `DMRG_loop_2site!` returns `(energies, trunc_errors)`
- `DMRG_converge!` returns `(final_energy, sites_updated)` where `sites_updated = -1` if max limit reached without convergence

- **CUDA Toolkit**: Configure with `CUDA.set_runtime_version!(v"12.8"; local_toolkit=false)` to use Julia's artifact CUDA
- **Memory Management**: Automatic garbage collection triggers when VRAM usage exceeds 80%. Call `CUDA.reclaim()` manually if needed.

## License

This project is distributed under the terms of the `LICENSE` file in this repository.
