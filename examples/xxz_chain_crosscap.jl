using MyDMRGPkg
using TensorOperations
using Plots, Printf

function xxz_luttinger_exact(Δ::Float64)
    @assert -1 < Δ && Δ < 1
    K = pi / (2 * (pi - acos(Δ)))
    return K
end

function xxz_crosscap_overlay(mps::MPS)
    N = mps.N
    @assert N % 4 == 0

    half_N = div(N, 2)
    A1 = mps.A[1][1, :, :]
    C = nothing
    @tensor C[u, o, l] := A1[i, l] * mps.A[1+half_N][u, i, o]
    for n in 2:half_N-1
        @tensor C[u, y, p] := C[u, o, l] * mps.A[n][l, j, p] * mps.A[n+half_N][o, j, y]
    end
    AN = mps.A[N][:, :, 1]
    @tensor co = C[u, o, l] * mps.A[half_N][l, j, u] * AN[o, j]
    return co
end

# %% parameters

N = 100 # number of sites
d = 2 # physical dim
BC = "PBC"
Δ = 0.5
mpo = xxz_chain_MPO(N, Δ, BC);

# %% DMRG sweep

max_loops = 3
D = 80 # bond dim
mps = MPS{Float64}(N, d, D)
r2l_LQ!(mps)
_ = DMRG_loop!(mps, mpo, max_loops, -1.)
co = xxz_crosscap_overlay(mps)

K_exact = xxz_luttinger_exact(Δ)
K_dmrg = 1 / co^4

println("Δ = $Δ, Exact K = ", K_exact)
println("D = $D, DMRG  K = ", K_dmrg)