using MyDMRGPkg
using TensorOperations
using Plots, Printf

function xxz_luttinger_exact(Δ::Float64)
    @assert -1 < Δ && Δ <= 1
    K = pi / (2 * (pi - acos(Δ)))
    return K
end

function xxz_crosscap_overlay(mps::MPS)
    N = mps.N
    @assert N % 4 == 0

    half_N = div(N, 2)
    A1 = mps.A[1][1, :, :]
    @tensor C[u, o, l] := A1[i, l] * mps.A[1+half_N][u, i, o]
    for n in 2:half_N-1
        @tensor C[u, y, p] := C[u, o, l] * mps.A[n][l, j, p] * mps.A[n+half_N][o, j, y]
    end
    AN = mps.A[N][:, :, 1]
    @tensor co = C[u, o, l] * mps.A[half_N][l, j, u] * AN[o, j]
    return co
end

# %% parameters

N = 20 # number of sites
d = 2 # physical dim
BC = "PBC"
Δ = -0.5
mpo = xxz_chain_MPO(N, Δ, BC);

# %% DMRG sweep

max_loops = 4
D = 30 # bond dim
mps = MPS{Float64}(N, d, D)
r2l_LQ!(mps)
_, _ = DMRG_loop_2site!(mps, mpo, max_loops, -1.)
co = xxz_crosscap_overlay(mps)

K_exact = xxz_luttinger_exact(Δ)
K_dmrg = 1 / co^4

println("Δ = $Δ, Exact K = ", K_exact)
println("D = $D, DMRG  K = ", K_dmrg)
# %%
N = 20 # number of sites
d = 2 # physical dim
D = 30
max_loops = 4
BC = "PBC"

Δs = collect(-0.9:0.1:1)
l = size(Δs, 1)
Ks_dmrg = zeros(l)

Δs_plt = collect(-0.9:0.01:1)
Ks_exact = xxz_luttinger_exact.(Δs_plt)

mps = MPS{Float64}(N, d, D)
r2l_LQ!(mps)
for i in 1:l
    Δ = Δs[i]

    mpo = xxz_chain_MPO(N, Δ, BC)
    _, _ = DMRG_loop_2site!(mps, mpo, max_loops, -1.)
    co = xxz_crosscap_overlay(mps)
    Ks_dmrg[i] = 1 / co^4
end

p = plot(Δs_plt, Ks_exact, label="Exact K", xlabel="Δ", ylabel="K", linewidth=2)
scatter!(p, Δs, Ks_dmrg, label="DMRG K", xlabel="Δ", ylabel="K", linewidth=2, marker=:circle, markersize=2)
title!(p, "XXZ chain, two-site DMRG, N=$N, D=$D")
file_name = "xxz_crosscap_K" * "_N$N" * "_D$D"
savefig(p, "./examples/img/" * file_name)