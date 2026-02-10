using MyDMRGPkg
using Plots, Printf

function xxz_crosscap_overlay_exact(Δ::Float64)
    @assert -1 < Δ && Δ < 1
    K = pi / (2 * (pi - acos(Δ)))
    return 1 / sqrt(K)
end


function xxz_crosscap_overlay(mps::MPS, Δ::Float64)
    @assert -1 < Δ && Δ < 1

end

# %% DMRG sweep

N = 100 # number of sites
d = 2 # physical dim
D = 20 # bond dim
max_loops = 2
E_Bethe = heisen_chain_Bethe(N, BC)

Δ = 0.5
BC = "PBC"
mpo = xxz_chain_MPO(N, Δ, BC)


