using MyDMRGPkg
using Plots, Printf

# %% DMRG sweep

N = 50 # number of sites
d = 2 # physical dim
D = 50 # bond dim
max_loops = 3
max_sweeps = max_loops * 2

BC = "PBC"
E_Bethe = heisen_chain_Bethe(N, BC)
# generate mpo of N-site PBC heisenberg chain
mpo = heisen_chain_MPO(N, BC)

# generate random mps
mps_rnd = MPS{Float64}(N, d, D)
r2l_LQ!(mps_rnd)

@time λs, trunc_errors = DMRG_loop_2site!(mps_rnd, mpo, max_loops, -1.)
E_dmrg = λs[end]

println("DMRG Final Energy:   ", E_dmrg)
println("Bethe Ansatz Energy: ", E_Bethe)
println("variance: ", mpo_variance(mps_rnd, mpo))

# %% plot truncation errors
p = plot(trunc_errors; label="two-site DMRG truncation errors", xlabel="update steps", ylabel="truncation errors",
    linewidth=2, marker=:circle, markersize=2)
title!(p, "N=$N, D=$D, $BC")

# %% plot relative errors
E_errs = abs.((λs .- E_Bethe) / E_Bethe)
p = plot(E_errs; label="two-site DMRG relative errors", xlabel="update steps", ylabel="relative error",
    linewidth=2, marker=:circle, markersize=2)

title!(p, "$N sites, $D bond dim, $BC, error = $(round(E_errs[end], digits=4))")

# %%

λ_sweeps = ones(max_sweeps)
for i in 1:max_sweeps
    λ_sweeps[i] = λs[i*(N-1)]
end
E_errs_sweeps = abs.((λ_sweeps .- E_Bethe) / E_Bethe)
p = plot(E_errs_sweeps; label="two-site DMRG relative errors", xlabel="sweeps", ylabel="relative error", linewidth=2, marker=:circle, markersize=2)
title!(p, "$N sites, $D bond dim, $BC, error = $(round(E_errs_sweeps[end], digits=4))")
