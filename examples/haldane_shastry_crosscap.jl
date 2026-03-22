using MyDMRGPkg
using Plots, Printf

# %% 

N = 60
d = 2
mpo = haldane_shastry_MPO(N)

max_loops = 4
D = 40 # mps bond dim

mps = MPS{Float64}(N, d, 20)
r2l_LQ!(mps)
_, _ = DMRG_loop_2site!(mps, mpo, 4, -1.; store_all=false)

mps_padding!(mps, D)
r2l_LQ!(mps)
_, _ = DMRG_loop_2site!(mps, mpo, max_loops, -1.; store_all=false)

co2 = abs2(crosscap_overlap(mps))

M = div(N, 2)
# co2_exact = 2^M * factorial(M)^3 / (factorial(div(M, 2))^2 * factorial(2 * M))
co2_exact = hs_crosscap_overlap_exact(M)

println("DMRG crosscap overlap: ", co2)
println("Exact crosscap overlap: ", co2_exact)

# %%

N_lst = collect(4:4:40)
M_lst = div.(N_lst, 2)
co2_exact_lst = hs_crosscap_overlap_exact.(M_lst)
co2_dmrg_lst = []

d = 2
D = 40
max_loops = 4

for N in N_lst
    mpo = haldane_shastry_MPO(N)
    if N > 16
        mps = MPS{Float64}(N, d, 20)
        r2l_LQ!(mps)
        _, _ = DMRG_loop_2site!(mps, mpo, max_loops, -1.)

        mps_padding!(mps, D)
        r2l_LQ!(mps)
        _, _ = DMRG_loop_2site!(mps, mpo, max_loops, -1.)
    else
        mps = MPS{Float64}(N, d, D)
        r2l_LQ!(mps)
        _, _ = DMRG_loop_2site!(mps, mpo, max_loops, -1.)
    end

    co2 = abs2(crosscap_overlap(mps))
    push!(co2_dmrg_lst, co2)
end

p = plot(N_lst, co2_exact_lst, label="Exact Crosscap Overlap", xlabel="N", ylabel="Crosscap Overlap", linewidth=2)
scatter!(p, N_lst, co2_dmrg_lst, label="DMRG Crosscap Overlap", marker=:circle, markersize=4)
title!(p, "Haldane Shastry model, two-site DMRG, D=$D")
xticks!(p, N_lst)
file_name = "hs_crosscap" * "_D$D"
savefig(p, "./examples/img/" * file_name)

# N=40, crosscap overlap error between julia and mathematica exact value.
println(338690048 / 240990435 - co2_exact_lst[end])

# @save "./examples/jld2_data/haldane_shastry_crosscap.jld2" N_lst co2_dmrg_lst