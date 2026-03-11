using MyDMRGPkg
using LinearAlgebra
using Plots
using JLD2

# function prepare_mps(mpo::MPO, N::Int, d::Int, D_small::Int, D_large::Int)
#     mps = MPS{Float64}(N, d, D_small)
#     r2l_LQ!(mps)
#     if D_large > D_small
#         _, _ = DMRG_loop_2site!(mps, mpo, 2, -1.)
#         mps_padding!(mps, D_large)
#         r2l_LQ!(mps)
#     end
#     return mps
# end
# %%

N = 60 # number of sites
d = 2 # physical dim
D = 2 # bond dim
J = 1
h = 0.5
max_loops = 10

mpo = transverse_ising_MPO(N, J, h)
mpo_gpu = cu(mpo)
mps = MPS{Float64}(N, d, D)
r2l_LQ!(mps)
mps_gpu = cu(mps);
λs, trunc_errors = DMRG_loop_2site_cuda!(mps_gpu, mpo_gpu, max_loops, -1);
# λs, trunc_errors = DMRG_loop_2site!(mps, mpo, max_loops, -1);

mps = cpu(mps_gpu)
co = crosscap_overlap(mps)
println("N=$N, ", "D=$D, ", "h=$h, ", abs2(co))

E_dmrg = λs[end]
E_exact = transverse_ising_ground_exact(N, J, h)

println("DMRG Final Energy:   ", E_dmrg)
println("Exact ground Energy: ", E_exact)


# %%

J = 1
d = 2
# D = 40
max_loops = 3
N_lst = [20, 32, 48, 60]
D_lst = [30, 40, 60, 100]
h_lst = collect(1.5:-0.05:0.5)
co2_N_h = zeros(length(N_lst), length(h_lst))

for i in eachindex(N_lst)
    N = N_lst[i]
    D = D_lst[i]
    mps = MPS{Float64}(N, d, D)
    r2l_LQ!(mps)
    mps_gpu = cu(mps)
    for j in eachindex(h_lst)
        h = h_lst[j]
        mpo = transverse_ising_MPO(N, J, h)
        mpo_gpu = cu(mpo)
        λs, _ = DMRG_loop_2site_cuda!(mps_gpu, mpo_gpu, max_loops, -1.)
        jld2_path = "./examples/jld2_data/transverse_ising_mps_data_cuda/" * "N$N" * "_h$h" * "_D$D" * "_mps.jdl2"
        mps = cpu(mps_gpu)
        @save jld2_path mps
        co = crosscap_overlap(mps)
        co2_N_h[i, j] = abs2(co)
        # println("N=$N, ", "h=$h, ", abs2(co))
    end
end

# %%
p = plot(xlabel="h", ylabel="Crosscap Overlap")
for i in eachindex(N_lst)
    N = N_lst[i]
    plot!(h_lst, co2_N_h[i, :], label="N=$N", marker=:circle, markersize=2)
end
savefig(p, "./examples/img/transverse_ising_crosscap(cuda).png")

# %%

N_lst = collect(20:4:60)
# h_lst = 