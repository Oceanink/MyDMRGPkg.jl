using MyDMRGPkg, CUDA, cuTENSOR
using MyDMRGPkg: cu, cpu
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
D = 30 # bond dim
J = 1
h = 0.5
max_loops = 3

mpo = transverse_ising_MPO(N, J, h)
# mpo_gpu = cu(mpo)
mps = MPS{Float64}(N, d, D)
r2l_LQ!(mps)
# mps_gpu = cu(mps);
# λs, trunc_errors = DMRG_loop_2site!(mps_gpu, mpo_gpu, max_loops, -1; store_all=false, show_vram=false);
λs, trunc_errors = DMRG_loop_2site!(mps, mpo, max_loops, -1; use_gpu=true);

# mps = cpu(mps_gpu)
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
max_loops = 4
N_lst = [20, 32, 48, 60]
D_lst = [30, 40, 80, 80]
h_lst = collect(0.99:0.002:1.01)
# h_lst = collect(1.5:-0.05:0.8)
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
        λs, _ = DMRG_loop_2site!(mps_gpu, mpo_gpu, max_loops, 1e-12)
        # jld2_path = "./examples/jld2_data/transverse_ising_mps_data_cuda/" * "N$N" * "_h$h" * "_D$D" * "_mps.jdl2"
        mps = cpu(mps_gpu)
        # @save jld2_path mps
        co = crosscap_overlap(mps)
        co2_N_h[i, j] = abs2(co)
        # println("N=$N, ", "h=$h, ", abs2(co))
    end
end

# %%
# p = plot(xlabel="h", ylabel="Crosscap Overlap")
# for i in eachindex(N_lst)
#     N = N_lst[i]
#     plot!(h_lst, co2_N_h[i, :], label="N=$N", marker=:circle, markersize=2)
# end
# savefig(p, "./examples/img/transverse_ising_crosscap(cuda).png")
# jld2_path = "./examples/jld2_data/transverse_ising/crosscap.jl2"
# @save jld2_path co2_N_h N_lst h_lst
# %%
# jld2_path = "./examples/jld2_data/transverse_ising/crosscap.jl2"
# @load jld2_path co2_N_h N_lst h_lst

s_N_h = zeros(length(N_lst), length(h_lst))
for i in eachindex(N_lst)
    s_N_h[i, :] = (h_lst .- 1) * N_lst[i]
end

p = plot(xlabel="(h-1)N", ylabel="Crosscap Overlap")
marker_N = [:cross, :xcross, :rect, :circle]
for i in eachindex(N_lst)
    N = N_lst[i]
    scatter!(s_N_h[i, :], co2_N_h[i, :], label="N=$N", marker=marker_N[i], markersize=3)
end
display(p)


# %%
# s_N_lst = zeros(length(N_lst), length(h_lst))

# for i in eachindex(N_lst)
#     s = (h_lst .- 1) * N_lst[i]^(15 / 8)
#     s_N_lst[i, :] = (h_lst .- 1) * N_lst[i]^(15 / 8)
# end

