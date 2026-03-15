using MyDMRGPkg, CUDA, cuTENSOR, LinearMaps
using MyDMRGPkg: cu, cpu
using Plots
using JLD2

# %%

N = 40 # number of sites
d = 2 # physical dim
D = 60 # bond dim
h = 1
α = 3
max_loops = 3

mpo = long_range_transverse_ising_MPO(N, α, h)
mps = MPS{Float64}(N, d, D)
r2l_LQ!(mps)

mpo_gpu = cu(mpo)
mps_gpu = cu(mps)

λs, trunc_errors = DMRG_loop_2site!(mps_gpu, mpo_gpu, max_loops, -1.; store_all=false, show_progress=true)
mps = cpu(mps_gpu)
E_dmrg = λs[end]

println("DMRG Final Energy:   ", E_dmrg)
println("variance: ", mpo_variance(mps, mpo))

# %%

max_loops = 4
α = 3
N_lst = [20, 32, 48, 60]
D_lst = [30, 40, 60, 80]
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
        mpo = long_range_transverse_ising_MPO(N, α, h)
        mpo_gpu = cu(mpo)
        λs, _ = DMRG_loop_2site!(mps_gpu, mpo_gpu, max_loops, -1.; store_all=false, show_progress=false, show_vram=false)
        mps = cpu(mps_gpu)
        co = crosscap_overlap(mps)
        co2_N_h[i, j] = abs2(co)
    end
end

jld2_path = "./examples/jld2_data/long_range_transverse_ising/" * "crosscap.jdl2"
@save jld2_path co2_N_h, N_lst, h_lst
