using MyDMRGPkg
using TensorOperations
using LinearAlgebra
using Plots

function prepare_mps(mpo::MPO, N::Int, d::Int, D_small::Int, D_large::Int)
    mps = MPS{Float64}(N, d, D_small)
    r2l_LQ!(mps)
    if D_large > D_small
        _, _ = DMRG_loop_2site!(mps, mpo, 2, -1.)
        mps_padding!(mps, D_large)
        r2l_LQ!(mps)
    end
    return mps
end
# %%

N = 24 # number of sites
d = 2 # physical dim
D = 30 # bond dim
J = 1
h = 0.5
max_loops = 3

mpo = nearest_ising_MPO(N, J, h)

mps = prepare_mps(mpo, N, d, 20, D)

@time λs, trunc_errors = DMRG_loop_2site!(mps, mpo, max_loops, -1.)
co = crosscap_overlap(mps)
println("N=$N, ", "D=$D, ", "h=$h, ", abs2(co))
plot(trunc_errors)

# E_dmrg = λs[end]
# E_exact = nearest_ising_ground_exact(N, J, h)

# println("DMRG Final Energy:   ", E_dmrg)
# println("Exact ground Energy: ", E_exact)
# println("variance: ", mpo_variance(mps, mpo))


# %%

J = 1
d = 2
D = 40
max_loops = 4
N_lst = [20, 32, 48, 60]
h_lst = collect(0.5:0.1:1.5)
co2_N_h = zeros(length(N_lst), length(h_lst))

for i in eachindex(N_lst)
    N = N_lst[i]
    mps = MPS{Float64}(N, d, D)
    r2l_LQ!(mps)
    for j in eachindex(h_lst)
        h = h_lst[j]
        mpo = nearest_ising_MPO(N, J, h)
        λs, _ = DMRG_loop_2site!(mps, mpo, max_loops, -1.)
        co = crosscap_overlap(mps)
        co2_N_h[i, j] = abs2(co)
        println("N=$N, ", "h=$h, ", abs2(co))
    end
end

# %%
p = plot(xlabel="h", ylabel="Crosscap Overlap")
for i in eachindex(N_lst)
    N = N_lst[i]
    plot!(h_lst, co2_N_h[i, :], label="N=$N")
end
display(p)