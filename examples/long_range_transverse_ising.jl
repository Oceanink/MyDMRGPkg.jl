using MyDMRGPkg
using TensorOperations
using LinearAlgebra

N = 20 # number of sites
d = 2 # physical dim
D = 30 # bond dim
h = 1
α = 1
max_loops = 3

mpo = long_range_transverse_ising_MPO(N, α, h)

mps = MPS{Float64}(N, d, D)
r2l_LQ!(mps)

@time λs, trunc_errors = DMRG_loop_2site!(mps, mpo, max_loops, -1.)
E_dmrg = λs[end]

println("DMRG Final Energy:   ", E_dmrg)
println("variance: ", mpo_variance(mps, mpo))