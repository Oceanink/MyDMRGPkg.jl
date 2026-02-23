using MyDMRGPkg
using TensorOperations
using LinearAlgebra

function nearest_ising_ground_exact(N::Int, J, h)
    g = h / J
    m_lst = collect(1:2:2N-1)
    E_even = -J * sum(sqrt.(1 .+ g^2 .- 2g * cospi.(m_lst / N)))
    m_lst = collect(0:2:2N-2)
    E_odd = -J * sum(sqrt.(1 .+ g^2 .- 2g * cospi.(m_lst / N)))
    E = min(E_even, E_odd)
    return E
end

N = 40 # number of sites
d = 2 # physical dim
D = 30 # bond dim
J = 1
h = 1.2
max_loops = 3

mpo = nearest_ising_MPO(N, J, h)

mps = MPS{Float64}(N, d, D)
r2l_LQ!(mps)

@time λs, trunc_errors = DMRG_loop_2site!(mps, mpo, max_loops, -1.)
E_dmrg = λs[end]
E_exact = nearest_ising_ground_exact(N, J, h)

println("DMRG Final Energy:   ", E_dmrg)
println("Exact ground Energy: ", E_exact)
println("variance: ", mpo_variance(mps, mpo))