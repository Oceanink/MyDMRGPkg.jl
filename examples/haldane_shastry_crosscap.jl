using MyDMRGPkg
using TensorOperations
using LinearAlgebra
using Plots, Printf
using SpecialFunctions

function hs_crosscap_overlap_exact(M::Integer)
    # log(co2) = M*log(2) + 3*log(M!) - 2*log((M/2)!) - log((2M)!)
    log_co2 = M * log(2) + 3 * loggamma(M + 1) - 2 * loggamma(div(M, 2) + 1) - loggamma(2M + 1)
    return exp(log_co2)
end

N = 16
d = 2
mpo = haldane_shastry_MPO(N)


D = 40 # mps bond dim
mps = MPS{Float64}(N, d, D)
r2l_LQ!(mps)

max_loops = 4
_, _ = DMRG_loop_2site!(mps, mpo, max_loops, -1.)
co2 = abs2(crosscap_overlap(mps))

M = div(N, 2)
# co2_exact = 2^M * factorial(M)^3 / (factorial(div(M, 2))^2 * factorial(2 * M))
co2_exact = hs_crosscap_overlap_exact(M)

println("DMRG crosscap overlap: ", co2)
println("Exact crosscap overlap: ", co2_exact)
