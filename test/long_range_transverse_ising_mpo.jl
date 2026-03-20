using Test
using MyDMRGPkg
using LinearAlgebra
using TensorOperations

@testset "Long Range Transverse Ising MPO exact check" begin
    N = 3
    α = 1.0
    h = 0.5
    mpo = long_range_transverse_ising_MPO(N, α, h)

    # Contract MPO to get full Hamiltonian matrix
    # Following same pattern as haldane_shastry_mpo.jl
    H1 = mpo.O[1][1, :, :, :]  # (D2, d, d)
    HN = mpo.O[N][:, 1, :, :]  # (DN, d, d)
    
    if N == 3
        @tensor H[u, i, o, v, b, n] := H1[j, u, v] * mpo.O[2][j, k, i, b] * HN[k, o, n]
        H_mpo = reshape(H, 2^N, 2^N)
    else
        error("Only N=3 is implemented in this test")
    end
    
    H_mat = long_range_transverse_ising_H_matrix(N, α, h)

    e = norm(H_mpo - H_mat)
    @test e < 1e-9
end
