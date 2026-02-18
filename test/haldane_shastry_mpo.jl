using Test
using MyDMRGPkg
using LinearAlgebra
using TensorOperations

@testset "Haldane Shastry MPO exact check" begin
    N = 3
    mpo = haldane_shastry_MPO(N)

    H1 = mpo.O[1][1, :, :, :]
    HN = mpo.O[N][:, 1, :, :]
    @tensor H[u, i, o, v, b, n] := H1[j, u, v] * mpo.O[2][j, k, i, b] * HN[k, o, n]
    H_mpo = reshape(H, 2^N, 2^N)
    H_mat = haldane_shastry_H_matrix(N)

    e = norm(H_mpo - H_mat)
    @test e < 1e-9
end