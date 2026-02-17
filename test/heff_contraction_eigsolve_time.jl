using Test
using MyDMRGPkg
using LinearAlgebra
using KrylovKit
using Random
using Printf


@testset "XXZ Chain H_eff Contraction and Eigensolve Benchmark" begin
    # Parameters
    N = 20          # Number of sites
    D = 40          # Bond dimension
    d = 2           # Physical dimension
    Delta = -0.9    # XXZ anisotropy
    BC = "PBC"      # Boundary condition

    println("\n" * "="^60)
    println("XXZ Chain H_eff Contraction and Eigensolve Benchmark")
    println("="^60)
    println("N = $N, D = $D, d = $d, Delta = $Delta, BC = $BC")

    # Generate XXZ chain MPO
    mpo = xxz_chain_MPO(N, Delta, BC)
    println("Generated XXZ chain MPO with $BC")

    # Generate random MPS
    Random.seed!(42)
    mps = MPS{Float64}(N, d, D)

    # Canonicalize MPS (right-to-left LQ)
    r2l_LQ!(mps)
    @test is_right_canonical(mps; atol=1e-8)
    println("Canonicalized MPS (right-canonical)")

    # Prepare right environments (following l2r_DMRG_prep_2site)
    right_envs = Vector{Array{Float64,3}}(undef, N - 1)
    right_envs[N-1] = ones(Float64, 1, 1, 1)

    for n in N:-1:3
        On = mpo.O[n]
        An = mps.A[n]
        right_env = right_envs[n-1]
        @tensor right_env[u, y, j] := right_env[o, p, l] * conj(An)[u, i, o] * On[y, p, i, k] * An[j, k, l]
        right_envs[n-2] = right_env
    end

    # Benchmark H_eff contraction and eigensolve for each site pair
    # Left environments are updated on-the-fly like in real DMRG
    println("\nBenchmarking H_eff contraction and eigensolve:")
    println("-"^60)

    total_contraction_time = 0.0
    total_eigsolve_time = 0.0

    contraction_times = Float64[]
    eigsolve_times = Float64[]
    eigenvalues = Float64[]

    # Initialize left environment
    left_env = ones(Float64, 1, 1, 1)

    for n in 1:N-1
        right_env = right_envs[n]
        O1 = mpo.O[n]
        O2 = mpo.O[n+1]

        # Get dimensions
        Dl = size(left_env, 3)
        Dr = size(right_env, 3)

        # Contract H_eff (from DMRG_1step_2site)
        contraction_time = @elapsed begin
            @tensor H_eff[u, i, o, p, v, b, n_idx, m] := left_env[u, j, v] * O1[j, k, i, b] * O2[k, l, o, n_idx] * right_env[p, l, m]
        end
        total_contraction_time += contraction_time

        # Get H_eff dimensions
        H_eff_size = size(H_eff)
        dim1 = prod(H_eff_size[1:4])
        dim2 = prod(H_eff_size[5:8])

        # Reshape to matrix
        H_eff_mat = reshape(H_eff, dim1, dim2)

        # Eigensolve
        eigsolve_time = @elapsed begin
            λs, vecs, _ = eigsolve(H_eff_mat, 1, :SR)
        end
        total_eigsolve_time += eigsolve_time

        λ = real(λs[1])

        push!(contraction_times, contraction_time)
        push!(eigsolve_times, eigsolve_time)
        push!(eigenvalues, λ)

        @printf("Site pair (%2d,%2d): Dl=%3d, Dr=%3d, H_eff: %5d x %5d\n",
            n, n + 1, Dl, Dr, dim1, dim2)
        @printf("  Contraction: %8.6f s, Eigensolve: %8.6f s\n",
            contraction_time, eigsolve_time)

        # Update left environment for next iteration (like in l2r_DMRG_2site!)
        if n <= N - 2
            # For benchmarking, use the contracted H_eff info
            # In real DMRG, this would use the updated Al tensor
            # Here we just continue with the loop for timing purposes
            @tensor left_env[o, p, l] := left_env[u, y, j] * conj(mps.A[n])[u, i, o] * O1[y, p, i, k] * mps.A[n][j, k, l]
        end
    end

    println("-"^60)
    @printf("Total H_eff contraction time: %.6f s\n", total_contraction_time)
    @printf("Total eigensolve time:        %.6f s\n", total_eigsolve_time)
    @printf("Total time:                   %.6f s\n", total_contraction_time + total_eigsolve_time)

    # Verify results are finite
    @test all(isfinite, contraction_times)
    @test all(isfinite, eigsolve_times)
    @test all(isfinite, eigenvalues)

    # Verify eigenvalues are reasonable (negative for ground state)
    @test all(<(0), eigenvalues)

    println("="^60)
end
