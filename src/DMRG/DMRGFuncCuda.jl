export DMRG_loop_2site_cuda!, cu

using LinearMaps

function cu(mpo::MPO{T}) where T
    O_gpu = [CuArray(W) for W in mpo.O]
    return CuMPO{T}(O_gpu, mpo.N, mpo.d)
end

function cu(mpo::MPO, dtype::Type)
    O_gpu = [CuArray(dtype.(W)) for W in mpo.O]
    return CuMPO{dtype}(O_gpu, mpo.N, mpo.d)
end

function cu(mps::MPS{T}) where T
    A_gpu = [CuArray(W) for W in mps.A]
    return CuMPS{T}(A_gpu, mps.N, mps.d)
end

function cu(mps::MPS, dtype::Type)
    A_gpu = [CuArray(dtype.(W)) for W in mps.A]
    return CuMPS{dtype}(A_gpu, mps.N, mps.d)
end

function cpu(mps::CuMPS{T}) where T
    A_cpu = [Array(A) for A in mps.A]
    return MPS{T}(A_cpu, mps.N, mps.d)
end

"""
    _estimate_vram_usage_gb(N, D, d, T)

Estimate VRAM usage in GB for two-site DMRG.

Accounts for:
- MPS/MPO storage
- Environment tensors (left and right)
- Two-site wavefunction and H_eff application
- Krylov subspace vectors (eigsolve)
- SVD workspace buffers
- Tensor contraction intermediates
"""
function _estimate_vram_usage_gb(N::Int, D::Int, d::Int, ::Type{T}) where T
    bytes_per_elem = sizeof(T)

    # Base storage
    mps_size = N * D^2 * d * bytes_per_elem
    mpo_size = N * D^2 * d^2 * bytes_per_elem
    env_size = 2 * (N - 1) * D^3 * bytes_per_elem

    # Two-site tensor (D*D, d*d) matrix
    two_site_size = D^2 * d^2 * bytes_per_elem

    # H_eff matvec temporaries (largest intermediate contraction)
    # left_env[u,j,v] * O1[j,k,i,b] * X[v,b,n,m] * O2[k,l,o,n] * right_env[p,l,m]
    # The largest intermediate is roughly D^4 * d^2
    heff_temp_size = D^4 * d^2 * bytes_per_elem

    # Krylov subspace: eigsolve keeps ~20-30 vectors of size D^2*d^2
    krylov_size = 20 * D^2 * d^2 * bytes_per_elem

    # SVD workspace: gesvdj needs ~5-10x the matrix size (D^2*d x D^2*d)
    svd_workspace_size = 5 * D^2 * d * D^2 * d * bytes_per_elem

    # Environment update temporaries (each @tensor creates temporaries)
    # For updating left_env: contraction of 4 tensors, intermediate ~D^4
    env_update_temp = 2 * D^4 * d * bytes_per_elem

    # Peak memory (during first sweep when all environments are computed)
    total_bytes = mps_size + mpo_size + env_size + two_site_size +
                  heff_temp_size + krylov_size + svd_workspace_size + env_update_temp

    return total_bytes / (1024^3)
end

"""
    _maybe_gc_collect()

Check GPU memory usage and trigger garbage collection if needed.
Triggers when memory usage exceeds 80% of total VRAM.
"""
function _maybe_gc_collect()
    free_mem = CUDA.free_memory()
    total_mem = CUDA.total_memory()
    used_mem = total_mem - free_mem
    usage_ratio = used_mem / total_mem

    println("vram usage: ", used_mem / 2^30, " GB")
    if usage_ratio > 0.8
        GC.gc(true)
        CUDA.reclaim()
    end
end

"""
    l2r_DMRG_prep_2site_cuda(mps::CuMPS{T}, mpo::CuMPO{T}) where T

Prepare right environments for the first left-to-right sweep on GPU.
"""
function l2r_DMRG_prep_2site_cuda(mps::CuMPS{T}, mpo::CuMPO{T}) where T
    N = mps.N
    right_envs = Vector{CuArray{T,3}}(undef, N - 1)
    right_envs[N-1] = CUDA.ones(T, 1, 1, 1)

    for n in N:-1:3
        On = mpo.O[n]
        An = mps.A[n]
        right_env = right_envs[n-1]
        @tensor right_env[u, y, j] := right_env[o, p, l] * conj(An)[u, i, o] * On[y, p, i, k] * An[j, k, l]
        right_envs[n-2] = right_env
    end
    return right_envs
end

"""
    _H_eff_matfree!(y, x, left_env, O1, O2, right_env, Dl, d, Dr)

Matrix-free application of H_eff to vector x, storing result in y.
H_eff |x> = left_env * O1 * O2 * right_env contracted with x.
"""
function _H_eff_matfree!(y::CuVector{T}, x::CuVector{T},
    left_env::CuArray{T,3}, O1::CuArray{T,4}, O2::CuArray{T,4}, right_env::CuArray{T,3},
    Dl::Int, d::Int, Dr::Int) where T

    X = reshape(x, Dl, d, d, Dr)

    @tensor Y[u, i, o, p] := left_env[u, j, v] * O1[j, k, i, b] * X[v, b, n, m] * O2[k, l, o, n] * right_env[p, l, m]

    copyto!(y, vec(Y))
    return y
end

"""
    _create_H_eff_map(left_env, O1, O2, right_env, Dl, d, Dr)

Create a LinearMap for matrix-free H_eff operator.
"""
function _create_H_eff_map(left_env::CuArray{T,3}, O1::CuArray{T,4}, O2::CuArray{T,4}, right_env::CuArray{T,3},
    Dl::Int, d::Int, Dr::Int) where T

    dim = Dl * d * d * Dr

    function matvec!(y, x)
        _H_eff_matfree!(y, x, left_env, O1, O2, right_env, Dl, d, Dr)
        return y
    end

    return LinearMap{T}(matvec!, dim; issymmetric=true, isposdef=false)
end

"""
    DMRG_1step_2site_cuda(left_env, O1, O2, right_env, D::Int, direction::String; x0=nothing)

Single two-site DMRG optimization step on GPU using matrix-free eigensolver.
"""
function DMRG_1step_2site_cuda(left_env::CuArray{T,3}, O1::CuArray{T,4}, O2::CuArray{T,4},
    right_env::CuArray{T,3}, D::Int, direction::String;
    x0::Union{Nothing,CuVector{T}}=nothing) where T

    @assert direction == "l2r" || direction == "r2l"

    Dl = size(left_env, 3)
    Dr = size(right_env, 3)
    d = size(O1, 4)

    H_map = _create_H_eff_map(left_env, O1, O2, right_env, Dl, d, Dr)

    dim = Dl * d * d * Dr

    if x0 !== nothing
        λs, vecs, _ = eigsolve(H_map, x0, 1, :SR, ishermitian=true, tol=1e-10)
    else
        x0_rand = CUDA.rand(T, dim) .- T(0.5)
        x0_rand ./= norm(x0_rand)
        λs, vecs, _ = eigsolve(H_map, x0_rand, 1, :SR, ishermitian=true, tol=1e-10)
    end

    λ = real(λs[1])

    B_vec = vecs[1]
    B_mat = reshape(B_vec, Dl * d, d * Dr)

    D_keep = min(Dl * d, d * Dr, D)

    U, S, V = CUDA.CUSOLVER.gesvdj!('V', D_keep, B_mat)

    e_trunc = D_keep < length(S) ? sum(abs2, S[D_keep+1:end]) : zero(real(T))

    S_trunc = S[1:D_keep]
    U_trunc = U[:, 1:D_keep]
    Vh_trunc = V'[1:D_keep, :]

    # Convert to CPU for reshape operations, keeping the diagonal multiplication on GPU first
    if direction == "l2r"
        # U is already orthonormal, use directly
        Al_cpu = reshape(collect(U_trunc), Dl, d, D_keep)
        # For Ar: need to multiply Vh' by S from left (Vh is already V')
        # Vh_trunc is D_keep × (d*Dr), we need to multiply S from left
        Ar_mat_gpu = Diagonal(S_trunc) * Vh_trunc
        Ar_cpu = reshape(collect(Ar_mat_gpu), D_keep, d, Dr)
    else
        # For Al: need to multiply U by S from right
        Al_mat_gpu = U_trunc * Diagonal(S_trunc)
        Al_cpu = reshape(collect(Al_mat_gpu), Dl, d, D_keep)
        # Vh is already orthonormal
        Ar_cpu = reshape(collect(Vh_trunc), D_keep, d, Dr)
    end

    return CuArray(Al_cpu), CuArray(Ar_cpu), λ, Float64(e_trunc)
end

"""
    l2r_DMRG_2site_cuda!(mps::CuMPS, mpo::CuMPO, right_envs, left_envs, λs, trunc_errors)

Left-to-right two-site DMRG sweep on GPU.
"""
function l2r_DMRG_2site_cuda!(mps::CuMPS{T}, mpo::CuMPO{T},
    right_envs::Vector{CuArray{T,3}},
    left_envs::Vector{CuArray{T,3}},
    λs::Vector{Float64},
    trunc_errors::Vector{Float64}) where T

    N = mps.N
    d = mps.d

    iter = ProgressBar(1:N-1)
    for n in iter
        left_env = left_envs[n]
        right_env = right_envs[n]
        O1 = mpo.O[n]
        O2 = mpo.O[n+1]
        D = size(mps.A[n], 3)

        Dl = size(left_env, 3)
        Dr = size(right_env, 3)

        x0 = nothing
        if n >= 2
            Dl_curr, d1, Dmid = size(mps.A[n])
            Dmid2, d2, Dr_curr = size(mps.A[n+1])
            if Dl_curr == Dl && Dr_curr == Dr && d1 == d && d2 == d && Dmid == Dmid2
                @tensor B_curr[v, b, n_idx, m] := mps.A[n][v, b, k] * mps.A[n+1][k, n_idx, m]
                x0 = vec(B_curr)
            end
        end

        Al, Ar, λ, e_trunc = DMRG_1step_2site_cuda(left_env, O1, O2, right_env, D, "l2r"; x0=x0)
        set_description(iter, string(@sprintf("λ: %.6f", λ)))

        mps.A[n] = Al
        mps.A[n+1] = Ar
        λs[n] = λ
        trunc_errors[n] = e_trunc

        if n <= N - 2
            @tensor left_env_new[o, p, l] := left_env[u, y, j] * conj(Al)[u, i, o] * O1[y, p, i, k] * Al[j, k, l]
            left_envs[n+1] = left_env_new
        end
    end

    return nothing
end

"""
    r2l_DMRG_2site_cuda!(mps::CuMPS, mpo::CuMPO, left_envs, right_envs, λs, trunc_errors)

Right-to-left two-site DMRG sweep on GPU.
"""
function r2l_DMRG_2site_cuda!(mps::CuMPS{T}, mpo::CuMPO{T},
    left_envs::Vector{CuArray{T,3}},
    right_envs::Vector{CuArray{T,3}},
    λs::Vector{Float64},
    trunc_errors::Vector{Float64}) where T

    N = mps.N
    d = mps.d

    iter = ProgressBar(N:-1:2)
    for n in iter
        left_env = left_envs[n-1]
        right_env = right_envs[n-1]
        O1 = mpo.O[n-1]
        O2 = mpo.O[n]
        D = size(mps.A[n-1], 3)

        Dl = size(left_env, 3)
        Dr = size(right_env, 3)

        x0 = nothing
        if n <= N - 1
            Dl_curr, d1, Dmid = size(mps.A[n-1])
            Dmid2, d2, Dr_curr = size(mps.A[n])
            if Dl_curr == Dl && Dr_curr == Dr && d1 == d && d2 == d && Dmid == Dmid2
                @tensor B_curr[v, b, n_idx, m] := mps.A[n-1][v, b, k] * mps.A[n][k, n_idx, m]
                x0 = vec(B_curr)
            end
        end

        Al, Ar, λ, e_trunc = DMRG_1step_2site_cuda(left_env, O1, O2, right_env, D, "r2l"; x0=x0)
        set_description(iter, string(@sprintf("λ: %.6f", λ)))

        mps.A[n-1] = Al
        mps.A[n] = Ar
        λs[N+1-n] = λ
        trunc_errors[N+1-n] = e_trunc

        if n >= 3
            @tensor right_env_new[u, y, j] := right_env[o, p, l] * conj(Ar)[u, i, o] * O2[y, p, i, k] * Ar[j, k, l]
            right_envs[n-2] = right_env_new
        end
    end

    return nothing
end

"""
    DMRG_loop_2site_cuda!(mps::CuMPS{T}, mpo::CuMPO{T}, times::Int, threshold::Real) where T

Main GPU two-site DMRG loop with automatic memory management.

Parameters:
- mps: Initial MPS on GPU (will be modified in-place)
- mpo: MPO on GPU
- times: Maximum number of DMRG loops
- threshold: Convergence threshold for energy difference

Returns:
- λs_all: Energy values at each update step
- trunc_errs_all: Truncation errors at each update step
"""
function DMRG_loop_2site_cuda!(mps::CuMPS{T}, mpo::CuMPO{T}, times::Int, threshold::Real) where T
    N = mps.N

    vram_estimate = _estimate_vram_usage_gb(N, maximum(size.(mps.A, 3)), mps.d, T)
    println("Estimated VRAM usage: $(round(vram_estimate, digits=2)) GB")

    d = mps.d
    D_max = maximum([size(A, 3) for A in mps.A])

    left_envs = Vector{CuArray{T,3}}(undef, N - 1)
    left_envs[1] = CUDA.ones(T, 1, 1, 1)
    right_envs = l2r_DMRG_prep_2site_cuda(mps, mpo)

    max_size = times * 2 * (N - 1)
    λs_all = Vector{Float64}(undef, max_size)
    trunc_errs_all = Vector{Float64}(undef, max_size)

    λs = Vector{Float64}(undef, N - 1)
    trunc_errs = Vector{Float64}(undef, N - 1)

    idx = 0
    i = 0
    e = Inf

    while i < times && e > threshold
        println("DMRG loop $(i+1), left-to-right sweep...")
        l2r_DMRG_2site_cuda!(mps, mpo, right_envs, left_envs, λs, trunc_errs)
        copyto!(λs_all, idx + 1, λs, 1, N - 1)
        λ_lr = λs[N-1]
        copyto!(trunc_errs_all, idx + 1, trunc_errs, 1, N - 1)
        idx += N - 1

        _maybe_gc_collect()

        println("DMRG loop $(i+1), right-to-left sweep...")
        r2l_DMRG_2site_cuda!(mps, mpo, left_envs, right_envs, λs, trunc_errs)
        copyto!(λs_all, idx + 1, λs, 1, N - 1)
        λ_rl = λs[N-1]
        copyto!(trunc_errs_all, idx + 1, trunc_errs, 1, N - 1)
        idx += N - 1

        _maybe_gc_collect()

        e = abs(λ_lr - λ_rl)
        # println("Energy difference: $(e)")

        i += 1
    end

    resize!(λs_all, idx)
    resize!(trunc_errs_all, idx)

    GC.gc(true)
    CUDA.reclaim()

    return λs_all, trunc_errs_all
end
