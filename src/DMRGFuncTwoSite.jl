export DMRG_loop_2site!

function l2r_DMRG_prep_2site(mps::MPS{T}, mpo::MPO) where {T}
    """Prepare right environments for the first left-to-right sweep"""
    N = mps.N
    right_envs = Vector{Array{T,3}}(undef, N - 1)
    right_envs[N-1] = ones(1, 1, 1)

    for n in N:-1:3
        On = mpo.O[n]
        An = mps.A[n]
        right_env = right_envs[n-1]
        @tensor right_env[u, y, j] := right_env[o, p, l] * conj(An)[u, i, o] * On[y, p, i, k] * An[j, k, l]
        right_envs[n-2] = right_env
    end
    return right_envs
end


function DMRG_1step_2site(left_env::Array{T,3}, O1::Array{T2,4}, O2::Array{T2,4}, right_env::Array{T,3}, D::Int, direction::String) where {T,T2}
    """Single two-site DMRG optimization step using iterative eigensolver
    """
    @assert direction == "l2r" || direction == "r2l"
    @tensor H_eff[u, i, o, p, v, b, n, m] := left_env[u, j, v] * O1[j, k, i, b] * O2[k, l, o, n] * right_env[p, l, m]

    H_eff_size = size(H_eff)
    dim1 = prod(H_eff_size[1:4])
    dim2 = prod(H_eff_size[5:8])
    Dl, d, _, Dr = H_eff_size[5:8]

    # Find only the smallest eigenvalue using iterative method
    # :SR means "smallest real" eigenvalue
    H_eff_mat = reshape(H_eff, dim1, dim2)
    λs, vecs, _ = eigsolve(H_eff_mat, 1, :SR)
    λ = real(λs[1])

    B_mat = reshape(vecs[1], Dl * d, d * Dr)
    U, S, V = svd(B_mat)
    D_keep = min(Dl * d, d * Dr, D)
    e_trunc = sum(S[D_keep+1:end] .^ 2)
    U_trunc = U[:, 1:D_keep]
    S_trunc = S[1:D_keep]
    Vh_trunc = V'[1:D_keep, :]

    if direction == "l2r"
        Al = reshape(U_trunc, Dl, d, D_keep)
        Ar = reshape(diagm(S_trunc) * Vh_trunc, D_keep, d, Dr)
    else
        Al = reshape(U_trunc * diagm(S_trunc), Dl, d, D_keep)
        Ar = reshape(Vh_trunc, D_keep, d, Dr)
    end

    return Al, Ar, λ, e_trunc
end

function l2r_DMRG_2site!(mps::MPS, mpo::MPO, right_envs::Vector{Array{T,3}}, left_envs::Vector{Array{T,3}}, λs::Vector{Float64}, trunc_errors::Vector{Float64}) where {T}
    """Left-to-right two-site DMRG sweep from site 1 to site N-1
    Modifies MPS in-place and reuses preallocated left_envs and λs arrays.
    """
    N = mps.N

    for n in 1:N-1
        left_env = left_envs[n]
        right_env = right_envs[n]
        O1 = mpo.O[n]
        O2 = mpo.O[n+1]
        D = size(mps.A[n], 3)

        # update site n
        Al, Ar, λ, e_trunc = DMRG_1step_2site(left_env, O1, O2, right_env, D, "l2r")

        # store
        mps.A[n] = Al
        mps.A[n+1] = Ar
        λs[n] = λ
        trunc_errors[n] = e_trunc

        # Update left environment
        if n <= N - 2
            @tensor left_env_new[o, p, l] := left_env[u, y, j] * conj(Al)[u, i, o] * O1[y, p, i, k] * Al[j, k, l]
            left_envs[n+1] = left_env_new
        end
    end

    return nothing
end

function r2l_DMRG_2site!(mps::MPS, mpo::MPO,
    left_envs::Vector{Array{T,3}},
    right_envs::Vector{Array{T,3}},
    λs::Vector{Float64},
    trunc_errors::Vector{Float64}) where {T}
    """Right-to-left DMRG sweep from site N to site 2
    Modifies MPS in-place and reuses preallocated right_envs and λs arrays.
    """
    N = mps.N

    for n in N:-1:2
        left_env = left_envs[n-1]
        right_env = right_envs[n-1]
        O1 = mpo.O[n-1]
        O2 = mpo.O[n]
        D = size(mps.A[n-1], 3)

        # update site n
        Al, Ar, λ, e_trunc = DMRG_1step_2site(left_env, O1, O2, right_env, D, "r2l")

        # store
        mps.A[n-1] = Al
        mps.A[n] = Ar
        λs[N+1-n] = λ
        trunc_errors[N+1-n] = e_trunc

        # Update right environment
        if n >= 3
            @tensor right_env_new[u, y, j] := right_env[o, p, l] * conj(Ar)[u, i, o] * O2[y, p, i, k] * Ar[j, k, l]
            right_envs[n-2] = right_env_new
        end
    end

    return nothing
end


function DMRG_loop_2site!(mps::MPS{T}, mpo::MPO, times::Int, threshold::Float64) where {T}
    """Main DMRG loop 
    - Preallocates all arrays
    - Reuses environment tensors
    - Modifies MPS in-place
    """
    N = mps.N

    # Preallocate environments (reused across sweeps)
    left_envs = Vector{Array{T,3}}(undef, N - 1)
    left_envs[1] = ones(1, 1, 1)
    right_envs = l2r_DMRG_prep_2site(mps, mpo)

    # Preallocate energy array with maximum possible size
    max_size = times * 2 * (N - 1)
    λs_all = Vector{Float64}(undef, max_size)
    trunc_errs_all = Vector{Float64}(undef, max_size)

    # Preallocate temporary arrays for each sweep
    # λs_lr = Vector{Float64}(undef, N - 1)
    # trunc_errors_lr = Vector{Float64}(undef, N - 1)
    # λs_rl = Vector{Float64}(undef, N - 1)
    # trunc_errors_rl = Vector{Float64}(undef, N - 1)
    λs = Vector{Float64}(undef, N - 1)
    trunc_errs = Vector{Float64}(undef, N - 1)


    idx = 0 # index of last stored energy
    i = 0 # index of loops
    e = 100 # initial error

    while i < times && e > threshold
        # Left-to-right sweep
        l2r_DMRG_2site!(mps, mpo, right_envs, left_envs, λs, trunc_errs)
        λs_all[idx+1:idx+N-1] .= λs[1:N-1]
        trunc_errs_all[idx+1:idx+N-1] .= trunc_errs[1:N-1]
        idx += N - 1

        # Right-to-left sweep
        r2l_DMRG_2site!(mps, mpo, left_envs, right_envs, λs, trunc_errs)
        λs_all[idx+1:idx+N-1] .= λs[1:N-1]
        trunc_errs_all[idx+1:idx+N-1] .= trunc_errs[1:N-1]
        idx += N - 1

        # Check convergence
        if idx >= 2
            e = λs_all[idx-1] - λs_all[idx]
        end

        i += 1
    end

    return λs_all[1:idx], trunc_errs_all[1:idx]
end