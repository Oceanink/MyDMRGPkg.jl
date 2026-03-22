export DMRG_loop_2site!

function l2r_DMRG_prep_2site(mps::MPS{T}, mpo::MPO) where {T}
    """Prepare right environments for the first left-to-right sweep"""
    N = mps.N
    right_envs = Vector{Array{T,3}}(undef, N - 1)
    right_envs[N-1] = ones(T, 1, 1, 1)

    for n in N:-1:3
        On = mpo.O[n]
        An = mps.A[n]
        right_env = right_envs[n-1]
        @tensor right_env[u, y, j] := right_env[o, p, l] * conj(An)[u, i, o] * On[y, p, i, k] * An[j, k, l]
        right_envs[n-2] = right_env
    end
    return right_envs
end


function DMRG_1step_2site(left_env::Array{T,3}, O1::Array{T2,4}, O2::Array{T2,4}, right_env::Array{T,3}, D::Int, direction::String; x0=nothing) where {T,T2}
    """Single two-site DMRG optimization step using iterative eigensolver

    Optional x0: initial guess vector (reshaped to size of two-site tensor)
    """
    @assert direction == "l2r" || direction == "r2l"
    Dl = size(left_env, 3)
    d = size(O1, 4)
    Dr = size(right_env, 3)

    function apply_H_eff(x_vec::AbstractVector, left_env, O1, O2, right_env)
        dim_v = size(left_env, 3)
        dim_b = size(O1, 4)
        dim_n = size(O2, 4)
        dim_m = size(right_env, 3)

        # 获取输出维度，用于最后展平
        dim_u = size(left_env, 1)
        dim_i = size(O1, 3)
        dim_o = size(O2, 3)
        dim_p = size(right_env, 1)

        x_tensor = reshape(x_vec, dim_v, dim_b, dim_n, dim_m)

        @tensor begin
            tmp1[u, j, b, n, m] := left_env[u, j, v] * x_tensor[v, b, n, m]
            tmp2[u, k, i, n, m] := tmp1[u, j, b, n, m] * O1[j, k, i, b]
            tmp3[u, l, i, o, m] := tmp2[u, k, i, n, m] * O2[k, l, o, n]
            y_tensor[u, i, o, p] := tmp3[u, l, i, o, m] * right_env[p, l, m]
        end

        return reshape(y_tensor, dim_u * dim_i * dim_o * dim_p)
    end
    H_action = x -> apply_H_eff(x, left_env, O1, O2, right_env)

    # Find only the smallest eigenvalue using iterative method
    # :SR means "smallest real" eigenvalue
    # H_eff_mat = reshape(H_eff, dim1, dim2)
    if x0 !== nothing
        λs, vecs, _ = eigsolve(H_action, x0, 1, :SR, ishermitian=true)
    else
        x0 = rand(T, Dl * d * d * Dr)
        λs, vecs, _ = eigsolve(H_action, x0, 1, :SR, ishermitian=true)
    end
    λ = real(λs[1])

    B_mat = reshape(vecs[1], Dl * d, d * Dr)
    U, S, V = svd(B_mat)
    D_keep = min(Dl * d, d * Dr, D)
    @views begin
        e_trunc = D_keep < length(S) ? sum(abs2, S[D_keep+1:end]) : zero(real(eltype(S)))
        S_trunc = S[1:D_keep]
        U_trunc = U[:, 1:D_keep]
        Vh_trunc = V'[1:D_keep, :]
    end

    if direction == "l2r"
        Al = reshape(Matrix(U_trunc), Dl, d, D_keep)
        Ar_mat = Matrix(Vh_trunc)
        lmul!(Diagonal(S_trunc), Ar_mat)
        Ar = reshape(Ar_mat, D_keep, d, Dr)
    else
        Al_mat = Matrix(U_trunc)
        rmul!(Al_mat, Diagonal(S_trunc))
        Al = reshape(Al_mat, Dl, d, D_keep)
        Ar = reshape(Matrix(Vh_trunc), D_keep, d, Dr)
    end

    return Al, Ar, λ, e_trunc
end

function l2r_DMRG_2site!(mps::MPS, mpo::MPO, right_envs::Vector{Array{T,3}}, left_envs::Vector{Array{T,3}}, λs::Vector{Float64}, trunc_errors::Vector{Float64}; show_progress::Bool=true) where {T}
    """Left-to-right two-site DMRG sweep from site 1 to site N-1
    Modifies MPS in-place and reuses preallocated left_envs and λs arrays.
    Uses current two-site tensor as initial guess for next pair.
    Stores λ and trunc_errors in the provided arrays.
    """
    N = mps.N

    sweep_range = 1:N-1
    iter = show_progress ? ProgressBar(sweep_range) : sweep_range
    for n in iter
        left_env = left_envs[n]
        right_env = right_envs[n]
        O1 = mpo.O[n]
        O2 = mpo.O[n+1]
        D = size(mps.A[n], 3)

        # Compute expected dimensions for two-site tensor
        Dl = size(left_env, 3)  # left bond dimension
        Dr = size(right_env, 3) # right bond dimension
        d = size(O1, 4)         # physical dimension (should be same as size(O2,4))

        # Prepare initial guess from current MPS tensors if dimensions match
        x0 = nothing
        if n >= 2  # For n=1, no previous update; for n>=2, site n may have been updated by previous step
            # Check if current tensors have expected outer dimensions
            Dl_curr, d1, Dmid = size(mps.A[n])
            Dmid2, d2, Dr_curr = size(mps.A[n+1])
            if Dl_curr == Dl && Dr_curr == Dr && d1 == d && d2 == d && Dmid == Dmid2
                # Contract current two-site tensor
                @tensor B_curr[v, b, n_idx, m] := mps.A[n][v, b, k] * mps.A[n+1][k, n_idx, m]
                x0 = vec(B_curr)
            end
        end

        # update site n
        Al, Ar, λ, e_trunc = DMRG_1step_2site(left_env, O1, O2, right_env, D, "l2r"; x0=x0)
        show_progress && set_description(iter, string(@sprintf("λ: %.2f", λ)))

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

function l2r_DMRG_2site!(mps::MPS, mpo::MPO, right_envs::Vector{Array{T,3}}, left_envs::Vector{Array{T,3}}; show_progress::Bool=true) where {T}
    """Left-to-right two-site DMRG sweep from site 1 to site N-1
    Modifies MPS in-place and reuses preallocated left_envs.
    Uses current two-site tensor as initial guess for next pair.
    Returns the final energy (λ at site N-1).
    """
    N = mps.N
    λ_final = 0.0

    sweep_range = 1:N-1
    iter = show_progress ? ProgressBar(sweep_range) : sweep_range
    for n in iter
        left_env = left_envs[n]
        right_env = right_envs[n]
        O1 = mpo.O[n]
        O2 = mpo.O[n+1]
        D = size(mps.A[n], 3)

        # Compute expected dimensions for two-site tensor
        Dl = size(left_env, 3)  # left bond dimension
        Dr = size(right_env, 3) # right bond dimension
        d = size(O1, 4)         # physical dimension (should be same as size(O2,4))

        # Prepare initial guess from current MPS tensors if dimensions match
        x0 = nothing
        if n >= 2  # For n=1, no previous update; for n>=2, site n may have been updated by previous step
            # Check if current tensors have expected outer dimensions
            Dl_curr, d1, Dmid = size(mps.A[n])
            Dmid2, d2, Dr_curr = size(mps.A[n+1])
            if Dl_curr == Dl && Dr_curr == Dr && d1 == d && d2 == d && Dmid == Dmid2
                # Contract current two-site tensor
                @tensor B_curr[v, b, n_idx, m] := mps.A[n][v, b, k] * mps.A[n+1][k, n_idx, m]
                x0 = vec(B_curr)
            end
        end

        # update site n
        Al, Ar, λ, e_trunc = DMRG_1step_2site(left_env, O1, O2, right_env, D, "l2r"; x0=x0)
        show_progress && set_description(iter, string(@sprintf("λ: %.2f", λ)))

        # store
        mps.A[n] = Al
        mps.A[n+1] = Ar
        λ_final = λ

        # Update left environment
        if n <= N - 2
            @tensor left_env_new[o, p, l] := left_env[u, y, j] * conj(Al)[u, i, o] * O1[y, p, i, k] * Al[j, k, l]
            left_envs[n+1] = left_env_new
        end
    end

    return λ_final
end

function r2l_DMRG_2site!(mps::MPS, mpo::MPO,
    left_envs::Vector{Array{T,3}},
    right_envs::Vector{Array{T,3}},
    λs::Vector{Float64},
    trunc_errors::Vector{Float64};
    show_progress::Bool=true) where {T}
    """Right-to-left DMRG sweep from site N to site 2
    Modifies MPS in-place and reuses preallocated right_envs and λs arrays.
    Uses current two-site tensor as initial guess for next pair.
    Stores λ and trunc_errors in the provided arrays.
    """
    N = mps.N

    sweep_range = N:-1:2
    iter = show_progress ? ProgressBar(sweep_range) : sweep_range
    for n in iter
        left_env = left_envs[n-1]
        right_env = right_envs[n-1]
        O1 = mpo.O[n-1]
        O2 = mpo.O[n]
        D = size(mps.A[n-1], 3)

        # Compute expected dimensions for two-site tensor
        Dl = size(left_env, 3)  # left bond dimension
        Dr = size(right_env, 3) # right bond dimension
        d = size(O1, 4)         # physical dimension (should be same as size(O2,4))

        # Prepare initial guess from current MPS tensors if dimensions match
        x0 = nothing
        if n <= N - 1  # For n=N, no previous update; for n<=N-1, site n-1 may have been updated by previous step
            # Check if current tensors have expected outer dimensions
            Dl_curr, d1, Dmid = size(mps.A[n-1])
            Dmid2, d2, Dr_curr = size(mps.A[n])
            if Dl_curr == Dl && Dr_curr == Dr && d1 == d && d2 == d && Dmid == Dmid2
                # Contract current two-site tensor
                @tensor B_curr[v, b, n_idx, m] := mps.A[n-1][v, b, k] * mps.A[n][k, n_idx, m]
                x0 = vec(B_curr)
            end
        end

        # update site n
        Al, Ar, λ, e_trunc = DMRG_1step_2site(left_env, O1, O2, right_env, D, "r2l"; x0=x0)
        show_progress && set_description(iter, string(@sprintf("λ: %.2f", λ)))

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

function r2l_DMRG_2site!(mps::MPS, mpo::MPO,
    left_envs::Vector{Array{T,3}},
    right_envs::Vector{Array{T,3}};
    show_progress::Bool=true) where {T}
    """Right-to-left DMRG sweep from site N to site 2
    Modifies MPS in-place and reuses preallocated right_envs.
    Uses current two-site tensor as initial guess for next pair.
    Returns (final_energy, final_trunc_error).
    """
    N = mps.N
    λ_final = 0.0
    trunc_err_final = 0.0

    sweep_range = N:-1:2
    iter = show_progress ? ProgressBar(sweep_range) : sweep_range
    for n in iter
        left_env = left_envs[n-1]
        right_env = right_envs[n-1]
        O1 = mpo.O[n-1]
        O2 = mpo.O[n]
        D = size(mps.A[n-1], 3)

        # Compute expected dimensions for two-site tensor
        Dl = size(left_env, 3)  # left bond dimension
        Dr = size(right_env, 3) # right bond dimension
        d = size(O1, 4)         # physical dimension (should be same as size(O2,4))

        # Prepare initial guess from current MPS tensors if dimensions match
        x0 = nothing
        if n <= N - 1  # For n=N, no previous update; for n<=N-1, site n-1 may have been updated by previous step
            # Check if current tensors have expected outer dimensions
            Dl_curr, d1, Dmid = size(mps.A[n-1])
            Dmid2, d2, Dr_curr = size(mps.A[n])
            if Dl_curr == Dl && Dr_curr == Dr && d1 == d && d2 == d && Dmid == Dmid2
                # Contract current two-site tensor
                @tensor B_curr[v, b, n_idx, m] := mps.A[n-1][v, b, k] * mps.A[n][k, n_idx, m]
                x0 = vec(B_curr)
            end
        end

        # update site n
        Al, Ar, λ, e_trunc = DMRG_1step_2site(left_env, O1, O2, right_env, D, "r2l"; x0=x0)
        show_progress && set_description(iter, string(@sprintf("λ: %.2f", λ)))

        # store
        mps.A[n-1] = Al
        mps.A[n] = Ar
        λ_final = λ
        trunc_err_final = e_trunc

        # Update right environment
        if n >= 3
            @tensor right_env_new[u, y, j] := right_env[o, p, l] * conj(Ar)[u, i, o] * O2[y, p, i, k] * Ar[j, k, l]
            right_envs[n-2] = right_env_new
        end
    end

    return λ_final, trunc_err_final
end


function DMRG_loop_2site!(mps::MPS{T}, mpo::MPO, times::Int, threshold::Real;
    store_all::Bool=true, show_progress::Bool=true) where {T}
    """Main DMRG loop 
    - Preallocates all arrays
    - Reuses environment tensors
    - Modifies MPS in-place

    Keyword arguments:
    - store_all: if true (default), store all energies and truncation errors; if false, only return the final values
    - show_progress: if true (default), display progress bar during sweeps
    """
    @assert is_right_canonical(mps)
    N = mps.N

    # Preallocate environments (reused across sweeps)
    left_envs = Vector{Array{T,3}}(undef, N - 1)
    left_envs[1] = ones(T, 1, 1, 1)
    right_envs = l2r_DMRG_prep_2site(mps, mpo)

    # Preallocate energy array with maximum possible size if storing all
    if store_all
        max_size = times * 2 * (N - 1)
        λs_all = Vector{Float64}(undef, max_size)
        trunc_errs_all = Vector{Float64}(undef, max_size)
    else
        λs_all = Float64[]
        trunc_errs_all = Float64[]
    end

    # Variables to track final values (needed when store_all=false)
    λ_lr = 0.0
    λ_rl = 0.0
    final_trunc_err = 0.0

    if store_all
        λs = Vector{Float64}(undef, N - 1)
        trunc_errs = Vector{Float64}(undef, N - 1)
    else
        # Dummy arrays when not storing - inner functions won't use them
        λs = Float64[]
        trunc_errs = Float64[]
    end

    idx = 0 # index of last stored energy
    i = 0 # index of loops
    e = Inf # initial error

    while i < times && e > threshold
        # Left-to-right sweep
        show_progress && println("DMRG loop $(i+1), left-to-right sweep...")
        if store_all
            l2r_DMRG_2site!(mps, mpo, right_envs, left_envs, λs, trunc_errs; show_progress=show_progress)
            copyto!(λs_all, idx + 1, λs, 1, N - 1)
            copyto!(trunc_errs_all, idx + 1, trunc_errs, 1, N - 1)
            idx += N - 1
            λ_lr = λs[N-1]
        else
            λ_lr = l2r_DMRG_2site!(mps, mpo, right_envs, left_envs; show_progress=show_progress)
        end

        # Right-to-left sweep
        show_progress && println("DMRG loop $(i+1), right-to-left sweep...")
        if store_all
            r2l_DMRG_2site!(mps, mpo, left_envs, right_envs, λs, trunc_errs; show_progress=show_progress)
            copyto!(λs_all, idx + 1, λs, 1, N - 1)
            copyto!(trunc_errs_all, idx + 1, trunc_errs, 1, N - 1)
            idx += N - 1
            λ_rl = λs[N-1]
        else
            λ_rl, final_trunc_err = r2l_DMRG_2site!(mps, mpo, left_envs, right_envs; show_progress=show_progress)
        end

        # Check convergence
        e = λ_lr - λ_rl

        i += 1
    end

    if store_all
        resize!(λs_all, idx)
        resize!(trunc_errs_all, idx)
        return λs_all, trunc_errs_all
    else
        # Return only final values
        return [λ_rl], [final_trunc_err]
    end
end
