export DMRG_loop!, DMRG_converge!


function _l2r_QR(A::Array{T,3}) where {T}
    Dl, d, Dr = size(A)
    A_mat = reshape(A, Dl * d, Dr)

    Q, R = qr(A_mat)
    Q = Matrix(Q)

    return reshape(Q, Dl, d, size(Q, 2)), R
end

function _r2l_LQ(A::Array{T,3}) where {T}
    Dl, d, Dr = size(A)
    A_mat = reshape(A, Dl, d * Dr)

    L, Q = lq(A_mat)
    Q = Matrix(Q)

    return L, reshape(Q, size(Q, 1), d, Dr)
end

# DMRG functions

function DMRG_1step(left_env::Array{T,3}, O::Array{T2,4}, right_env::Array{T,3}; x0=nothing) where {T,T2}
    """Single DMRG optimization step using iterative eigensolver

    Optional x0: initial guess vector (reshaped to size of site tensor)
    """
    @tensor H_eff[u, i, o, j, k, l] := left_env[u, y, j] * O[y, p, i, k] * right_env[o, p, l]

    H_eff_size = size(H_eff)
    dim1 = prod(H_eff_size[1:3])
    dim2 = prod(H_eff_size[4:6])


    # Find only the smallest eigenvalue using iterative method
    # :SR means "smallest real" eigenvalue
    H_eff_mat = reshape(H_eff, dim1, dim2)
    if x0 !== nothing
        λs, vecs, _ = eigsolve(H_eff_mat, x0, 1, :SR)
    else
        λs, vecs, _ = eigsolve(H_eff_mat, 1, :SR)
    end

    λ = real(λs[1])
    An_new = reshape(vecs[1], H_eff_size[4:6])

    return An_new, λ
end

function l2r_DMRG!(mps::MPS, mpo::MPO,
    right_envs::Vector{Array{T,3}},
    left_envs::Vector{Array{T,3}},
    λs::Vector{Float64}) where {T}
    """Left-to-right DMRG sweep from site 1 to site N-1
    Modifies MPS in-place and reuses preallocated left_envs and λs arrays.
    After QR on site n, applies R to site n+1 and uses it as initial guess for next site.
    """
    N = mps.N
    # left_envs[1] = ones(1, 1, 1)

    # Apply R from previous site to current site before optimization
    prev_R = nothing  # R factor from previous site (n-1)

    for n in 1:N-1
        left_env = left_envs[n]
        right_env = right_envs[n]
        On = mpo.O[n]

        # If we have R from previous site, apply it to current site n
        if prev_R !== nothing && n >= 2
            # Apply R to site n from left side
            Dl, d, Dr = size(mps.A[n])
            # prev_R size should be (Dl, Dl) square
            if size(prev_R, 1) == Dl && size(prev_R, 2) == Dl
                A_mat = reshape(mps.A[n], Dl, d * Dr)
                A_mat = prev_R * A_mat
                mps.A[n] = reshape(A_mat, Dl, d, Dr)
            end
        end

        # Prepare initial guess: use current site tensor if dimensions match
        x0 = nothing
        if n >= 2  # For sites 2 and onwards, we have previous R applied
            # Check if current tensor dimensions match the expected dimensions of An_new
            j_dim = size(left_env, 3)  # left bond dimension from left_env
            k_dim = size(On, 4)        # physical dimension d
            l_dim = size(right_env, 3) # right bond dimension from right_env
            if size(mps.A[n]) == (j_dim, k_dim, l_dim)
                x0 = vec(mps.A[n])
            end
        end

        # update site n
        An_new, λ = DMRG_1step(left_env, On, right_env; x0=x0)

        # QR decomposition
        Q, R = _l2r_QR(An_new)
        mps.A[n] = Q
        λs[n] = λ

        # Store R to apply to next site (n+1) in next iteration
        prev_R = R

        # Update left environment
        @tensor left_env_new[o, p, l] := left_env[u, y, j] * conj(Q)[u, i, o] * On[y, p, i, k] * Q[j, k, l]
        left_envs[n+1] = left_env_new
    end

    return nothing
end

function r2l_DMRG!(mps::MPS, mpo::MPO,
    left_envs::Vector{Array{T,3}},
    right_envs::Vector{Array{T,3}},
    λs::Vector{Float64}) where {T}
    """Right-to-left DMRG sweep from site N to site 2
    Modifies MPS in-place and reuses preallocated right_envs and λs arrays.
    After LQ on site n, applies L to site n-1 and uses it as initial guess for next site.
    """
    N = mps.N
    # right_envs[N] = ones(1, 1, 1)

    # Apply L from previous site (n+1) to current site n before optimization
    prev_L = nothing  # L factor from previous site (n+1)

    for n in N:-1:2
        left_env = left_envs[n]
        right_env = right_envs[n]
        On = mpo.O[n]

        # If we have L from previous site (n+1), apply it to current site n from right side
        if prev_L !== nothing && n <= N - 1
            # Apply L to site n from right side
            Dl, d, Dr = size(mps.A[n])
            # prev_L size should be (Dr, Dr) square (since L from LQ of site n+1)
            if size(prev_L, 1) == Dr && size(prev_L, 2) == Dr
                A_mat = reshape(mps.A[n], Dl * d, Dr)
                A_mat = A_mat * prev_L'
                mps.A[n] = reshape(A_mat, Dl, d, Dr)
            end
        end

        # Prepare initial guess: use current site tensor if dimensions match
        x0 = nothing
        if n <= N - 1  # For sites N-1 and less, we may have previous L applied
            # Check if current tensor dimensions match the expected dimensions of An_new
            j_dim = size(left_env, 3)  # left bond dimension from left_env
            k_dim = size(On, 4)        # physical dimension d
            l_dim = size(right_env, 3) # right bond dimension from right_env
            if size(mps.A[n]) == (j_dim, k_dim, l_dim)
                x0 = vec(mps.A[n])
            end
        end

        # update site n
        An_new, λ = DMRG_1step(left_env, On, right_env; x0=x0)

        # LQ decomposition
        L, Q = _r2l_LQ(An_new)
        mps.A[n] = Q
        λs[N+1-n] = λ

        # Store L to apply to next site (n-1) in next iteration
        prev_L = L

        # Update right environment
        @tensor right_env_new[u, y, j] := right_env[o, p, l] * conj(Q)[u, i, o] * On[y, p, i, k] * Q[j, k, l]
        right_envs[n-1] = right_env_new
    end

    return prev_L
end

function l2r_DMRG_prep(mps::MPS{T}, mpo::MPO) where {T}
    """Prepare right environments for the first left-to-right sweep"""
    N = mps.N
    right_envs = Vector{Array{T,3}}(undef, N)
    right_envs[N] = ones(1, 1, 1)

    for n in N:-1:2
        On = mpo.O[n]
        An = mps.A[n]
        right_env = right_envs[n]
        @tensor right_env[u, y, j] := right_env[o, p, l] * conj(An)[u, i, o] * On[y, p, i, k] * An[j, k, l]
        right_envs[n-1] = right_env
    end
    return right_envs
end

function DMRG_loop!(mps::MPS{T}, mpo::MPO, times::Int, threshold::Float64) where {T}
    """Main DMRG loop 
    - Preallocates all arrays
    - Reuses environment tensors
    - Modifies MPS in-place
    """
    N = mps.N

    # Preallocate environments (reused across sweeps)
    left_envs = Vector{Array{T,3}}(undef, N)
    left_envs[1] = ones(1, 1, 1)
    right_envs = l2r_DMRG_prep(mps, mpo)

    # Preallocate energy array with maximum possible size
    max_size = times * 2 * (N - 1)
    λs_all = Vector{Float64}(undef, max_size)

    # Preallocate temporary arrays for each sweep
    λs = Vector{Float64}(undef, N - 1)
    # λs_rl = Vector{Float64}(undef, N - 1)

    idx = 0 # index of last stored energy
    i = 0 # index of loops
    e = 100 # initial error

    L = nothing
    while i < times && e > threshold
        # Left-to-right sweep
        l2r_DMRG!(mps, mpo, right_envs, left_envs, λs)
        λs_all[idx+1:idx+N-1] .= λs[1:N-1]
        λ_lr = λs[N-1]
        idx += N - 1

        # Right-to-left sweep
        L = r2l_DMRG!(mps, mpo, left_envs, right_envs, λs)
        λs_all[idx+1:idx+N-1] .= λs[1:N-1]
        λ_rl = λs[N-1]
        idx += N - 1

        # Check convergence
        e = λ_lr - λ_rl

        i += 1
    end

    A1 = nothing
    @tensor A1[i, j, l] := mps.A[1][i, j, k] * L[k, l]
    mps.A[1] = A1 ./ norm(A1)

    return λs_all[1:idx]
end

"""
    DMRG_converge!(mps::MPS{T}, mpo::MPO, max_sites_updated::Int, threshold::Float64) where {T}

Perform DMRG sweeps until convergence, returning a tuple (final_energy, sites_updated).
The algorithm stops when either:
1. The energy change after a site update is below `threshold` (convergence reached), or
2. Total number of site updates reaches `max_sites_updated`.
Modifies the MPS in-place.

`sites_updated` is the number of site updates performed before convergence,
or -1 if convergence was not reached within `max_sites_updated`.
"""
function DMRG_converge!(mps::MPS{T}, mpo::MPO, max_sites_updated::Int, threshold::Float64) where {T}
    N = mps.N

    # Preallocate environments (reused across sweeps)
    left_envs = Vector{Array{T,3}}(undef, N)
    right_envs = l2r_DMRG_prep(mps, mpo)

    # Tracking variables
    λ_prev = Inf  # previous energy value (from last site update)
    sites_updated = 0
    converged = false
    final_energy = NaN

    while (sites_updated < max_sites_updated) && !converged
        # Left-to-right sweep (sites 1 to N-1)
        left_envs[1] = ones(1, 1, 1)
        prev_R = nothing  # R factor from previous site
        for n in 1:N-1
            # Check if we've reached max sites
            if sites_updated >= max_sites_updated
                break
            end

            left_env = left_envs[n]
            right_env = right_envs[n]
            On = mpo.O[n]

            # If we have R from previous site, apply it to current site n
            if prev_R !== nothing && n >= 2
                # Apply R to site n from left side
                Dl, d, Dr = size(mps.A[n])
                # prev_R size should be (Dl, Dl) square
                if size(prev_R, 1) == Dl && size(prev_R, 2) == Dl
                    A_mat = reshape(mps.A[n], Dl, d * Dr)
                    A_mat = prev_R * A_mat
                    mps.A[n] = reshape(A_mat, Dl, d, Dr)
                end
            end

            # Prepare initial guess: use current site tensor if dimensions match
            x0 = nothing
            if n >= 2  # For sites 2 and onwards, we have previous R applied
                # Check if current tensor dimensions match the expected dimensions of An_new
                j_dim = size(left_env, 3)  # left bond dimension from left_env
                k_dim = size(On, 4)        # physical dimension d
                l_dim = size(right_env, 3) # right bond dimension from right_env
                if size(mps.A[n]) == (j_dim, k_dim, l_dim)
                    x0 = vec(mps.A[n])
                end
            end

            # update site n
            An_new, λ = DMRG_1step(left_env, On, right_env; x0=x0)

            # Check convergence with previous energy
            if abs(λ - λ_prev) < threshold
                converged = true
                final_energy = λ
                # Update the current site before breaking
                Q, R = _l2r_QR(An_new)
                sites_updated += 1
                mps.A[n] = Q
                # Update left environment for consistency (optional)
                @tensor left_env_new[o, p, l] := left_env[u, y, j] * conj(Q)[u, i, o] * On[y, p, i, k] * Q[j, k, l]
                left_envs[n+1] = left_env_new
                break
            end
            λ_prev = λ

            # normalize and store
            Q, R = _l2r_QR(An_new)
            sites_updated += 1
            mps.A[n] = Q

            # Update left environment
            @tensor left_env_new[o, p, l] := left_env[u, y, j] * conj(Q)[u, i, o] * On[y, p, i, k] * Q[j, k, l]
            left_envs[n+1] = left_env_new

            # Store R to apply to next site (n+1) in next iteration
            prev_R = R
        end

        if converged || (sites_updated >= max_sites_updated)
            break
        end

        # Right-to-left sweep (sites N to 2)
        right_envs[N] = ones(1, 1, 1)
        prev_L = nothing  # L factor from previous site (n+1)
        for n in N:-1:2
            # Check if we've reached max sites
            if sites_updated >= max_sites_updated
                break
            end

            left_env = left_envs[n]
            right_env = right_envs[n]
            On = mpo.O[n]

            # If we have L from previous site (n+1), apply it to current site n from right side
            if prev_L !== nothing && n <= N - 1
                # Apply L to site n from right side
                Dl, d, Dr = size(mps.A[n])
                # prev_L size should be (Dr, Dr) square (since L from LQ of site n+1)
                if size(prev_L, 1) == Dr && size(prev_L, 2) == Dr
                    A_mat = reshape(mps.A[n], Dl * d, Dr)
                    A_mat = A_mat * prev_L'
                    mps.A[n] = reshape(A_mat, Dl, d, Dr)
                end
            end

            # Prepare initial guess: use current site tensor if dimensions match
            x0 = nothing
            if n <= N - 1  # For sites N-1 and less, we may have previous L applied
                # Check if current tensor dimensions match the expected dimensions of An_new
                j_dim = size(left_env, 3)  # left bond dimension from left_env
                k_dim = size(On, 4)        # physical dimension d
                l_dim = size(right_env, 3) # right bond dimension from right_env
                if size(mps.A[n]) == (j_dim, k_dim, l_dim)
                    x0 = vec(mps.A[n])
                end
            end

            # update site n
            An_new, λ = DMRG_1step(left_env, On, right_env; x0=x0)

            # Check convergence with previous energy
            if abs(λ - λ_prev) < threshold
                converged = true
                final_energy = λ
                # Update the current site before breaking
                L, Q = _r2l_LQ(An_new)
                sites_updated += 1
                mps.A[n] = Q
                # Update right environment for consistency (optional)
                @tensor right_env_new[u, y, j] := right_env[o, p, l] * conj(Q)[u, i, o] * On[y, p, i, k] * Q[j, k, l]
                right_envs[n-1] = right_env_new
                break
            end
            λ_prev = λ

            # normalize and store
            L, Q = _r2l_LQ(An_new)
            sites_updated += 1
            mps.A[n] = Q

            # Update right environment
            @tensor right_env_new[u, y, j] := right_env[o, p, l] * conj(Q)[u, i, o] * On[y, p, i, k] * Q[j, k, l]
            right_envs[n-1] = right_env_new

            # Store L to apply to next site (n-1) in next iteration
            prev_L = L
        end
    end

    # Set return values
    if !converged
        final_energy = λ_prev
        sites_updated = -1
    end

    return (final_energy, sites_updated)
end