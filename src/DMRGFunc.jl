export DMRG_loop!, DMRG_converge!


function _l2r_QR(A::Array{T,3})::Array{T,3} where {T}
    Dl, d, Dr = size(A)
    A_mat = reshape(A, Dl * d, Dr)

    Q, _ = qr(A_mat)
    Q = Matrix(Q)

    return reshape(Q, Dl, d, size(Q, 2))
end

function _r2l_LQ(A::Array{T,3})::Array{T,3} where {T}
    Dl, d, Dr = size(A)
    A_mat = reshape(A, Dl, d * Dr)

    _, Q = lq(A_mat)
    Q = Matrix(Q)

    return reshape(Q, size(Q, 1), d, Dr)
end

# DMRG functions

function DMRG_1step(left_env::Array{T,3}, O::Array{T2,4}, right_env::Array{T,3}) where {T,T2}
    """Single DMRG optimization step using iterative eigensolver

    """
    @tensor H_eff[u, i, o, j, k, l] := left_env[u, y, j] * O[y, p, i, k] * right_env[o, p, l]

    H_eff_size = size(H_eff)
    dim1 = prod(H_eff_size[1:3])
    dim2 = prod(H_eff_size[4:6])


    # Find only the smallest eigenvalue using iterative method
    # :SR means "smallest real" eigenvalue
    H_eff_mat = reshape(H_eff, dim1, dim2)
    λs, vecs, _ = eigsolve(H_eff_mat, 1, :SR)

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
    """
    N = mps.N
    # left_envs[1] = ones(1, 1, 1)

    for n in 1:N-1
        left_env = left_envs[n]
        right_env = right_envs[n]
        On = mpo.O[n]

        # update site n
        An_new, λ = DMRG_1step(left_env, On, right_env)

        # normalize and store
        Q = _l2r_QR(An_new)
        mps.A[n] = Q
        λs[n] = λ

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
    """
    N = mps.N
    # right_envs[N] = ones(1, 1, 1)

    for n in N:-1:2
        left_env = left_envs[n]
        right_env = right_envs[n]
        On = mpo.O[n]

        # update site n
        An_new, λ = DMRG_1step(left_env, On, right_env)

        # normalize and store
        Q = _r2l_LQ(An_new)
        mps.A[n] = Q
        λs[N+1-n] = λ

        # Update right environment
        @tensor right_env_new[u, y, j] := right_env[o, p, l] * conj(Q)[u, i, o] * On[y, p, i, k] * Q[j, k, l]
        right_envs[n-1] = right_env_new
    end

    return nothing
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

    while i < times && e > threshold
        # Left-to-right sweep
        l2r_DMRG!(mps, mpo, right_envs, left_envs, λs)
        λs_all[idx+1:idx+N-1] .= λs[1:N-1]
        idx += N - 1

        # Right-to-left sweep
        r2l_DMRG!(mps, mpo, left_envs, right_envs, λs)
        λs_all[idx+1:idx+N-1] .= λs[1:N-1]
        idx += N - 1

        # Check convergence
        if idx >= 2
            e = λs_all[idx-1] - λs_all[idx]
        end

        i += 1
    end

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
        for n in 1:N-1
            # Check if we've reached max sites
            if sites_updated >= max_sites_updated
                break
            end

            left_env = left_envs[n]
            right_env = right_envs[n]
            On = mpo.O[n]

            # update site n
            An_new, λ = DMRG_1step(left_env, On, right_env)

            # Check convergence with previous energy
            if abs(λ - λ_prev) < threshold
                converged = true
                final_energy = λ
                # Update the current site before breaking
                Q = _l2r_QR(An_new)
                sites_updated += 1
                mps.A[n] = Q
                # Update left environment for consistency (optional)
                @tensor left_env_new[o, p, l] := left_env[u, y, j] * conj(Q)[u, i, o] * On[y, p, i, k] * Q[j, k, l]
                left_envs[n+1] = left_env_new
                break
            end
            λ_prev = λ

            # normalize and store
            Q = _l2r_QR(An_new)
            sites_updated += 1
            mps.A[n] = Q

            # Update left environment
            @tensor left_env_new[o, p, l] := left_env[u, y, j] * conj(Q)[u, i, o] * On[y, p, i, k] * Q[j, k, l]
            left_envs[n+1] = left_env_new
        end

        if converged || (sites_updated >= max_sites_updated)
            break
        end

        # Right-to-left sweep (sites N to 2)
        right_envs[N] = ones(1, 1, 1)
        for n in N:-1:2
            # Check if we've reached max sites
            if sites_updated >= max_sites_updated
                break
            end

            left_env = left_envs[n]
            right_env = right_envs[n]
            On = mpo.O[n]

            # update site n
            An_new, λ = DMRG_1step(left_env, On, right_env)

            # Check convergence with previous energy
            if abs(λ - λ_prev) < threshold
                converged = true
                final_energy = λ
                # Update the current site before breaking
                Q = _r2l_LQ(An_new)
                sites_updated += 1
                mps.A[n] = Q
                # Update right environment for consistency (optional)
                @tensor right_env_new[u, y, j] := right_env[o, p, l] * conj(Q)[u, i, o] * On[y, p, i, k] * Q[j, k, l]
                right_envs[n-1] = right_env_new
                break
            end
            λ_prev = λ

            # normalize and store
            Q = _r2l_LQ(An_new)
            sites_updated += 1
            mps.A[n] = Q

            # Update right environment
            @tensor right_env_new[u, y, j] := right_env[o, p, l] * conj(Q)[u, i, o] * On[y, p, i, k] * Q[j, k, l]
            right_envs[n-1] = right_env_new
        end
    end

    # Set return values
    if !converged
        final_energy = λ_prev
        sites_updated = -1
    end

    return (final_energy, sites_updated)
end