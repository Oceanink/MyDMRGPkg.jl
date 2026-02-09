import Base: *
import Base: reshape!
import Base: eltype


export is_left_canonical, is_right_canonical, mps_inner_porduct, mps_norm, mpo_mean, mpo_variance, r2l_LQ!, mps_padding!

function is_left_canonical(mps::MPS; atol::Float64=1e-9)
    N = mps.N
    for n in 1:N
        A = mps.A[n]
        Dl, d, Dr = size(A)
        A_mat = reshape(A, Dl * d, Dr)
        if !isapprox(A_mat' * A_mat, I(Dl); atol=atol, rtol=0.0)
            return false
        end
    end
    return true
end

function is_right_canonical(mps::MPS; atol::Float64=1e-9)
    N = mps.N
    for n in 1:N
        A = mps.A[n]
        Dl, d, Dr = size(A)
        A_mat = reshape(A, Dl, d * Dr)
        if !isapprox(A_mat * A_mat', I(Dl); atol=atol, rtol=0.0)
            return false
        end
    end
    return true
end

function mps_inner_porduct(mps1::MPS, mps2::MPS)
    @assert mps1.N == mps2.N
    N = mps1.N
    C = ones(1, 1)

    for n in 1:N
        An = mps1.A[n]
        Bn = mps2.A[n]
        An_dag = conj(An)
        @tensor C[k, l] := C[i, j] * An_dag[j, d, l] * Bn[i, d, k]
    end

    return C[1, 1]
end

function mps_norm(mps::MPS)::Float64
    N = mps.N
    C = ones(ComplexF64, 1, 1)
    for n in 1:N
        An = mps.A[n]
        An_dag = conj(An)
        @tensor C[k, l] := C[i, j] * An_dag[j, d, l] * An[i, d, k]
    end

    return sqrt(abs(C[1, 1]))
end

function r2l_LQ!(mps::MPS)
    """Right-to-left LQ decomposition, modifies MPS in-place"""
    N = mps.N

    An = mps.A[N]
    Dl, d, Dr = size(An)
    An_mat = reshape(An, Dl, d * Dr)

    for n in N-1:-1:1
        L, Q = lq(An_mat)
        Q = Matrix(Q)

        mps.A[n+1] = reshape(Q, size(Q, 1), d, Dr)

        @tensor An[i, j, l] := mps.A[n][i, j, k] * L[k, l]
        Dl, d, Dr = size(An)
        An_mat = reshape(An, Dl, d * Dr)
    end

    mps.A[1] = An / norm(An)
    return nothing
end

function mps_padding!(mps::MPS{T}, D_new::Int) where {T}
    N = mps.N
    D_vec = Vector{Int}(undef, N + 1)
    D_vec[1] = 1
    D_vec[N+1] = 1
    for i in 2:N
        D_vec[i] = D_new
    end

    for i in 1:N
        Dl, d, Dr = size(mps.A[i])
        Ai_new = zeros(T, D_vec[i], d, D_vec[i+1])
        Ai_new[1:Dl, :, 1:Dr] = mps.A[i]
        mps.A[i] = Ai_new
    end

    return nothing
end

function apply_mpo(mpo::MPO, mps::MPS)::MPS
    @assert mpo.N == mps.N
    T1 = eltype(mpo)
    T2 = eltype(mps)
    T = promote_type(T1, T2)
    A = Vector{Array{T,3}}(undef, mps.N)
    for n in 1:mps.N
        @tensor An[j, n, i, l, m] := mpo.O[n][j, l, i, k] * mps.A[n][n, k, m]
        An = reshape(An, prod(size(An)[1:2]), size(An)[3], prod(size(An)[4:5]))
        A[n] = An
    end
    return MPS{T}(A, mps.N, mps.d)
end

function mpo_mean(mps::MPS, mpo::MPO)
    @assert mps.N == mpo.N
    C = ones(1, 1, 1)
    for i in 1:mps.N
        @tensor C[o, l, m] = C[u, j, n] * conj(mps.A[i])[u, i, o] * mpo.O[i][j, l, i, k] * mps.A[i][n, k, m]
    end
    return C[1, 1, 1]
end

function mpo_variance(mps::MPS, mpo::MPO)
    @assert mps.N == mpo.N
    mps_H_psi = mpo * mps
    psi_H_psi = mps * mps_H_psi
    psi_HSquare_psi = mps_H_psi * mps_H_psi
    return psi_HSquare_psi - psi_H_psi^2
end


# Inner product
*(mps1::MPS, mps2::MPS) = mps_inner_porduct(mps1, mps2)

*(mpo::MPO, mps::MPS) = apply_mpo(mpo, mps)

# Scalar scaling
*(a::Number, mps::MPS{T}) where {T} = MPS{promote_type(T, typeof(a))}([a .* A for A in mps.A], mps.N, mps.d)

*(mps::MPS, a::Number) = a * mps