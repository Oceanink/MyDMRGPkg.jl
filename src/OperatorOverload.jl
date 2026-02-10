import Base: *
import Base: +
import Base: eltype

export mpo_mean, mpo_variance

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
function *(mps1::MPS, mps2::MPS)
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

# Scalar scaling
*(a::Number, mps::MPS{T}) where {T} = MPS{promote_type(T, typeof(a))}([a .* A for A in mps.A], mps.N, mps.d)
*(mps::MPS, a::Number) = a * mps

# apply mpo on mps
function *(mpo::MPO{T1}, mps::MPS{T2}) where {T1,T2}
    @assert mpo.N == mps.N
    T = promote_type(T1, T2)
    A = Vector{Array{T,3}}(undef, mps.N)
    for n in 1:mps.N
        @tensor An[j, n, i, l, m] := mpo.O[n][j, l, i, k] * mps.A[n][n, k, m]
        An = reshape(An, prod(size(An)[1:2]), size(An)[3], prod(size(An)[4:5]))
        A[n] = An
    end
    return MPS{T}(A, mps.N, mps.d)
end

# add two mpo
function +(mpo1::MPO{T1}, mpo2::MPO{T2}) where {T1,T2}
    @assert mpo1.N == mpo2.N
    @assert mpo1.d == mpo2.d

    N = mpo1.N
    T = promote_type(T1, T2)
    D1 = size(mpo1.O[1], 2)
    D2 = size(mpo2.O[1], 2)
    D = D1 + D2

    O = Vector{Array{T,4}}(undef, N)
    O[1] = zeros(1, D, d, d)
    O[N] = zeros(D, 1, d, d)

    for n in 2:N-1
        O[n] = zeros(D, D, d, d)
    end

    O[1][1, 1:D1, :, :] = mpo1.O[1]
    O[1][1, D1+1:end, :, :] = mpo2.O[1]

    for n in 2:N-1
        O[n][1:D1, 1:D1, :, :] = mpo1.O[n]
        O[n][D1+1:end, D1+1:end, :, :] = mpo2.O[n]
    end

    O[N][1:D1, 1, :, :] = mpo1.O[N]
    O[N][D1+1:end, 1, :, :] = mpo2.O[N]

    return MPO{T}(O, N, mpo1.d)
end

