export MPS, MPO

struct MPO{T}
    O::Vector{Array{T,4}}
    N::Int
    d::Int

    function MPO{T}(O::Vector{Array{T,4}}, N::Int, d::Int) where {T}
        @assert length(O) == N
        new{T}(O, N, d)
    end
end

struct MPS{T}
    A::Vector{Array{T,3}}
    N::Int
    d::Int

    function MPS{T}(A::Vector{Array{T,3}}, N::Int, d::Int) where {T}
        @assert length(A) == N
        new{T}(A, N, d)
    end
end

function MPS{T}(N::Int, d::Int, D::Int) where {T}
    @assert N >= 1 && d >= 1 && D >= 1

    D_vec = Vector{Int}(undef, N + 1)
    D_vec[1] = 1
    D_vec[N+1] = 1
    for i in 2:N
        D_vec[i] = D
    end

    A = Vector{Array{T,3}}(undef, N)
    for i in 1:N
        Ai = Array{T,3}(undef, D_vec[i], d, D_vec[i+1])
        randn!(Ai)
        A[i] = Ai
    end

    return MPS{T}(A, N, d)
end