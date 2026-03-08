import Base: eltype

export MPS, MPO, CuMPO, CuMPS

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

struct CuMPO{T}
    O::Vector{CuArray{T,4,CUDA.DeviceMemory}}
    N::Int
    d::Int

    function CuMPO{T}(O::Vector{CuArray{T,4,CUDA.DeviceMemory}}, N::Int, d::Int) where {T}
        new{T}(O, N, d)
    end
end

struct CuMPS{T}
    A::Vector{CuArray{T,3,CUDA.DeviceMemory}}
    N::Int
    d::Int

    function CuMPS{T}(A::Vector{CuArray{T,3,CUDA.DeviceMemory}}, N::Int, d::Int) where {T}
        new{T}(A, N, d)
    end
end


Base.eltype(::Type{MPS{T}}) where {T} = T
Base.eltype(::MPS{T}) where {T} = T
Base.eltype(::Type{MPO{T}}) where {T} = T
Base.eltype(::MPO{T}) where {T} = T

Base.eltype(::Type{CuMPS{T}}) where {T} = T
Base.eltype(::CuMPS{T}) where {T} = T
Base.eltype(::Type{CuMPO{T}}) where {T} = T
Base.eltype(::CuMPO{T}) where {T} = T