export heisen_chain_MPO

function heisen_chain_MPO(N::Int, BC::String)::MPO
    @assert BC == "OBC" || BC == "PBC"
    if BC == "OBC"
        return heisen_chain_MPO_OBC(N)
    elseif BC == "PBC"
        return heisen_chain_MPO_PBC(N)
    end
end


function heisen_chain_MPO_OBC(N::Int)::MPO
    hbar = 1
    Sz = (hbar / 2) * [1 0; 0 -1]
    Sp = hbar * [0 1; 0 0]
    Sm = hbar * [0 0; 1 0]
    I2 = [1 0; 0 1]

    d = 2

    row = zeros(4, d, d)
    row[1, :, :] = I2
    row[2, :, :] = Sp
    row[3, :, :] = Sm
    row[4, :, :] = Sz

    column = zeros(4, d, d)
    column[1, :, :] = 0.5 * Sm
    column[2, :, :] = 0.5 * Sp
    column[3, :, :] = Sz
    column[4, :, :] = I2

    D_vec = Vector{Int}(undef, N + 1)
    D_vec[1] = 1
    D_vec[N+1] = 1
    for i in 2:N
        D_vec[i] = 5
    end

    # (Dl, Dr, d, d)
    O = Vector{Array{Float64,4}}(undef, N)
    for i in 1:N
        O[i] = zeros(D_vec[i], D_vec[i+1], d, d)
    end

    O[1][1, 1:4, :, :] = row
    O[N][2:5, 1, :, :] = column

    O[2][1, 1:4, :, :] = row
    O[2][2:5, 5, :, :] = column

    for i in 3:N-1
        O[i] = O[2]
    end

    mpo = MPO{Float64}(O, N, d)

    return mpo
end


function heisen_chain_MPO_PBC(N::Int)::MPO
    hbar = 1
    Sz = (hbar / 2) * [1 0; 0 -1]
    Sp = hbar * [0 1; 0 0]
    Sm = hbar * [0 0; 1 0]
    I2 = [1 0; 0 1]

    d = 2

    row = zeros(7, d, d)
    row[1, :, :] = I2
    row[2, :, :] = Sp
    row[3, :, :] = Sm
    row[4, :, :] = Sz
    row[5, :, :] = Sp
    row[6, :, :] = Sm
    row[7, :, :] = Sz

    column = zeros(7, d, d)
    column[1, :, :] = 0.5 * Sm
    column[2, :, :] = 0.5 * Sp
    column[3, :, :] = Sz
    column[4, :, :] = 0.5 * Sm
    column[5, :, :] = 0.5 * Sp
    column[6, :, :] = Sz
    column[7, :, :] = I2

    D_vec = Vector{Int}(undef, N + 1)
    D_vec[1] = 1
    D_vec[N+1] = 1
    for i in 2:N
        D_vec[i] = 8
    end

    # (Dl, Dr, d, d)
    O = Vector{Array{Float64,4}}(undef, N)
    for i in 1:N
        O[i] = zeros(D_vec[i], D_vec[i+1], d, d)
    end

    O[1][1, 1:7, :, :] = row
    O[N][2:8, 1, :, :] = column

    O[2][1, 1:4, :, :] = row[1:4, :, :]
    O[2][2:4, 8, :, :] = column[1:3, :, :]
    for i in 5:8
        O[2][i, i, :, :] = I2
    end

    for i in 3:N-1
        O[i] = O[2]
    end

    mpo = MPO{Float64}(O, N, d)

    return mpo
end