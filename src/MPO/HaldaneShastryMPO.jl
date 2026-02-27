export haldane_shastry_MPO, haldane_shastry_H_matrix

function cal_J(k::Int, N::Int)
    return (pi / N)^2 / (sinpi(k / N)^2)
end

# const Sz = 0.5 * [1 0; 0 -1]
# const Sp = [0 1.0; 0 0]
# const Sm = [0 0; 1.0 0]
# const I2 = [1.0 0; 0 1]

function haldane_shastry_MPO(N::Int)
    d = 2
    D = 3 * N - 1
    k_lst = collect(1:1:N-1)
    J_lst = cal_J.(k_lst, N)

    row = zeros(D, d, d)
    row[1, :, :] = I2
    for k in 1:N-1
        row[k+1, :, :] = J_lst[k] * Sz
        row[k+N, :, :] = 0.5 * J_lst[k] * Sp
        row[k+2*N-1, :, :] = 0.5 * J_lst[k] * Sm
    end

    column = zeros(D, d, d)
    column[2, :, :] = Sz
    column[N+1, :, :] = Sm
    column[2N, :, :] = Sp
    column[end, :, :] = I2

    D_vec = Vector{Int}(undef, N + 1)
    D_vec[1] = 1
    D_vec[N+1] = 1
    for i in 2:N
        D_vec[i] = D
    end

    # (Dl, Dr, d, d)
    O = Vector{Array{Float64,4}}(undef, N)
    for i in 1:N
        O[i] = zeros(D_vec[i], D_vec[i+1], d, d)
    end

    O[1][1, :, :, :] = row
    O[N][:, 1, :, :] = column

    O[2][1, :, :, :] = row
    O[2][:, end, :, :] = column
    for k in 2:N-1
        O[2][k+1, k, :, :] = I2
        O[2][k+N, k+N-1, :, :] = I2
        O[2][k+2*N-1, k+2*N-2, :, :] = I2
    end

    for i in 3:N-1
        O[i] = O[2]
    end

    mpo = MPO{Float64}(O, N, d)
    return mpo
end

function haldane_shastry_H_matrix(N::Int)
    H = zeros(2^N, 2^N)
    # j < l
    for j in 1:N-1
        for l in j+1:N
            k = l - j
            J = cal_J(k, N)

            # Sz_j Sz_l
            path = fill(I2, N)
            path[j] = J * Sz
            path[l] = Sz
            H .+= foldl(kron, path)

            # 1/2 Sp_j Sm_l
            path = fill(I2, N)
            path[j] = 0.5 * J * Sp
            path[l] = Sm
            H .+= foldl(kron, path)

            # 1/2 Sm_j Sp_l
            path = fill(I2, N)
            path[j] = 0.5 * J * Sm
            path[l] = Sp
            H .+= foldl(kron, path)
        end
    end
    return H
end
