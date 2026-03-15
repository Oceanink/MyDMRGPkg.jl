export long_range_transverse_ising_MPO

# This generates the MPO of the following Hamiltonian
# H = - Σ_{1≤i<j≤N}J_{ij} σ_i^x σ_j^x - h Σ_j σ_j^z
# J_{ij} = J_k when k = |i-j|

function cal_J(k::Int, N::Int, α::Real)
    return 1 / abs(sinpi(k / N) * N / pi)^α
end

# const σz = [1 0; 0 -1]
# const σx = [0 1.0; 1 0]
# const I2 = [1.0 0; 0 1]

function long_range_transverse_ising_MPO(N::Int, α::Real, h::Real)
    d = 2
    D = N + 1
    k_lst = collect(1:1:N-1)
    J_lst = cal_J.(k_lst, N, α)

    row = zeros(D, d, d)
    row[1, :, :] = I2
    for k in 1:N-1
        row[k+1, :, :] = -J_lst[k] * σx
    end
    row[end, :, :] = -h * σz

    column = zeros(D, d, d)
    column[1, :, :] = -h * σz
    column[2, :, :] = σx
    column[end, :, :] = I2

    D_vec = fill(D, N + 1)
    D_vec[1] = 1
    D_vec[end] = 1

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
    end

    for i in 3:N-1
        O[i] = O[2]
    end

    mpo = MPO{Float64}(O, N, d)
    return mpo
end