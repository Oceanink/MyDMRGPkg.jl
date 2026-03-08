export transverse_ising_MPO, transverse_ising_ground_exact

# This generates the MPO of the following Hamiltonian
# H = - Σ_j J σ_j^x σ_{j+1}^x - h Σ_j σ_j^z

# const σz = [1 0; 0 -1]
# const σx = [0 1.0; 1 0]
# const I2 = [1.0 0; 0 1]

function transverse_ising_MPO(N::Int, J::Real, h::Real)
    d = 2
    D = 4

    row = zeros(D, d, d)
    row[1, :, :] = I2
    row[2, :, :] = -J * σx
    row[3, :, :] = -J * σx
    row[4, :, :] = -h * σz

    column = zeros(D, d, d)
    column[1, :, :] = -h * σz
    column[2, :, :] = σx
    column[3, :, :] = σx
    column[4, :, :] = I2

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
    O[2][1, 3, :, :] .= 0
    O[2][:, 4, :, :] = column
    O[2][3, 4, :, :] .= 0
    O[2][3, 3, :, :] = I2

    for i in 3:N-1
        O[i] = O[2]
    end

    mpo = MPO{Float64}(O, N, d)
    return mpo
end

function transverse_ising_ground_exact(N::Int, J::Real, h::Real)
    g = h / J
    m_lst = collect(1:2:2N-1)
    E_even = -J * sum(sqrt.(1 .+ g^2 .- 2g * cospi.(m_lst / N)))
    m_lst = collect(0:2:2N-2)
    E_odd = -J * sum(sqrt.(1 .+ g^2 .- 2g * cospi.(m_lst / N)))
    E = min(E_even, E_odd)
    return E
end