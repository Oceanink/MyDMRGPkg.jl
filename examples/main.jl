using MyDMRGPkg
using CSV
# %%
function run_dmrg(α, N_lst, D_lst, h_lst)
    max_loops = 5
    d = 2
    co2_N_h = zeros(length(N_lst), length(h_lst))
    λ_N_h = zeros(length(N_lst), length(h_lst))

    # Initialize CSV file for this α value
    csv_dir = "./examples/csv"
    mkpath(csv_dir)
    csv_path = joinpath(csv_dir, "LRTFIM_alpha=$α.csv")
    # Write header to CSV file only if file doesn't exist
    if !isfile(csv_path)
        open(csv_path, "w") do io
            println(io, "size,h,crosscap_overlap,energy,bond_dim")
        end
    end

    for i in eachindex(N_lst)
        N = N_lst[i]
        D = D_lst[i]
        mps = MPS{Float64}(N, d, D)
        r2l_LQ!(mps)
        # mps_gpu = cu(mps)
        for j in eachindex(h_lst)
            h = h_lst[j]
            println("Start N=$N ", "h=$h ", "α=$α")
            mpo = long_range_transverse_ising_MPO(N, α, h)
            # mpo_gpu = cu(mpo)
            λs, _ = DMRG_loop_2site!(mps, mpo, max_loops, 1e-12; store_all=false, show_progress=false)
            # mps = cpu(mps_gpu)
            co = crosscap_overlap(mps)
            co2_N_h[i, j] = abs(co)
            λ_N_h[i, j] = λs[end]

            # Append result to CSV
            open(csv_path, "a") do io
                println(io, "$N,$h,$(abs2(co)),$(λs[end]),$D")
            end
        end
    end

    return co2_N_h, λ_N_h
end

# Parse command line arguments if provided
# Usage: julia --project=. main.jl N_lst D_lst h_min h_step h_max
if length(ARGS) >= 5
    N_lst = parse.(Int, split(ARGS[1], ","))
    D_lst = parse.(Int, split(ARGS[2], ","))
    h_min = parse(Float64, ARGS[3])
    h_step = parse(Float64, ARGS[4])
    h_max = parse(Float64, ARGS[5])
    h_lst = collect(h_min:h_step:h_max)
else
    N_lst = [20, 32, 48, 60]
    D_lst = [40, 50, 60, 80]
    h_lst = collect(0.8:0.1:2)
end

α = 3
co2_N_h, λ_N_h = run_dmrg(α, N_lst, D_lst, h_lst)