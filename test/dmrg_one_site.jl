using Test
using MyDMRGPkg
using Random


@testset "One-site DMRG input canonicalization" begin
    Random.seed!(1234)

    N = 20
    d = 2
    D = 10
    mps = MPS{Float64}(N, d, D)

    @test !is_right_canonical(mps; atol=1e-8)

    r2l_LQ!(mps)
    @test is_right_canonical(mps; atol=1e-8)
end

@testset "One-site DMRG relative errors" begin
    Random.seed!(2026)

    N = 40
    d = 2
    D = 20
    BC = "PBC"

    mps = MPS{Float64}(N, d, D)
    r2l_LQ!(mps)
    @test is_right_canonical(mps; atol=1e-8)

    mpo = heisen_chain_MPO(N, BC)
    E_bethe = heisen_chain_Bethe(N, BC)

    max_sweeps = 2
    energies = DMRG_loop!(mps, mpo, max_sweeps, 1e-12)

    steps = collect(1:length(energies))
    rel_errors = abs.((energies .- E_bethe) ./ E_bethe)

    @test length(energies) == length(steps)
    @test all(isfinite, rel_errors)
    @test (energies[end-1] - energies[end]) < 1e-6


    mkpath("test/output")
    p1 = plot(
        steps,
        rel_errors;
        label="|E_dmrg - E_bethe| / |E_bethe|",
        xlabel="update steps",
        ylabel="relative error",
        linewidth=2,
        marker=:circle,
        markersize=2,
    )

    title!(p1, "One-site DMRG relative errors, N=$N, D=$D, $BC")

    savefig(p1, "test/output/one_site_relative_errors.png")
end
