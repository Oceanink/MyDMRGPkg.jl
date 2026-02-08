using Test
using MyDMRGPkg
using Random
using LinearAlgebra
using Plots


@testset "two_site_dmrg" begin
    # Write your tests here.
    include("dmrg_two_site.jl")
end

@testset "one_site_dmrg" begin
    # Write your tests here.
    include("dmrg_one_site.jl")
end
