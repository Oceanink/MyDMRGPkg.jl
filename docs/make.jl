using MyDMRGPkg
using Documenter

DocMeta.setdocmeta!(MyDMRGPkg, :DocTestSetup, :(using MyDMRGPkg); recursive=true)

makedocs(;
    modules=[MyDMRGPkg],
    authors="Oceanink",
    sitename="MyDMRGPkg.jl",
    format=Documenter.HTML(;
        canonical="https://Oceanink.github.io/MyDMRGPkg.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Oceanink/MyDMRGPkg.jl",
    devbranch="master",
)
