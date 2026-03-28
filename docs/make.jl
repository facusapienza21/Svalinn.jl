using GeoReversal
using Documenter

DocMeta.setdocmeta!(GeoReversal, :DocTestSetup, :(using GeoReversal); recursive=true)

makedocs(;
    modules=[GeoReversal],
    authors="Facundo Sapienza <fsapienza@dm.uba.ar>",
    sitename="GeoReversal.jl",
    format=Documenter.HTML(;
        canonical="https://facusapienza21.github.io/GeoReversal.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/facusapienza21/GeoReversal.jl",
    devbranch="main",
)
