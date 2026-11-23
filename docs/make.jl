using Svalinn
using Documenter

DocMeta.setdocmeta!(Svalinn, :DocTestSetup, :(using Svalinn); recursive=true)

makedocs(;
    modules=[Svalinn],
    authors="Facundo Sapienza <fsapienza@dm.uba.ar>",
    sitename="Svalinn.jl",
    format=Documenter.HTML(;
        canonical="https://facusapienza21.github.io/Svalinn.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/facusapienza21/Svalinn.jl",
    devbranch="main",
)
