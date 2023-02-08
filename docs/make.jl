using SimSpread
using Documenter

DocMeta.setdocmeta!(SimSpread, :DocTestSetup, :(using SimSpread); recursive=true)

makedocs(;
    modules=[SimSpread],
    authors="Carlos Vigil Vásquez <carlos.vigil.v@gmail.com> and contributors",
    repo="https://github.com/cvigilv/SimSpread.jl/blob/{commit}{path}#{line}",
    sitename="SimSpread.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cvigilv.github.io/SimSpread.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cvigilv/SimSpread.jl",
    devbranch="main",
)
