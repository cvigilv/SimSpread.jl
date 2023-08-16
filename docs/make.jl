using SimSpread
using Documenter
using Pkg, Literate, Glob, CairoMakie

CairoMakie.activate!(type="svg")

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

# generate examples
TUTORIALS = joinpath(@__DIR__, "src", "tutorial")
SOURCE_FILES = Glob.glob("*.jl", TUTORIALS)
foreach(fn -> Literate.markdown(fn, TUTORIALS), SOURCE_FILES)

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
        assets=[],
    ),
    pages=[
        "Welcome to SimSpread.jl" => "index.md",
        "Tutorials" => [
            "Getting started" => "tutorial/getting-started-v3.md",
            # "Fisher's flowers" => "tutorial/fishers-flowers.md",
            # "Optimizing SimSpread" => "tutorial/optimizing-simspread.md",
        ],
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/cvigilv/SimSpread.jl/",
    branch = "gh-pages",
    devbranch="develop",
    devurl = "develop",
    deps = Pkg.add(["MLDatasets", "NamedArrays", "Distances", "DataFrames", "AlgebraOfGraphics", "CairoMakie"])
)
