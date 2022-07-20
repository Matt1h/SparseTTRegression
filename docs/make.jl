using SparseTTRegression
using Documenter

DocMeta.setdocmeta!(SparseTTRegression, :DocTestSetup, :(using SparseTTRegression); recursive=true)

makedocs(;
    modules=[SparseTTRegression],
    authors="Matthias Holzenkamp",
    repo="https://github.com/Matt1h/SparseTTRegression.jl/blob/{commit}{path}#{line}",
    sitename="SparseTTRegression.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Matt1h.github.io/SparseTTRegression.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Matt1h/SparseTTRegression.jl",
    devbranch="master",
)
