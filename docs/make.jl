using Documenter
using POETs

makedocs(
    sitename = "POETs.jl",
    modules = [POETs],
    authors = "Jeffrey Varner <jdv27@cornell.edu>",
    format = Documenter.HTML(
        canonical = "https://varnerlab.github.io/POETs.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Algorithm" => "algorithm.md",
        "API Reference" => "api.md",
        "Tutorial" => "tutorial.md",
    ],
)

deploydocs(
    repo = "github.com/varnerlab/POETs.jl.git",
    devbranch = "master",
)
