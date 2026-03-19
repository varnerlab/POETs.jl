using Documenter
using ParetoEnsembles

makedocs(
    sitename = "ParetoEnsembles.jl",
    modules = [ParetoEnsembles],
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
    repo = "github.com/varnerlab/ParetoEnsembles.jl.git",
    devbranch = "master",
)
