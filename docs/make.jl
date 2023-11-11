using Documenter
include("../src/DiffusionMaps.jl")
using .DiffusionMaps

makedocs(
    sitename="DiffMap Documentation",
    modules=[DiffusionMaps],
    repo="https://github.com/MelaniePospisil/DiffusionMaps.git",
    format = Documenter.HTML(; repolink = "https://github.com/MelaniePospisil/DiffusionMaps.git")
)

