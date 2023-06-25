using Documenter
include("../src/DiffusionMaps.jl")
using .DiffusionMaps

makedocs(sitename="My Documentation", modules = [DiffusionMaps])
