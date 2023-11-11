module DiffusionMaps

    using MultivariateStats: NonlinearDimensionalityReduction
    using Distances
    using Arpack: eigs
    using LinearAlgebra: Diagonal, eigen
    using SparseArrays

    import MultivariateStats: fit, predict, projection, reconstruct, indim, outdim, eigvals, eigvecs
    import Base.size

    include("diffmap.jl")

    export fit, predict, projection, reconstruct, DiffMap, metric, kernel, indim, outdim, eigvals, eigvecs
end #module