module DiffusionMaps

    using MultivariateStats: NonlinearDimensionalityReduction
    using Distances
    using Arpack 
    using LinearAlgebra: Diagonal
    using StatsAPI: RegressionModel

    import MultivariateStats: fit, predict, projection, indim, outdim, eigvals, eigvecs
    import StatsAPI: fit
    import Base.size

    include("diffmap.jl")

    export fit, predict, projection, DiffMap, metric, kernel, indim, outdim, eigvals, eigvecs

    
end #module