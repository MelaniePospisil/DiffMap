using Random
#using LinearAlgebra: Diagonal
#using Distances
#using StatsAPI: RegressionModel
#import StatsAPI: fit
#using Arpack 

#include("types.jl")

# Diffusion Map

"""
Dimension Reduction with Diffusion Maps
"""
struct DiffMap{T <: Real} <: NonlinearDimensionalityReduction
    X::AbstractMatrix{T}        # input data
    d::Number                   # number of dimensions of the output
    t::Int                      # timescale of the diffusion process; affects the granularity of the resulting representation
    ɛ::Real
    α::Real                     # normalization parameter
    metric::PreMetric           # the metric used for the kernel matrix
    λ::AbstractVector{T}        # biggest d eigenvalues of the distance matrix K
    V::AbstractMatrix{T}        # corresponding d eigenvalues of the distance matrix K
    K::AbstractMatrix{T}        # kernel matrix
    proj::AbstractMatrix{T}     # projection
end

"""
Structure for the here used distance function
"""
struct DiffMetric <: PreMetric 
    ɛ::Real
end
@inline (dist::DiffMetric)(x, y) = exp(-sum((x .- y) .^ 2)/dist.ɛ)

## properties
"""
    size(M)

Returns a tuple with the dimensions of the reduced output (=d)
and obervations (=the number of columns of the initial input).
"""
size(M::DiffMap) = size(M.proj)

"""
    indim(M)

Returns the input dimensions of the model.
"""
indim(M::DiffMap) = size(M.X, 1)

"""
    outdim(M)

Returns the output dimensions of the model.
"""
outdim(M::DiffMap) = M.d

"""
    eigvals(M)

Return eigenvalues of the kernel matrix of the model `M`.
"""
eigvals(M::DiffMap) = M.λ

"""
    eigvecs(M)

Return eigenvectors of the kernel matrix of the model `M`.
"""
eigvecs(M::DiffMap) = M.V

"""
    metric(M)

Returns the metric used to calculate the kernel matrix of the model `M`. 
Defaults to the here defined DiffMetric.
"""
metric(M::DiffMap) = M.metric

"""
    kernel(M)

Returns the kernel matrix of the model `M`. 
This is the normalized and symmetric distance matrix.
The distance is calculated with the metric function.
Call metric(M) to display the model's metric function.
"""
kernel(M::DiffMap) = M.K


# use
function fit(::Type{DiffMap}, X::AbstractMatrix; 
    metric::PreMetric=DiffMetric(1.0), 
    maxoutdim::Int=2,
    ɛ::Real=1.0, 
    α::Real=0.5,
    t::Int=1)::DiffMap    

    # transpose the data matrix to have the features as the rows
    X_t = transpose(X)
    # compute the Distancematrix 
    L = pairwise(DiffMetric(ɛ), eachcol(X_t))

    # Normalize the Distancematrix
    T = normalize_laplacian(L, α)
    

    # Eigendecomposition & reduction
    #a, b = KrylovKit.eigsolve(S, maxoutdim, :LR, eltype(S))
    #a, b = partialschur(S, which =:LM)
    #c, e = partialeigen(a)
    #println(c)
    #println(e)
    c, e= eigs(T, nev=maxoutdim, which=:LM)
    λ, V = real(c), real(e)

    # turn the eigenvectors in the same direction every time
    normalize_direction!(V)

    Y = (λ .^ t) .* V'
    # transpose the projection to get back to the representation of rows = data points
    Y = transpose(Y)
    
    return DiffMap(X, maxoutdim, t, ɛ, α, metric, λ, V, L, Y)
end

function normalize_laplacian(A::AbstractMatrix, α::Real)
    D=Diagonal(vec(sum(A, dims=2)))
    res= D^-(α) * A * D^-(α)
    return res
end

function normalize_direction!(A::AbstractMatrix)
    for j in 1:size(A, 2)
        if A[1, j] < 0
            A[:, j] .= -A[:, j]
        end
    end
end

# show
function show(io::IO, M::DiffMap)
    idim, odim = size(M)
    print(io, "DiffMap(indim = $idim, outdim = $odim, principalratio = $(r2(M)))")
end

#embedding of new points into the reduced dimension space.
function predict(model::DiffMap, new_points::AbstractMatrix, k::Int = 10)
    #check for matching dimensions. 
    if size(model.X, 1) != size(new_points, 1)
        error("Dimension Mismatch, model data has $(size(model.X, 1)) rows and new data has $(size(new_points, 1)) rows.")
    end
    
    #check if enough data is given
    n = min(k, size(model.X, 2))

    # Initialize similarity matrix to store the indices of the k most similar points.
    similarity_matrix_pos = Array{Int}(undef, n, size(new_points, 2))
    similarity_matrix = similar(similarity_matrix_pos, Float64)
    similarities = pairwise(model.metric, model.X, new_points)

    
    # Calculate similarity for each column (data point) in new_points.
    for i in 1:size(new_points, 2)
        # Get the indices of the k most similar points.
        indices = sortperm(similarities[:, i], rev=true)[1:n]

        # Store the indices in the similarity matrix.
        similarity_matrix_pos[:, i] = indices
        similarity_matrix[:, i] = similarities[indices]
    end

    # normalize the k neareast neighbours similarieties to get the weights
    for j in 1:size(similarity_matrix, 2)
        column_sum = sum(similarity_matrix[:, j])
        similarity_matrix[:, j] ./= column_sum
    end

    new_proj = Array{Float64}(undef, size(model.proj, 1), size(new_points, 2))
    for i in 1:size(new_points, 2)
        indices = similarity_matrix_pos[:, i]
        weighted_points = model.proj[:, indices]

        for j in 1:size(weighted_points, 2)
            weighted_points[:, j] = similarity_matrix[j, i] *  weighted_points[:, j]
        end
        new_proj[:, i] = sum(weighted_points, dims=2)
    end

    return new_proj

end




