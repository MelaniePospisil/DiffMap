"""
Dimension Reduction with Diffusion Maps
"""

"""

    DiffMap{T <: Real} <: NonlinearDimensionalityReduction

The `DiffMap` type represents diffusion maps model constructed for `T` type data.
It stores all the relevant information for the reduction.
"""
struct DiffMap{T <: Real} <: NonlinearDimensionalityReduction
    X::AbstractMatrix{T}        # input data
    d::Number                   # number of dimensions of the output
    t::Int                      # timescale of the diffusion process; affects the granularity of the resulting representation
    ɛ::Real                     # Parameter for the metric
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


"""
    fit(::Type{DiffMap}, X::AbstractMatrix; 
        metric::PreMetric=DiffMetric(1.0), 
        maxoutdim::Int=2,
        ɛ::Real=1.0, 
        α::Real=0.5,
        t::Int=1)::DiffMap    

Fit a DiffMap model to `X` using the specified parameters.

# Arguments
* `::Type{DiffMap}`: Type of the DiffMap model.
* `X::AbstractMatrix`: Data matrix of observations. Each row of `X` is an observation.
    
# Keyword arguments
* `metric::PreMetric`: The metric used for computing the distance matrix.
    Default: DiffMetric(1.0)
* `maxoutdim::Int`: The dimension of the reduced space.
    Default: 2
* `ɛ::Real`: The scale parameter for the Gaussian kernel.
    Default: 1.0
* `α::Real`: A normalization parameter.
    Default: 0.5
* `t::Int`: The number of transitions.
    Default: 1
    
# Returns
* `DiffMap`: A DiffMap object representing the fitted model.

# Examples
X = rand(100, 3)            # toy data set, 100 observations with 3 features
model = fit(DiffMap, X)     # construct a DiffMap model
R = projection(model)       # Obtain the lower dimensional embedding

"""
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
    c, e= eigs(T, nev=maxoutdim, which=:LM)
    λ, V = real(c), real(e)

    # turn the eigenvectors in the same direction every time
    normalize_direction!(V)

    Y = (λ .^ t) .* V'
    # transpose the projection to get back to the representation of rows = data points
    Y = transpose(Y)
    
    return DiffMap(X, maxoutdim, t, ɛ, α, metric, λ, V, L, Y)
end

"""
normalize_laplacian(A::AbstractMatrix, α::Real)

**Helper function within fit.**
Normalize the Laplacian matrix `A` using a power normalization factor `α`.

# Arguments
- `A::AbstractMatrix`: The Laplacian matrix to be normalized.
- `α::Real`: The power normalization factor.

# Returns
- `res::AbstractMatrix`: The normalized Laplacian matrix.

This function computes the normalized Laplacian matrix by applying a power 
normalization factor `α` to the input matrix `A`. 

It first constructs a diagonal matrix `D` by summing the rows of `A` and 
placing the resulting sums on the diagonal. 

Then, it applies the power normalization operation `D^(-α) * A * D^(-α)` 
to obtain the normalized Laplacian matrix.
"""
function normalize_laplacian(A::AbstractMatrix, α::Real)
    D=Diagonal(vec(sum(A, dims=2)))
    res= D^-(α) * A * D^-(α)
    return res
end

"""
normalize_direction!(A::AbstractMatrix)

**Helper function within fit.**
Adjust the direction of columns in the matrix `A`.

# Arguments
- `A::AbstractMatrix`: The matrix whose column directions need to be normalized.

This function adjusts the direction of each column in the matrix `A`. 
It iterates over the columns of `A` and checks if the first element of 
each column is negative. If so, it flips the sign of all elements in that column, 
effectively normalizing its direction.
This makes sure, that the fitting of the same data has the same 
lower dimensional embedding and doesn't differ because of the eigenvector's sign.

Note: This function modifies the input matrix `A` in place.
"""
function normalize_direction!(A::AbstractMatrix)
    for j in 1:size(A, 2)
        if A[1, j] < 0
            A[:, j] .= -A[:, j]
        end
    end
end

"""
projection(M::DiffMap)

Returns the lower-dimensional embeddings of the input data.
The format (rows = features or rows = observations) corresponds to the input format.

# Arguments
- `M::DiffMap`: The diffusion map model obtained from fitting.


"""
function projection(M::DiffMap)
    return M.proj
end

# show
function show(io::IO, M::DiffMap)
    idim, odim = size(M)
    print(io, "DiffMap(indim = $idim, outdim = $odim, principalratio = $(r2(M)))")
end

"""
Embed new points into the reduced-dimensional space.

# Arguments
- `model::DiffMap`: The diffusion map model obtained from fitting.
- `new_points::AbstractMatrix`: The new points to be embedded.
- `k::Int = 10`: The number of nearest neighbors to consider.

This function embeds new points into the reduced-dimensional space of a DiffMap model. 
It therefore calculates the k nearest neigbours in the model's input data X 
and gives back the weightet average of their lower-dimensional embeddings.

# Returns
An array containing the embedded representation of the new points.

# Note
If the dimension of `new_points` is different from the dimension of the training data, a dimension mismatch error is raised.

The "new_points" are not added to the model. The model stays unchanged. 
To alter the model and add the new points, consider calculating a new DiffMap model with 
new_data = hcat(model.X, new_points)
new_model = fit(DiffMap, new_data)
"""
function predict(model::DiffMap, new_points::AbstractMatrix, k::Int = 10)
    #check for matching dimensions. 
    if size(model.X, 2) != size(new_points, 2)
        error("Dimension Mismatch, model data has $(size(model.X, 1)) rows and new data has $(size(new_points, 1)) rows.")
    end
    
    #check if enough data is given
    n = min(k, size(model.X, 1))
    X_t = transpose(model.X)
    new_points_t = transpose(new_points)

    # Initialize similarity matrix to store the indices of the k most similar points.
    similarity_matrix_pos = Array{Int}(undef, n, size(new_points_t, 2))
    similarity_matrix = similar(similarity_matrix_pos, Float64)
    similarities = pairwise(model.metric, X_t, new_points_t)
    
    # Calculate similarity for each row (data point) in new_points_t.
    for i in 1:size(new_points_t, 2)
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

    new_proj = Array{Float64}(undef, model.d, size(new_points_t, 2))

    for i in 1:size(new_points_t, 2)
        indices = similarity_matrix_pos[:, i]
        weighted_points = model.proj[indices, :]

        for j in 1:size(weighted_points, 2)
            weighted_points[:, j] = similarity_matrix[j, i] *  weighted_points[:, j]
        end
        new_proj[:, i] = sum(weighted_points, dims=1)
    end

    return transpose(new_proj)

end

using Random

data1 = rand(100, 3)
data2 = rand(10, 3)
model = fit(DiffMap, data1)

test = predict(model, data2, 3)




