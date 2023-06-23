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



#### TESTING ##############################################

using Random: default_rng

#Random.seed!(123)

function swiss_roll(n::Int = 1000, noise::Real=0.00; segments=1, hlims=(-10.0,10.0),
            rng::AbstractRNG=default_rng())
    t = (3 * pi/2) * (1 .+ 2 * rand(rng, n, 1))
    height = (hlims[2]-hlims[1]) * rand(rng, n, 1) .+ hlims[1]
    X = [t .* cos.(t) height t .* sin.(t)]
    X .+= noise * randn(rng, n, 3)
    return collect(transpose(X))
end

function toroidal_helix(n)
    t = range(0, stop=2π, length=n)
    x = cos.(t) .* (1 .+ 0.4 .* cos.(10 .* t))  
    y = sin.(t) .* (1 .+ 0.4 .* cos.(10 .* t))  
    z = 0.1 .* sin.(10 .* t)

    points = hcat(x, y, z)  
    return points
end



function swiss_roll_boundary(n::Int = 1000, noise::Real=0.00; segments=1, hlims=(-10.0,10.0),
            rng::AbstractRNG=default_rng())
    boundary_points = [zeros(n) hlims[1] .+ (hlims[2]-hlims[1]) * rand(rng, n, 1) zeros(n)]
    return collect(transpose(boundary_points))
end


using Plots
layout = @layout [a b]

# Generiere die Swiss Roll Daten
data = toroidal_helix(1000)

# Extrahiere die Koordinaten der ursprünglichen Daten
x = data[1, :]
y = data[2, :]
z = data[3, :]

# Erzeuge den Plot der Swiss Roll mit entsprechender Einfärbung
s1 = scatter(x, y, z, legend=false,
        xlabel="x", ylabel="y", zlabel="z", markersize=4,
        title="Swiss Roll mit Farbe", size=(3000, 1300), tickfontsize=20)

#plot!(aspect_ratio=:equal)

modelSR = fit(DiffMap, data, ɛ=1)
DM_S = modelSR.proj
s2 = scatter(DM_S[1, :], DM_S[2, :], legend=false, xlabel="DM1", ylabel="DM2", 
       markersize=4, title="Diffusion", size=(3000, 1300), tickfontsize=20)





# Generiere die Swiss Roll Daten zum testen der predict funktion
#swiss_roll_data_test = swiss_roll(50)
#DM_test = predict(modelSR, swiss_roll_data_test, 10)

#scatter!(DM_test[1, :], DM_test[2, :], color=:red, markersize=10, size=(3000, 1300))


plot(s1, s2, layout=layout)






#data = toroidal_helix(2000)
#x = data[:, 1]
#y = data[:, 2]
#z = data[:, 3]


# Create the main plot with adjusted aesthetics
#s1 = scatter(x, y, z, marker_z=z, color=:plasma, legend=false,
        #xlabel="x", ylabel="y", zlabel="z", markersize=8, #linewidth=2,
        #title="toroidal_helix", size=(800, 600), #tickfontsize=12)

# Adjusting camera view
#plot!(s1, camera=(45, 60))

# Setting z-axis limits
#zlims!(s1, (-2, 2))

#display(s1)

#reduced_data = fit(DiffMap, transpose(data))
#DM_S = reduced_data.proj

#s2 = scatter(DM_S[1, :], DM_S[2, :], color=:plasma, #legend=false, xlabel="DM1", ylabel="DM2", 
#       markersize=4, title="Diffusion", size=(3000, 1300), tickfontsize=20)









#data = generate_spiral_points(2000)
#x = data[:, 1]
#y = data[:, 2]
#z = data[:, 3]


# Create the main plot with adjusted aesthetics
#s1 = scatter(x, y, z, legend=false,
 #       xlabel="x", ylabel="y", zlabel="z", markersize=8, linewidth=2,
 #      title="spiral", size=(800, 600), tickfontsize=12)


#reduced_data = fit(DiffMap, transpose(data))
#DM_S = reduced_data.proj
       
#s2 = scatter(DM_S[1, :], DM_S[2, :], legend=false, xlabel="DM1", ylabel="DM2", 
#       markersize=4, title="Diffusion", size=(3000, 1300), tickfontsize=20)
       
       
       
#plot(s1, s2)

