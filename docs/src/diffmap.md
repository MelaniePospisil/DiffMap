# Diffusion Maps

Add: a part where "what are diffusion maps" is described in two sentences

## Properties

**size(M):** Returns a tuple with the dimensions of the reduced output (=d) and observations (=the number of columns of the initial input).

**indim(M):** Returns the input dimensions of the model.

**outdim(M):** Returns the output dimensions of the model.

**eigvals(M):** Return eigenvalues of the kernel matrix of the model M.

**eigvecs(M):** Return eigenvectors of the kernel matrix of the model M.

**metric(M):** Returns the metric used to calculate the kernel matrix of the model M. Defaults to the here defined DiffMetric.

**kernel(M):** Returns the kernel matrix of the model M. This is the normalized and symmetric distance matrix. The distance is calculated with the metric function.

## Functions

**fit(::Type{DiffMap}, X::AbstractMatrix; metric::PreMetric=DiffMetric(1.0), maxoutdim::Int=2, ɛ::Real=1.0, α::Real=0.5, t::Int=1)::DiffMap:** Fit a DiffMap model to X using the specified parameters.

**predict(model::DiffMap, new_points::AbstractMatrix, k::Int = 10):** Embed new points into the reduced-dimensional space.

normalize_laplacian(A::AbstractMatrix, α::Real): Normalize the Laplacian matrix A using a power normalization factor α.
normalize_direction!(A::AbstractMatrix): Adjust the direction of columns in the matrix A.
show(io::IO, M::DiffMap): Show the DiffMap model M.
predict(model::DiffMap, new_points::AbstractMatrix, k::Int = 10): Embed new points into the reduced-dimensional space.