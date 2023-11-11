# Diffusion Maps

Diffusion Maps is a similarity preserving dimensionality reduction.

## `DiffMap` Struct

The `DiffMap` type represents a Diffusion Maps model constructed for `T` type data. It stores all the relevant information for the reduction.

```julia
struct DiffMap{T <: Real} <: NonlinearDimensionalityReduction
    X::AbstractMatrix{T}
    d::Number
    t::Int
    ɛ::Real
    metric::PreMetric
    λ::AbstractVector{T}
    V::AbstractMatrix{T}
    P::AbstractMatrix{T}
end
```
```@docs
DiffusionMaps.DiffMap
```

## `DiffMetric` Struct

Structure for the distance function used in the Diffusion Maps. It is the implementation of the Gaussian Kernel.

```julia
struct DiffMetric <: PreMetric 
    ɛ::Real
end

```

```@docs
DiffusionMaps.DiffMetric
```

## Properties

**size(M):** Returns a tuple with the dimensions of the reduced output (=d) and observations (=the number of columns of the initial input).
```@docs
size
```

**indim(M):** Returns the input dimensions of the model.
```@docs
indim
```

**outdim(M):** Returns the output dimensions of the model.
```@docs
outdim
```

**eigvals(M):** Return eigenvalues of the kernel matrix of the model M.
```@docs
eigvals
```

**eigvecs(M):** Return eigenvectors of the kernel matrix of the model M.
```@docs
eigvecs
```

**metric(M):** Returns the metric used to calculate the kernel matrix of the model M. Defaults to the here defined DiffMetric.
```@docs
DiffusionMaps.metric
```

**kernel(M):** Returns the kernel matrix of the model M. This is the normalized and symmetric distance matrix. The distance is calculated with the metric function.
```@docs
DiffusionMaps.kernel
```

## Functions

### Exported Functions

**fit(::Type{DiffMap}, X::AbstractMatrix; ...)::DiffMap**: Fit a DiffMap model to X using specified parameters.
```@docs
fit
```

**projection(M::DiffMap)**: Returns the lower-dimensional embeddings of the input data.
```@docs
projection
```

**predict(model::DiffMap, data::AbstractMatrix; k::Int = 1)**: Embed new points into the reduced-dimensional space.
```@docs
predict
```

**reconstruct(model::DiffMap, data::AbstractMatrix; k::Int = 1):** Calculate an approximation of the higher-dimensional input, given the lower-dimensional output.
```@docs
reconstruct
```

### Helper Functions

**normalize_laplacian(S::AbstractMatrix, α::Real)::AbstractMatrix**: Helper function within fit to normalize the Laplacian matrix.
```@docs
DiffusionMaps.normalize_laplacian
```

**normalize_direction!(A::AbstractMatrix)**: Helper function within fit to adjust the direction of columns in the matrix A.
```@docs
DiffusionMaps.normalize_direction!
```