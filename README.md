
# Diffusion Maps

  

Diffusion Maps is a Julia module for dimension reduction using the Diffusion Maps algorithm. It provides a simple and efficient way to reduce the dimensionality of high-dimensional data while preserving the underlying structure. This module is part of the bachelor's thesis at TU Munich.

  

## Usage

  

### Access

  

To access the here implemented diffusion map functions, you need to add

  

```

include("../src/DiffusionMaps.jl")

using .DiffusionMaps

````

at the beginning of your code. replace the "../src/DiffusionMaps.jl" with the path to your local version of the DiffusionMaps.jl file.

  

### Creating a Diffusion Map model

  

You can create a DiffMap model by calling the fit function:

  

```

using Random

  

X = rand(100, 3) # toy data set, 100 observations with 3 features

model = fit(DiffMap, X) # construct a DiffMap model

```

  

The fit function takes the following arguments:

  

- **::Type{DiffMap}:** Type of the DiffMap model.

- **X::AbstractMatrix:** Data matrix of observations. Each row of X is an observation.

You can specify additional keyword arguments to customize the model:

  

- **metric::PreMetric:** The metric used for computing the distance matrix. (Default: DiffMetric(1.0))

- **maxoutdim::Int:** The dimension of the reduced space. (Default: 2)

- **ɛ::Real:** The scale parameter for the Gaussian kernel. (Default: 1.0)

- **α::Real:** A normalization parameter. (Default: 0.5)

- **t::Int:** The number of transitions. (Default: 1)

  

### Get the lower dimensional projection

  

To get the lower dimensional projection, call  ``` project(model)``` on the created model.

  

### Embedding New Points

  

You can embed new points into the reduced-dimensional space using the predict function:

  

```

new_points = rand(10, 3) # new points to be embedded

embedding = predict(model, new_points) # embed new points using the model

```

The predict function takes the following arguments:

  

- **model::DiffMap:** The diffusion map model obtained from fitting.

- **new_points::AbstractMatrix:** The new points to be embedded.

- **k::Int = 10:** The number of nearest neighbors to consider. (Default: 10)

## Accessing the model's properties

You can access various properties of the DiffMap model:

- **size(model):** Returns a tuple with the dimensions of the reduced output and observations.
- **indim(model):** Returns the input dimensions of the model.
- **outdim(model):** Returns the output dimensions of the model.
- **eigvals(model):** Returns the eigenvalues of the kernel matrix of the model.
- **eigvecs(model):** Returns the eigenvectors of the kernel matrix of the model.
- **metric(model):** Returns the metric used to calculate the kernel matrix of the model.
- **kernel(model):** Returns the kernel matrix of the model.

## Toy data sets

There are a few toy data sets including vizualization with a: The oroginal data b: the lower embedding using DIffMaps and c: the lower embedding using PCA (for comparisson.)

Those toy data sets includ:
- a double torus
- Oo_date (A small circle surrounded by a bigger circle)
- a spiral
- a swiss roll
- a torus helix
