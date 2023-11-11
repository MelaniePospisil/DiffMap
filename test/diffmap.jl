using Test
include("../src/DiffusionMaps.jl")
using .DiffusionMaps
using Distances

@testset "DiffusionMap Test" begin
    # Generate test data
    data = ([0.0 0.0 0.0; 1.0 1.0 1.0; 2.0 2.0 2.0; 3.0 3.0 3.0; 4.0 4.0 4.0])

    # fit a model
    model = fit(DiffMap, data, ɛ=1.0, maxoutdim=2, t=1)
    proj = projection(model)

    # testing fit
    @test size(proj) == (5, 2)
    @test indim(model) == 3
    @test outdim(model) == 2
    @test length(eigvals(model)) == 2
    @test size(eigvecs(model), 2) == 2
    @test size(eigvecs(model), 1) == size(data, 1)
    @test metric(model) isa PreMetric
    @test kernel(model) isa Matrix

    # testing projection
    reduced_data = projection(model)
    @test size(reduced_data) == (size(data, 1), 2)

    # testing predict
    new_points = ([0.5 0.5 0.5; 2.5 2.5 2.5])
    predicted_data = predict(model, new_points)
    @test size(predicted_data) == (size(new_points, 1), 2)
end
