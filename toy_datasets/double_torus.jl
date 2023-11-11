using Random
include("../src/DiffusionMaps.jl")
using .DiffusionMaps
using Plots

function generate_double_torus_points(n)
    phi = rand(n) .* 2π
    theta = rand(n) .* 2π
    R = 2  # Radius des Haupttorus
    r = 1  # Radius des zweiten Torus
    
    x = cos.(phi) .* (R .+ r .* cos.(theta))
    y = sin.(phi) .* (R .+ r .* cos.(theta))
    z = r .* sin.(theta)
    
    return hcat(x, y, z)
end

# Generate the structure
training_data = generate_double_torus_points(1000)

model = fit(DiffMap, training_data)
proj = projection(model)

s1 = scatter(training_data[:, 1], training_data[:, 2], training_data[:, 3], legend=false, color="blue", markersize=4)
s2 = scatter(proj[:, 1], proj[:, 2], legend=false, size=(800, 400), color="blue", markersize=4)

plot(s1, s2, layout = @layout[a b])