include("../src/DiffusionMaps.jl")
using .DiffusionMaps
using Plots


function toroidal_helix(n)
    t = range(0, stop=2π, length=n)
    x = cos.(t) .* (1 .+ 0.4 .* cos.(10 .* t))  
    y = sin.(t) .* (1 .+ 0.4 .* cos.(10 .* t))  
    z = 0.1 .* sin.(10 .* t)

    points = hcat(x, y, z)  
    return points
end

# Generate the structure
training_data = toroidal_helix(1000)

model = fit(DiffMap, training_data)
proj = projection(model)

s1 = scatter(training_data[:, 1], training_data[:, 2], training_data[:, 3], legend=false)
s2 = scatter(proj[:, 1], proj[:, 2], legend=false, size=(800, 400))

plot(s1, s2, layout = @layout[a b])