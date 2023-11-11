using Random
using Random: default_rng
include("../src/DiffusionMaps.jl")
using .DiffusionMaps
using Plots

function swiss_roll(n::Int = 1000, noise::Real=0.00; hlims=(-5.0,5.0),
    rng::AbstractRNG=default_rng())
    t = (3 * pi/2) * (1 .+ 2 * rand(rng, n, 1))
    height = (hlims[2]-hlims[1]) * rand(rng, n, 1) .+ hlims[1]
    X = [t .* cos.(t) height t .* sin.(t)]
    X .+= noise * randn(rng, n, 3)
    return collect((X))
end

# Generate the structure
training_data = swiss_roll(1000)

model = fit(DiffMap, training_data)
proj = projection(model)

s1 = scatter(training_data[:, 1], training_data[:, 2], training_data[:, 3], legend=false)
s2 = scatter(proj[:, 1], proj[:, 2], legend=false, size=(800, 400))

plot(s1, s2, layout = @layout[a b])



