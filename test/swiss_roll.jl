using Random
using Random: default_rng
include("../src/DiffusionMaps.jl")
using .DiffusionMaps
using Plots
using MultivariateStats: PCA

layout = @layout [a b c]

function swiss_roll(n::Int = 1000, noise::Real=0.00; segments=1, hlims=(-10.0,10.0),
    rng::AbstractRNG=default_rng())
    t = (3 * pi/2) * (1 .+ 2 * rand(rng, n, 1))
    height = (hlims[2]-hlims[1]) * rand(rng, n, 1) .+ hlims[1]
    X = [t .* cos.(t) height t .* sin.(t)]
    X .+= noise * randn(rng, n, 3)
    return collect((X))
end



function swiss_roll_boundary(n::Int = 1000, noise::Real=0.00; segments=1, hlims=(-10.0,10.0),
    rng::AbstractRNG=default_rng())
    boundary_points = [zeros(n) hlims[1] .+ (hlims[2]-hlims[1]) * rand(rng, n, 1) zeros(n)]
    return collect(transpose(boundary_points))
end

# Generiere die Swiss Roll Daten
data = swiss_roll(5000)

# Extrahiere die Koordinaten der ursprünglichen Daten
x = data[:, 1]
y = data[:, 2]
z = data[:, 3]

# Erzeuge den Plot der Swiss Roll mit entsprechender Einfärbung
s1 = scatter(x, y, z, legend=false,
        xlabel="x", ylabel="y", zlabel="z", markersize=4,
        title="Double Torus", size=(3000, 1300), tickfontsize=20)


modelSR = fit(DiffMap, data, ɛ=1)
DM_S = modelSR.proj

s2 = scatter(DM_S[:, 1], DM_S[:, 2], legend=false, xlabel="DM1", ylabel="DM2", 
       markersize=4, title="Diffusion", size=(3000, 1300), tickfontsize=10)

model2 = fit(PCA, data, maxoutdim=2)
DM2 = projection(model2)
       
s3 = scatter(DM2[:, 1], DM2[:, 2], legend=false, xlabel="DM1", ylabel="DM2", 
    markersize=4, title="PCA", size=(3000, 1300), tickfontsize=20)
       
       
plot(s1, s2, s3, layout=layout)

