using Random
include("../src/DiffusionMaps.jl")
using .DiffusionMaps
using Plots
using MultivariateStats: PCA

layout = @layout [a b c]

function toroidal_helix(n)
    t = range(0, stop=2π, length=n)
    x = cos.(t) .* (1 .+ 0.4 .* cos.(10 .* t))  
    y = sin.(t) .* (1 .+ 0.4 .* cos.(10 .* t))  
    z = 0.1 .* sin.(10 .* t)

    points = hcat(x, y, z)  
    return points
end

# Generiere die Swiss Roll Daten
data = toroidal_helix(5000)

# Extrahiere die Koordinaten der ursprünglichen Daten
x = data[:, 1]
y = data[:, 2]
z = data[:, 3]

# Erzeuge den Plot der Swiss Roll mit entsprechender Einfärbung
s1 = scatter(x, y, z, legend=false,
        xlabel="x", ylabel="y", zlabel="z", markersize=4,
        title="Torodial Helix", size=(3000, 1300), tickfontsize=20)


modelSR = fit(DiffMap, data, ɛ=1)
DM_S = modelSR.proj

s2 = scatter(DM_S[:, 1], DM_S[:, 2], legend=false, xlabel="DM1", ylabel="DM2", 
       markersize=4, title="Diffusion", size=(3000, 1300), tickfontsize=20)


model2 = fit(PCA, data, maxoutdim=2)
DM2 = projection(model2)
       
s3 = scatter(DM2[:, 1], DM2[:, 2], legend=false, xlabel="DM1", ylabel="DM2", 
    markersize=4, title="PCA", size=(3000, 1300), tickfontsize=20)
       
       
plot(s1, s2, s3, layout=layout)

