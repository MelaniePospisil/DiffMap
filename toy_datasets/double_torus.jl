using Random
include("../src/DiffusionMaps.jl")
using .DiffusionMaps
using Plots
using MultivariateStats: PCA

layout = @layout [a b c]

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

# Generiere die Swiss Roll Daten
data = generate_double_torus_points(10000)

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

