using Random
include("../src/DiffusionMaps.jl")
using .DiffusionMaps
using Plots
using MultivariateStats: PCA

layout = @layout [a b c]

function generate_spiral_points(n)
    points = zeros(n, 3)  # Matrix zur Speicherung der Punkte (x, y, z)

    theta = range(0, stop=8π, length=n)  # Winkelwerte für die Spirale

    for i in 1:n
        r = theta[i] / (8π)  # Radius wächst proportional zum Winkel
        z = 0.1 * theta[i] / (8π)  # z-Koordinate wächst linear mit dem Winkel
        
        x = r * cos(theta[i])  # x-Koordinate berechnen
        y = r * sin(theta[i])  # y-Koordinate berechnen
        
        points[i, :] = [x, y, z]  # Punkt in Matrix speichern
    end

    return points
end

# Generiere die Swiss Roll Daten
data = generate_spiral_points(5000)

# Extrahiere die Koordinaten der ursprünglichen Daten
x = data[:, 1]
y = data[:, 2]
z = data[:, 3]

# Erzeuge den Plot der Swiss Roll mit entsprechender Einfärbung
s1 = scatter(x, y, z, legend=false,
        xlabel="x", ylabel="y", zlabel="z", markersize=4,
        title="Spiral", size=(3000, 1300), tickfontsize=20)


modelSR = fit(DiffMap, data, ɛ=1)
DM_S = modelSR.proj
print(size(DM_S))

s2 = scatter(DM_S[:, 1], DM_S[:, 2], legend=false, xlabel="DM1", ylabel="DM2", 
       markersize=4, title="Diffusion", size=(3000, 1300), tickfontsize=20)


model2 = fit(PCA, data)
DM2 = projection(model2)
        
s3 = scatter(DM2[:, 1], DM2[:, 2], legend=false, xlabel="DM1", ylabel="DM2", 
    markersize=4, title="PCA", size=(3000, 1300), tickfontsize=20)
        
        
plot(s1, s2, s3, layout=layout)
