include("../src/DiffusionMaps.jl")
using .DiffusionMaps
using Plots


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

# Generate the structure
training_data = generate_spiral_points(1000)

model = fit(DiffMap, training_data)
proj = projection(model)

s1 = scatter(training_data[:, 1], training_data[:, 2], training_data[:, 3], legend=false)
s2 = scatter(proj[:, 1], proj[:, 2], legend=false, size=(800, 400))

plot(s1, s2, layout = @layout[a b])


