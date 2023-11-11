using Random
include("../src/DiffusionMaps.jl")
using .DiffusionMaps
using Plots

function generate_double_circle(n, m)
    xn = [] 
    yn = []     # Generate n points uniformly within the inner circle
    for _ in 1:n
        theta = rand() * 2 * π
        r = sqrt(rand() * 2^2)
        a = r * cos(theta)
        b = r * sin(theta)
        push!(xn, a)  # Use push! to add elements to arrays
        push!(yn, b)
    end
    x=hcat(xn, yn)

        theta = LinRange(0, 2π, m)
        xm = 12 * cos.(theta)
        ym = 12 * sin.(theta)
        y = hcat(xm, ym)
        return x, y
end

k=100
# Generate the structure
n, m = generate_double_circle(k, k)
training_data = (vcat(n, m))

model = fit(DiffMap, training_data, maxoutdim=1)
proj = projection(model)

s1 = scatter(training_data[1:k, 1], training_data[1:k, 2], 
            legend=false, color="blue")
scatter!(training_data[k+1:2k, 1], training_data[k+1:2k, 2], 
            legend=false, color="red", markerstrokecolor="red")

s2 = scatter(proj[1:k, 1], legend=false, 
            size=(800, 400), color = "blue", markerstrokecolor="blue")
scatter!(proj[k+1:2k, 1], legend=false, 
            size=(800, 400), color = "red", markerstrokecolor="red")

plot(s1, s2, layout = @layout[a b])