using Random
include("../src/DiffusionMaps.jl")
using .DiffusionMaps
using Plots
using MultivariateStats: PCA

layout = @layout [a b c]

function generate_double_circle(n, m)
        xn = rand(n) * 2 .-1
        yn = rand(n) * 2 .-1

        theta = LinRange(0, 2π, m)
        xm = 5 * cos.(theta)
        ym = 5 * sin.(theta)
        return xn, yn, xm, ym
end


# Generate the Swiss roll dataset with 1000 points
xn, yn, xm, ym = generate_double_circle(200, 200)

# Erzeuge den Plot der Swiss Roll mit entsprechender Einfärbung
s1 = scatter(xn, yn, legend=false,
        xlabel="x", ylabel="y", markersize=10, color="red",
        title="Oo", size=(3000, 1300), tickfontsize=40)
scatter!(xm, ym, legend=false,
        xlabel="x", ylabel="y", markersize=10, color="blue",
        title="Oo", size=(3000, 1300), tickfontsize=40)

x_data = vcat(xn, xm)
y_data = vcat(yn, ym)
data = hcat(x_data, y_data)


modelSR = fit(DiffMap, data, maxoutdim=1, ɛ=1)
DM_S = modelSR.proj


s2 = scatter(zeros(200), DM_S[1:200], legend=false, xlabel="DM", color = "red",
        markersize=10, title="Diffusion", size=(3000, 1300), tickfontsize=10)

scatter!(zeros(200), DM_S[201:400], legend=false, xlabel="DM", color = "blue",
        markersize=10, title="Diffusion", size=(3000, 1300), tickfontsize=10)

model2 = fit(PCA, data, maxoutdim=1)
DM2 = projection(model2)
        
s3 = scatter(zeros(200), DM2[1:200], legend=false, xlabel="DM", color = "red",
        markersize=10, title="Diffusion", size=(3000, 1300), tickfontsize=10)

scatter!(zeros(200), DM2[201:400], legend=false, xlabel="DM", color = "blue",
        markersize=3, title="Diffusion", size=(3000, 1300), tickfontsize=10)
        
        
plot(s1, s2, s3, layout=layout)
