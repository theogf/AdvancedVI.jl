using AdvancedVI
const AVI = AdvancedVI
using Distributions
using Flux
using LinearAlgebra
using ProgressMeter

n_dim = 2
d_target = MvNormal(0.5 * ones(n_dim), randn(n_dim, n_dim) |> x->x * x')
mutable struct RobbinsMonro
    κ::Float64
    τ::Float64
    state::IdDict
end
  
function RobbinsMonro(κ::Real = 0.51, τ::Real = 1)
    @assert 0.5 < κ <= 1 "κ should be in the interval (0.5,1]"
    @assert τ > 0 "τ should be positive"
    RobbinsMonro(κ, τ, IdDict())
end

function Flux.Optimise.apply!(o::RobbinsMonro, x, Δ)
    κ = o.κ
    τ = o.τ
    n = get!(o.state, x, 1)
    Δ .*= 1 / (τ + n)^κ
    o.state[x] = n + 1
    return Δ
end
logπ(x) = logpdf(d_target, x)
## Running algorithm
# μ = Vector(mean(d_target))
μ = rand(n_dim)
μ = -2 * ones(n_dim)
Γ = Matrix(1.0 * I(n_dim))
# Γ = reshape([1.0, 0.5], n_dim, :)
#rand(n_dim, n_dim)
nSamples = 10
alg = AVI.GaussFlow(1, nSamples, false, false)
alg2 = AVI.GaussPFlow(1, false, false)

function MvNormal(q::AVI.PosteriorMvNormal)
    MvNormal(mean(q), cov(q)+ 1e-5I)
end

## Testing full rank
q = AVI.LowRankMvNormal(μ, Γ)
q2 = AVI.SamplesMvNormal(rand(MvNormal(μ, Γ * Γ'), nSamples))
opt = Descent(0.001)
opt = [Descent(0.1), RobbinsMonro(0.99, 50)]

using Plots
xlin = range(-10, 10, length = 100)
ylin = range(-10, 10, length = 100)
a = Animation()
@showprogress for i in 1:100 
    contour(xlin, ylin, (x,y)->pdf(d_target, [x,y]), clims = (0, 0.2), color = :red, colorbar = false, title = "i = $i")
    contour!(xlin, ylin, (x,y)->pdf(MvNormal(q), [x,y]), color = :blue)
    scatter!(eachrow(q2.x)...)
    contour!(xlin, ylin, (x,y)->pdf(MvNormal(mean(q2), cov(q2)), [x,y]), color = :green)
    vi(logπ, alg, q, optimizer = opt)
    vi(logπ, alg2, q2, optimizer = opt)
    frame(a)
end
gif(a)
## Testing sampling
p1 = contour(xlin, ylin, (x,y)->pdf(MvNormal(q), [x,y]), color = :blue, colorbar=false)
scatter!(eachrow(rand(q, 100))..., label="")
p2 = contour(xlin, ylin, (x,y)->pdf(MvNormal(q2), [x,y]), color = :blue, colorbar=false)
scatter!(eachrow(rand(q2, 100))..., label="")

## Testing mean-field
q = AVI.MFMvNormal(μ, diag(Γ))
q2 = AVI.MFSamplesMvNormal(rand(MvNormal(μ, Γ * Γ'), nSamples))
opt = Descent(0.001)
opt = [Descent(0.1), RobbinsMonro(0.99, 50)]

using Plots
xlin = range(-10, 10, length = 100)
ylin = range(-10, 10, length = 100)
a = Animation()
@showprogress for i in 1:100 
    contour(xlin, ylin, (x,y)->pdf(d_target, [x,y]), clims = (0, 0.2), color = :red, colorbar = false, title = "i = $i")
    contour!(xlin, ylin, (x,y)->pdf(MvNormal(q), [x,y]), color = :blue)
    scatter!(eachrow(q2.x)...)
    contour!(xlin, ylin, (x,y)->pdf(MvNormal(mean(q2), cov(q2)), [x,y]), color = :green)
    vi(logπ, alg, q, optimizer = opt)
    vi(logπ, alg2, q2, optimizer = opt)
    frame(a)
end
gif(a)

## Testing sampling
p1 = contour(xlin, ylin, (x,y)->pdf(MvNormal(q), [x,y]), color = :blue, colorbar=false)
scatter!(eachrow(rand(q, 100))..., label="")
p2 = contour(xlin, ylin, (x,y)->pdf(MvNormal(q2), [x,y]), color = :blue, colorbar=false)
scatter!(eachrow(rand(q2, 100))..., label="")
plot(p1, p2)