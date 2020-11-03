using AdvancedVI
const AVI = AdvancedVI
using Distributions
using Flux
using ProgressMeter

n_dim = 2
d_target = MvNormal(0.5 * ones(n_dim), rand(n_dim, n_dim) |> x->x * x')
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
μ = Vector(mean(d_target))#rand(n_dim)
Γ = rand(n_dim, n_dim)
q = AVI.LowRankMvNormal(μ, Γ)
alg = AVI.GaussFlowVI(5, 1, false, false)
opt = Descent(0.001)
opt = RobbinsMonro(0.99, 20)
function MvNormal(q::LowRankMvNormal)
    MvNormal(mean(q), cov(q))
end
using Plots
xlin = range(-10, 10, length = 100)
ylin = range(-10, 10, length = 100)
a = Animation()
@showprogress for i in 1:100 
    vi(logπ, alg, q, optimizer = opt)
    contour(xlin, ylin, (x,y)->logpdf(d_target, [x,y]), clims = (-20, 1), color = :red, colorbar = false, title = "i = $i")
    contour!(xlin, ylin, (x,y)->logpdf(MvNormal(q), [x,y]), color = :blue)
    frame(a)
end
gif(a)

