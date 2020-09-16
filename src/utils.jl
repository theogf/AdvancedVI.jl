update(td::TransformedDistribution, θ...) = transformed(update(td.dist, θ...), td.transform)

update(d::TuringDiagMvNormal, μ, σ) = TuringDiagMvNormal(μ, σ)
function update(td::Union{<:TransformedDistribution{<:TuringDiagMvNormal},<:TuringDiagMvNormal}, θ::AbstractArray)
    μ, σ = θ[1:length(td)], θ[length(td) + 1:end]
    return update(td, μ, σ)
end

update(d::TuringDenseMvNormal, μ, L) = TuringDenseMvNormal(μ, L * L' + 1e-5I)
function update(td::Union{TransformedDistribution{<:TuringDenseMvNormal},TuringDenseMvNormal}, θ::AbstractArray)
    μ, L = θ[1:length(td)], make_triangular(θ[length(td) + 1:end], length(td))
    return update(td, μ, L)
end

function make_triangular(x, D)
    [i >= j ? x[div(j * (j - 1), 2)+i] : zero(eltype(x)) for i = 1:D, j = 1:D]
end

function eval_logπ(logπ, q::TransformedDistribution, x)
    z, logjac = forward(q.transform, x)
    return logπ(z) + logjac
end

function eval_logπ(logπ, q::Distribution, x)
    return logπ(x)
end

function hp_grad(vo, alg, q, logπ, hyperparameters, args...)
    ForwardDiff.gradient(x -> vo(alg, q, logπ(x), args...), hyperparameters)
end
