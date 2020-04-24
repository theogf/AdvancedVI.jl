update(td::TransformedDistribution, θ...) = transformed(update(td.dist, θ...), td.transform)

update(d::TuringDiagMvNormal, μ, σ) = TuringDiagMvNormal(μ, σ)
function update(td::Union{<:TransformedDistribution{<:TuringDiagMvNormal},<:TuringDiagMvNormal}, θ::AbstractArray)
    μ, σ = θ[1:length(td)], θ[length(td) + 1:end]
    return update(td, μ, σ)
end

update(d::TuringDenseMvNormal, μ, L) = TuringDiagMvNormal(μ, L*L')
function update(td::Union{TransformedDistribution{<:TuringDenseMvNormal},TuringDenseMvNormal}, θ::AbstractArray)
    μ, L = θ[1:length(td)], make_triangular(θ[length(td) + 1:end], length(td))
    return update(td, μ, L)
end

function make_triangular(x, D)
    [ i>=j ? x[div(j*(j-1),2)+i] : 0 for i=1:D, j=1:D ]
end
