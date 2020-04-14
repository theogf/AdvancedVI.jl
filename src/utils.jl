update(d::TuringDiagMvNormal, μ, σ) = TuringDiagMvNormal(μ, σ)
update(td::TransformedDistribution, θ...) = transformed(update(td.dist, θ...), td.transform)
function update(td::Union{<:TransformedDistribution{<:TuringDiagMvNormal},<:TuringDiagMvNormal}, θ::AbstractArray)
    μ, σ = θ[1:length(td)], θ[length(td) + 1:end]
    return update(td, μ, σ)
end
