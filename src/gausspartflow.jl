using StatsFuns
using DistributionsAD
using KernelFunctions
using Random: AbstractRNG, GLOBAL_RNG

struct SamplesMvNormal{T,M<:AbstractMatrix{T}} <: Distributions.ContinuousMultivariateDistribution
    dim::Int
    n_particles::Int
    x::M
    μ::Vector{T}
    Σ::Matrix{T}
    function SamplesMvNormal(x::M) where {T, M<: AbstractMatrix{T}}
        new{T,M}(size(x)..., x, vec(mean(x, dims = 2)), cov(x, dims = 2))
    end
end

function update_q!(d::SamplesMvNormal)
    d.μ .= vec(mean(d.x, dims = 2))
    d.Σ .= cov(d.x, dims = 2)
    nothing
end

Base.length(d::SamplesMvNormal) = d.dim

# Random._rand!(d::SteinDistribution, v::AbstractVector) = d.x
Base.eltype(::SamplesMvNormal{T}) where {T} = T
function Distributions._rand!(rng::AbstractRNG, d::SamplesMvNormal, x::AbstractVector)
    nDim = length(x)
    @assert nDim == d.dim "Wrong dimensions"
    x .= d.x[rand(rng, 1:d.n_particles),:]
end
function Distributions._rand!(rng::AbstractRNG, d::SamplesMvNormal, x::AbstractMatrix)
    nDim, nPoints = size(x)
    @assert nDim == d.dim "Wrong dimensions"
    x .= d.x[rand(rng, 1:d.n_particles, nPoints),:]'
end
Distributions.mean(d::SamplesMvNormal) = d.μ
Distributions.cov(d::SamplesMvNormal) = d.Σ
Distributions.var(d::SamplesMvNormal) = diag(d.Σ)

const SampMvNormal = Union{<:SamplesMvNormal,<:TransformedDistribution{<:SamplesMvNormal}}

"""
    PFlowVI(n_particles = 100, max_iters = 1000)

Gaussian Particle Flow Inference (PFlowVI) for a given model.
"""
struct PFlowVI{AD} <: VariationalInference{AD}
    max_iters::Int        # maximum number of gradient steps used in optimization
    precondΔ₁::Bool # Precondition the first gradient (mean)
    precondΔ₂::Bool # Precondition the second gradient (cov)
end

# params(alg::SteinVI) = nothing;#params(alg.kernel)

PFlowVI(args...) = PFlowVI{ADBackend()}(args...)
PFlowVI() = PFlowVI(100, true, false)

alg_str(::PFlowVI) = "PFlowVI"

function vi(logπ::Function, alg::PFlowVI, q::SampMvNormal; optimizer = TruncatedADAGrad(), callback = nothing)
    DEBUG && @debug "Optimizing $(alg_str(alg))..."
    # Initial parameters for mean-field approx
    # Optimize
    optimize!(alg, transformed(q, Identity{length(q)}()), logπ, [0.0]; optimizer = optimizer, callback = callback)

    # Return updated `Distribution`
    return q
end

function vi(logπ::Function, alg::PFlowVI, q::TransformedDistribution{SampMvNormal}; optimizer = TruncatedADAGrad(), callback = nothing)
    DEBUG && @debug "Optimizing $(alg_str(alg))..."
    # Initial parameters for mean-field approx
    # Optimize
    optimize!(alg, q, logπ, [0.0]; optimizer = optimizer, callback = callback)

    # Return updated `Distribution`
    return q
end


function phi(q::TransformedDistribution, z, logπ)
    (w, detJ) = forward(q.transform, z)
    return - logπ(w) - detJ
end

function phi(q::Distribution, z, logπ)
    - logπ(z)
end

function optimize!(
    alg::PFlowVI,
    q::SampMvNormal,
    logπ,
    θ::AbstractVector{<:Real};
    optimizer = TruncatedADAGrad(),
    callback = nothing
)
    alg_name = alg_str(alg)
    max_iters = alg.max_iters

    # diff_result = DiffResults.GradientResult(θ)

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end

    time_elapsed = @elapsed while (i < max_iters) # & converged


        g = mapslices( x -> ForwardDiff.gradient(
            z -> phi(q, z, logπ), x), q.dist.x, dims = 1)

        Δ₁ = if alg.precondΔ₁
            q.dist.Σ * vec(mean(g, dims = 2))
        else
            vec(mean(g, dims = 2))
        end
        global shift_x = q.dist.x .- q.dist.μ
        ψ = mean(eachcol(g) .* transpose.(eachcol(shift_x)))
        global A = ψ - I
        Δ₂ = if alg.precondΔ₂
            2*tr(A'*A)/(tr(A^2)+tr(A'*inv(q.dist.Σ)*A*q.dist.Σ))*A*shift_x
        else
            A*shift_x
        end

        # apply update rule
        Δ₁ = apply!(optimizer, q.dist.μ, Δ₁)
        Δ₂ = apply!(optimizer, q.dist.x, Δ₂)
        @. q.dist.x = q.dist.x - Δ₁ - Δ₂

        update_q!(q.dist)

        if !isnothing(callback)
            callback(q, i)
        end
        AdvancedVI.DEBUG && @debug "Step $i" Δ
        PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    return q
end
