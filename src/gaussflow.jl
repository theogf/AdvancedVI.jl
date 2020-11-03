using StatsFuns
using DistributionsAD
using Random: AbstractRNG, GLOBAL_RNG

abstract type AbstractLowRankMvNormal{T} <:
              Distributions.ContinuousMultivariateDistribution end

Base.eltype(::AbstractLowRankMvNormal{T}) where {T} = T
function Distributions._rand!(
  rng::AbstractRNG,
  d::AbstractLowRankMvNormal{T},
  x::AbstractVector,
) where {T}
  nDim = length(x)
  nDim == d.dim || error("Wrong dimensions")
  x .= d.μ + d.Γ * randn(rng, T, size(d.Γ, 2))
end
function Distributions._rand!(
  rng::AbstractRNG,
  d::AbstractLowRankMvNormal{T},
  x::AbstractMatrix,
) where {T}
  nDim, nPoints = size(x)
  nDim == d.dim || error("Wrong dimensions")
  x .= d.μ .+ d.Γ * randn(rng, T, nDim, size(d.Γ, 2))
end
Distributions.mean(d::AbstractLowRankMvNormal) = d.μ
Distributions.var(d::AbstractLowRankMvNormal) = vec(sum(d.Γ .* d.Γ, dims = 2))
Distributions.entropy(d::AbstractLowRankMvNormal) = 0.5 * (log2π + logdet(cov(d) + 1e-5I))

struct LowRankMvNormal{
    T,
    Tμ<:AbstractVector{T},
    TΓ<:AbstractMatrix{T},
} <: AbstractLowRankMvNormal{T}
    dim::Int
    μ::Tμ
    Γ::TΓ
    function LowRankMvNormal(μ::AbstractVector{T}, Γ::AbstractMatrix{T}) where {T}
        length(μ) == size(Γ, 1) || throw(DimensionMismatch("μ and Γ have incompatible sizes")) 
        new{T,typeof(μ),typeof(Γ)}(length(μ), μ, Γ)
    end
    function LowRankMvNormal(
        dim::Int,
        μ::Tμ,
        Γ::TΓ
    ) where {
        T,
        Tμ<:AbstractVector{T},
        TΓ<:AbstractMatrix{T},
    }
        length(μ) == size(Γ, 1) || throw(DimensionMismatch("μ and Γ have incompatible sizes")) 
        new{T,Tμ,TΓ}(dim, μ, Γ)
    end
end

Distributions.cov(d::LowRankMvNormal) = d.Γ * d.Γ'

@functor LowRankMvNormal

Base.length(d::AbstractLowRankMvNormal) = d.dim

# struct MFLowRankMvNormal{
#     T,
#     Tx<:AbstractMatrix{T},
#     Ti<:AbstractVector{<:Int},
#     Tμ<:AbstractVector{T},
# } <: AbstractSamplesMvNormal{T}
#     dim::Int
#     n_particles::Int
#     K::Int
#     id::Ti
#     x::Tx
#     μ::Tμ
#     function MFSamplesMvNormal(
#         x::M,
#         indices::AbstractVector{<:Int},
#     ) where {T,M<:AbstractMatrix{T}}
#         K = length(indices) - 1
#         μ = vec(mean(x, dims = 2))
#         return new{T,M,typeof(indices),typeof(μ)}(size(x)..., K, indices, x, μ)
#     end
#     function MFSamplesMvNormal(
#         dim::Int,
#         n_particles::Int,
#         K::Int,
#         indices::Ti,
#         x::Tx,
#         μ::Tμ,
#     ) where {T,Tx<:AbstractMatrix{T},Ti,Tμ<:AbstractVector{T}}
#         return new{T,Tx,Ti,Tμ}(dim, n_particles, K, indices, x, μ)
#     end
# end

# Distributions.cov(d::MFSamplesMvNormal) =
#     BlockDiagonal([cov(view(d.x, (d.id[i]+1):d.id[i+1], :), dims = 2) for i = 1:d.K])

# @functor MFSamplesMvNormal

# struct FullMFSamplesMvNormal{
#     T,
#     Tx<:AbstractMatrix{T},
#     Tμ<:AbstractVector{T},
# } <: AbstractSamplesMvNormal{T}
#     dim::Int
#     n_particles::Int
#     x::Tx
#     μ::Tμ
#     function FullMFSamplesMvNormal(
#         x::M,
#     ) where {T,M<:AbstractMatrix{T}}
#         μ = vec(mean(x, dims = 2))
#         return new{T,M,typeof(μ)}(size(x)..., x, μ)
#     end
#     function FullMFSamplesMvNormal(
#         dim::Int,
#         n_particles::Int,
#         x::Tx,
#         μ::Tμ,
#     ) where {T,Tx<:AbstractMatrix{T},Ti,Tμ<:AbstractVector{T}}
#         return new{T,Tx,Tμ}(dim, n_particles, x, μ)
#     end
# end

# Distributions.cov(d::FullMFSamplesMvNormal) = Diagonal(var(d.x, dims = 2))
# @functor FullMFSamplesMvNormal

const LRMvNormal = Union{
    #FullMFSamplesMvNormal,
    #MFSamplesMvNormal,
    LowRankMvNormal,
    TransformedDistribution{<:AbstractLowRankMvNormal},
}

"""
    GaussFlowVI(n_particles = 100, max_iters = 1000)

Gaussian Particle Flow Inference (PFlowVI) for a given model.
"""
struct GaussFlowVI{AD} <: VariationalInference{AD}
    max_iters::Int        # maximum number of gradient steps used in optimization
    nSamples::Int   # Number of samples per expectation
    precondΔ₁::Bool # Precondition the first gradient (mean)
    precondΔ₂::Bool # Precondition the second gradient (cov)
end

# params(alg::SteinVI) = nothing;#params(alg.kernel)

GaussFlowVI(args...) = GaussFlowVI{ADBackend()}(args...)
GaussFlowVI() = GaussFlowVI(100, 10, false, false)
nSamples(alg::GaussFlowVI) = alg.nSamples

alg_str(::GaussFlowVI) = "GaussFlowVI"

function vi(
    logπ::Function,
    alg::GaussFlowVI,
    q::AbstractLowRankMvNormal;
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing,
)
    DEBUG && @debug "Optimizing $(alg_str(alg))..."
    # Initial parameters for mean-field approx
    # Optimize
    optimize!(
        elbo,
        alg,
        transformed(q, Identity{1}()),
        logπ;
        optimizer = optimizer,
        callback = callback,
        hyperparams = hyperparams,
        hp_optimizer = hp_optimizer,
    )

    # Return updated `Distribution`
    return q
end

function vi(
    logπ::Function,
    alg::PFlowVI,
    q::TransformedDistribution{<:AbstractLowRankMvNormal};
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing,
)
    DEBUG && @debug "Optimizing $(alg_str(alg))..."
    # Initial parameters for mean-field approx
    # Optimize
    optimize!(
        elbo,
        alg,
        q,
        logπ;
        optimizer = optimizer,
        callback = callback,
        hyperparams = nothing,
        hp_optimizer = nothing,
    )

    # Return updated `Distribution`
    return q
end

function grad!(
    vo,
    alg::GaussFlowVI{<:ForwardDiffAD},
    q,
    logπ,
    x,
    out::DiffResults.MutableDiffResult,
    args...,
)
    f(x) = sum(mapslices(z -> phi(logπ, q, z), x, dims = 1))
    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(x), chunk_size))
    config = ForwardDiff.GradientConfig(f, x, chunk)
    ForwardDiff.gradient!(out, f, x, config)
end

function optimize!(
    vo,
    alg::GaussFlowVI,
    q::LRMvNormal,
    logπ;
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing,
)
    alg_name = alg_str(alg)
    samples_per_step = nSamples(alg)
    max_iters = alg.max_iters

    optimizer = if optimizer isa AbstractVector #Base.isiterable(typeof(optimizer))
        length(optimizer) == 2 || error("Optimizer should be of size 2 only")
        optimizer
    else
        fill(optimizer, 2)
    end

    x = zeros(length(q.dist), samples_per_step) # Storage for samples
    diff_result = DiffResults.GradientResult(x)

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end
    Δμ = similar(q.dist.μ)
    ΔΓ = similar(q.dist.Γ)
    time_elapsed = @elapsed while (i < max_iters) # & converged

        _logπ = if !isnothing(hyperparams)
            logπ(hyperparams)
        else
            logπ
        end

        Distributions.rand!(q, x)

        grad!(vo, alg, q, _logπ, x, diff_result, samples_per_step)

        Δ = DiffResults.gradient(diff_result)
        
        Δμ .= if alg.precondΔ₁
            cov(q.dist) * vec(mean(Δ, dims = 2))
        else
            vec(mean(Δ, dims = 2))
        end
        compute_cov_part!(ΔΓ, q.dist, x, Δ, alg)

        # apply update rule
        q.dist.μ .-= apply!(optimizer[1], q.dist.μ, Δμ)
        q.dist.Γ .-= apply!(optimizer[2], q.dist.Γ, ΔΓ)

        if !isnothing(hyperparams) && !isnothing(hp_optimizer)
            Δ = hp_grad(vo, alg, q, logπ, hyperparams)
            Δ = apply!(hp_optimizer, hyperparams, Δ)
            hyperparams .+= Δ
        end
        AdvancedVI.DEBUG && @debug "Step $i" Δ
        PROGRESS[] && (ProgressMeter.next!(prog))

        if !isnothing(callback)
            callback(i, q, hyperparams)
        end
        i += 1
    end

    return q
end

function compute_cov_part!(
    ΔΓ::AbstractMatrix,
    q::LowRankMvNormal,
    x::AbstractMatrix,
    Δ::AbstractMatrix,
    alg::GaussFlowVI,
)
    shift_x = x .- mean(q)
    n_dim, n_samples = size(x)
    ΔΓ .= q.Γ
    if alg.precondΔ₂
        A = Δ * x' / q.n_particles - I
        B = inv(cov(q)) # Approximation hessian
        # B = Δ * Δ' # Gauss-Newton approximation
        cond = tr(A' * A) / (tr(A^2) + tr(A' * B * A * q.Σ))
        lmul!(cond * A, Δ₂)
    else
        if n_samples < n_dim
            mul!(ΔΓ, Δ, shift_x' * q.Γ, Float32(inv(n_samples)), -1.0f0)
            # If N >> D it's more efficient to compute ϕ xᵀ first
        else
            mul!(ΔΓ, Δ * shift_x', q.Γ, Float32(inv(n_samples)), -1.0f0)
        end
    end
end