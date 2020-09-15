using StatsFuns
using DistributionsAD
using Random: AbstractRNG, GLOBAL_RNG

abstract type AbstractMvSamplesNormal <: Distributions.ContinousMultivariateDistribution

struct SamplesMvNormal{
    T,
    Tx<:AbstractMatrix{T},
    Tμ<:AbstractVector{T},
    TΣ<:AbstractMatrix{T},
} <: AbstractMvSamplesNormal
    dim::Int
    n_particles::Int
    x::Tx
    μ::Tμ
    Σ::TΣ
    function SamplesMvNormal(x::M) where {T,M<:AbstractMatrix{T}}
        μ = vec(mean(x, dims = 2))
        Σ = cov(x, dims = 2)
        new{T,M,typeof(μ),typeof(Σ)}(size(x)..., x, μ, Σ)
    end
    function SamplesMvNormal(
        dim::Int,
        n_particles::Int,
        x::Tx,
        μ::Tμ,
        Σ::TΣ,
    ) where {T,Tx<:AbstractMatrix{T},Tμ<:AbstractVector{T},TΣ<:AbstractMatrix{T}}
        new{T,Tx,Tμ,TΣ}(dim, n_particles, x, μ, Σ)
    end
end

function update_q!(d::SamplesMvNormal)
    d.μ .= vec(mean(d.x, dims = 2))
    d.Σ .= cov(d.x, dims = 2)
    nothing
end

@functor SamplesMvNormal

Base.length(d::SamplesMvNormal) = d.dim
nParticles(d::SamplesMvNormal) = d.n_particles

# Random._rand!(d::SteinDistribution, v::AbstractVector) = d.x
Base.eltype(::SamplesMvNormal{T}) where {T} = T
function Distributions._rand!(rng::AbstractRNG, d::SamplesMvNormal, x::AbstractVector)
    nDim = length(x)
    nDim == d.dim || error("Wrong dimensions")
    x .= d.x[rand(rng, 1:d.n_particles), :]
end
function Distributions._rand!(rng::AbstractRNG, d::SamplesMvNormal, x::AbstractMatrix)
    nDim, nPoints = size(x)
    nDim == d.dim || error("Wrong dimensions")
    x .= d.x[rand(rng, 1:d.n_particles, nPoints), :]'
end
Distributions.mean(d::SamplesMvNormal) = d.μ
Distributions.cov(d::SamplesMvNormal) = d.Σ
Distributions.var(d::SamplesMvNormal) = diag(d.Σ)
Distributions.entropy(d::SamplesMvNormal) = 0.5 * (log2π + logdet(cov(d) + 1e-5I))

struct MFSamplesMvNormal{
    T,
    Tx<:AbstractMatrix{T},
    Ti<:AbstractVector{<:Int}
    Tμ<:AbstractVector{T},
    TΣ<:AbstractMatrix{T},
} <: AbstractMvSamplesNormal
    dim::Int
    n_particles::Int
    K::Int
    id::Ti
    x::Tx
    μ::Tμ
    Σ::TΣ
    function MFSamplesMvNormal(x::M, indices::AbstractVector{<:Int}) where {T,M<:AbstractMatrix{T}}
        K = length(indices) - 1
        μ = vec(mean(x, dims = 2))
        Σ = BlockDiagonal([cov(view(x, indices[i]:indices[i+1]-1, :) , dims = 2) for i in 1:K])
        new{T,M,typeof(indices), typeof(μ),typeof(Σ)}(size(x)..., indices, x, μ, Σ)
    end
    function MFSamplesMvNormal(
        dim::Int,
        n_particles::Int,
        K::Int,
        indices::Ti
        x::Tx,
        μ::Tμ,
        Σ::TΣ,
    ) where {T,Ti,Tx<:AbstractMatrix{T},Tμ<:AbstractVector{T},TΣ<:AbstractMatrix{T}}
        new{T,Ti,Tx,Tμ,TΣ}(dim, n_particles, K, indices, x, μ, Σ)
    end
end

function update_q!(d::MFSamplesMvNormal)
    d.μ .= vec(mean(d.x, dims = 2))
    d.Σ .= BlockDiagonal([cov(view(x, d.id[i]:d.id[i+1]-1, :) , dims = 2) for i in 1:d.K])
    nothing
end

@functor MFSamplesMvNormal

Base.length(d::MFSamplesMvNormal) = d.dim
nParticles(d::MFSamplesMvNormal) = d.n_particles

# Random._rand!(d::SteinDistribution, v::AbstractVector) = d.x
Base.eltype(::SamplesMvNormal{T}) where {T} = T
function Distributions._rand!(rng::AbstractRNG, d::SamplesMvNormal, x::AbstractVector)
    nDim = length(x)
    nDim == d.dim || error("Wrong dimensions")
    x .= d.x[rand(rng, 1:d.n_particles), :]
end
function Distributions._rand!(rng::AbstractRNG, d::SamplesMvNormal, x::AbstractMatrix)
    nDim, nPoints = size(x)
    nDim == d.dim || error("Wrong dimensions")
    x .= d.x[rand(rng, 1:d.n_particles, nPoints), :]'
end
Distributions.mean(d::SamplesMvNormal) = d.μ
Distributions.cov(d::SamplesMvNormal) = d.Σ
Distributions.var(d::SamplesMvNormal) = diag(d.Σ)
Distributions.entropy(d::SamplesMvNormal) = 0.5 * (log2π + logdet(cov(d) + 1e-5I))



const SampMvNormal = Union{MFSamplesMvNormal, SamplesMvNormal,TransformedDistribution{<:AbstractMvSamplesNormal}}

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

function vi(
    logπ::Function,
    alg::PFlowVI,
    q::AbstractMvSamplesNormal;
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
        logπ,
        [0.0];
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
    q::TransformedDistribution{<:AbstractMvSamplesNormal};
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
        logπ,
        [0.0];
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
    alg::PFlowVI{<:ForwardDiffAD},
    q,
    logπ,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...,
)
    f(x) = sum(mapslices(z -> phi(logπ, q, z), x, dims = 1))
    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(q.dist.x), chunk_size))
    config = ForwardDiff.GradientConfig(f, q.dist.x, chunk)
    ForwardDiff.gradient!(out, f, q.dist.x, config)
end

phi(logπ, q, x) = -eval_logπ(logπ, q, x)

function optimize!(
    vo,
    alg::PFlowVI,
    q::SampMvNormal,
    logπ,
    θ::AbstractVector{<:Real};
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing,
)
    alg_name = alg_str(alg)
    samples_per_step = nSamples(alg)
    max_iters = alg.max_iters

    optimizer = if Base.isiterable(typeof(optimizer))
        length(optimizer) == 2 || error("Optimizer should be of size 2 only")
        optimizer
    else
        fill(optimizer, 2)
    end

    diff_result = DiffResults.GradientResult(q.dist.x)

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end

    time_elapsed = @elapsed while (i < max_iters) # & converged

        _logπ = if !isnothing(hyperparams)
            logπ(hyperparams)
        else
            logπ
        end

        grad!(vo, alg, q, _logπ, θ, diff_result, samples_per_step)

        Δ = DiffResults.gradient(diff_result)

        Δ₁ = if alg.precondΔ₁
            q.dist.Σ * vec(mean(Δ, dims = 2))
        else
            vec(mean(Δ, dims = 2))
        end
        shift_x = q.dist.x .- q.dist.μ
        Δ₂ = compute_cov_part(q.dist, shift_x, Δ)

        # apply update rule
        Δ₁ = apply!(optimizer[1], q.dist.μ, Δ₁)
        Δ₂ = apply!(optimizer[2], q.dist.x, Δ₂)
        @. q.dist.x = q.dist.x - Δ₁ - Δ₂
        update_q!(q.dist)

        if !isnothing(hyperparams) && !isnothing(hp_optimizer)
            Δ = hp_grad(vo, alg, q, logπ, hyperparams)
            Δ = apply!(hp_optimizer, hyperparams, Δ)
            hyperparams .+= Δ
        end

        if !isnothing(callback)
            callback(i, q, hyperparams)
        end
        AdvancedVI.DEBUG && @debug "Step $i" Δ
        PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    return q
end

function compute_cov_part(q::MvSamplesNormal, x::AbstractMatrix, Δ::AbstractMatrix, alg::PFlowVI)
    ψ = mean(eachcol(Δ) .* transpose.(eachcol(x)))
    A = ψ - I
    Δ₂ = if alg.precondΔ₂
        B = inv(q.Σ) # Approximation hessian
        # B = Δ * Δ' # Gauss-Newton approximation
        tr(A' * A) / (tr(A^2) + tr(A' * B * A * q.Σ)) * A * x
    else
        A * x
    end
    return Δ₂
end

function compute_cov_part(q::MFMvSamplesNormal, x::AbstractMatrix, Δ::AbstractMatrix, alg::PFlowVI)
    @views A = [mean(eachcol(Δ[q.id[i]:q.id[i+1]-1, :]) .* transpose.(eachcol(X[q.id[i]:q.id[i+1]-1, :]))) - I for i in 1:q.K]
    Δ₂ = if alg.precondΔ₂
        B = inv(q.Σ) # Approximation hessian
        # B = Δ * Δ' # Gauss-Newton approximation
        tr(A' * A) / (tr(A^2) + tr(A' * B * A * q.Σ)) * A * x
    else
        A * x
    end
    return Δ₂
end

function (elbo::ELBO)(
    rng::AbstractRNG,
    alg::PFlowVI,
    q::TransformedDistribution{<:SamplesMvNormal},
    logπ::Function,
)

    res = sum(mapslices(x -> -phi(logπ, q, x), q.dist.x, dims = 1))

    if q isa TransformedDistribution
        res += entropy(q.dist)
    else
        res += entropy(q)
    end
    return res
end
