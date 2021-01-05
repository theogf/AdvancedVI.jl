## Series of variation of the MvNormal distribution, different methods need different parametrizations ##
abstract type PosteriorMvNormal{T} <:
              Distributions.ContinuousMultivariateDistribution end

Base.eltype(::PosteriorMvNormal{T}) where {T} = T
Distributions.mean(d::PosteriorMvNormal) = d.μ
rank(d::PosteriorMvNormal) = d.dim

## Series of LowRank representation of the form Σ = Γ * Γ' ##
abstract type AbstractLowRankMvNormal{T} <:
              PosteriorMvNormal end

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

Distributions.var(d::AbstractLowRankMvNormal) = vec(sum(d.Γ .* d.Γ, dims = 2))
Distributions.entropy(d::AbstractLowRankMvNormal) = 0.5 * (log2π + logdet(cov(d) + 1e-5I))
rank(d::AbstractLowRankMvNormal) = size(d.Γ, 2)

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

struct BlockMFLowRankMvNormal{
    T,
    Ti<:AbstractVector{<:Int},
    Tμ<:AbstractVector{T},
    TΓ<:AbstractVector{<:AbstractMatrix{T}},
} <: AbstractLowRankMvNormal{T}
    dim::Int
    rank::Int
    id::Ti
    μ::Tμ
    Γ::TΓ
    function BlockMFLowRankMvNormal(
        μ::AbstractVector{T},
        indices::AbstractVector{<:Int},
        Γ::AbstractVector{<:AbstractMatrix{T}}
    ) where {T}
        rank = sum(x -> size(x, 2), Γ)
        return new{T,typeof(indices),typeof(μ),typeof(Γ)}(length(μ), rank, indices, μ, Γ)
    end
    function BlockMFLowRankMvNormal(
        dim::Int,
        rank::Int,
        indices::Ti,
        μ::Tμ,
        Γ::TΓ,
    ) where {T,Ti,Tμ<:AbstractVector{T},TΓ<:AbstractVector{<:AbstractMatrix{T}}}
        return new{T,Ti,Tμ,TΓ}(dim, rank, indices, μ, Γ)
    end
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::AbstractLowRankMvNormal{T},
  x::AbstractVector,
) where {T}
  nDim = length(x)
  nDim == d.dim || error("Wrong dimensions")
  x .= d.μ + BlockDiagonal(d.Γ) * randn(rng, T, rank(d))
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::AbstractLowRankMvNormal{T},
  x::AbstractMatrix,
) where {T}
  nDim, nPoints = size(x)
  nDim == d.dim || error("Wrong dimensions")
  x .= d.μ .+ BlockDiagonal(d.Γ) * randn(rng, T, nDim, rank(d))
end

rank(d::BlockMFLowRankMvNormal) = d.rank

Distributions.cov(d::BlockMFLowRankMvNormal) =
    BlockDiagonal(XXt.(d.Γ))

@functor BlockMFLowRankMvNormal

struct MFMvNormal{
    T,
    Tμ<:AbstractVector{T},
    TΓ<:AbstractVector{T},
} <: PosteriorMvNormal{T}
    dim::Int
    μ::Tμ
    Γ::TΓ
    function MFMvNormal(
        μ::AbstractVector{T},
        Γ::AbstractVector{T}
    ) where {T}
        return new{T,typeof(μ),typeof(Γ)}(length(μ), μ, Γ)
    end
    function MFMvNormal(
        dim::Int,
        μ::AbstractVector{T},
        Γ::AbstractVector{T}
    ) where {T, Tμ<:AbstractVector{T}, TΓ<:AbstractVector{T}}
        return new{T,Tμ,TΓ}(dim, μ, Γ)
    end
end

Distributions.cov(d::MFMvNormal) = Diagonal(abs2.(d.Γ))
@functor MFMvNormal

## Particle based distributions ##
abstract type AbstractSamplesMvNormal{T} <:
              PosteriorMvNormal end

function Distributions._rand!(
  rng::AbstractRNG,
  d::AbstractSamplesMvNormal,
  x::AbstractVector,
)
  nDim = length(x)
  nDim == d.dim || error("Wrong dimensions")
  x .= d.μ .+ (d.x .- d.μ)' * randn(rng, nDim) / nParticles(d)
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::AbstractSamplesMvNormal,
  x::AbstractMatrix,
)
  nDim, nPoints = size(x)
  nDim == d.dim || error("Wrong dimensions")
  x .= d.μ .+ (d.x .- d.μ)' * randn(rng, nDim, nPoints) / nParticles(d)
end
Base.length(d::AbstractSamplesMvNormal) = d.dim
nParticles(d::AbstractSamplesMvNormal) = d.n_particles
Distributions.mean(d::AbstractSamplesMvNormal) = d.μ
Distributions.var(d::AbstractSamplesMvNormal) = var(d.x, dims = 2)
Distributions.entropy(d::AbstractSamplesMvNormal) = 0.5 * (log2π + logdet(cov(d) + 1e-5I))

function update_q!(d::AbstractSamplesMvNormal)
    d.μ .= vec(mean(d.x, dims = 2))
    return nothing
end

"""
    SamplesMvNormal(x)

Create a sample based distribution.
"""
struct SamplesMvNormal{
    T,
    Tx<:AbstractMatrix{T},
    Tμ<:AbstractVector{T},
} <: AbstractSamplesMvNormal{T}
    dim::Int
    n_particles::Int
    x::Tx
    μ::Tμ
    function SamplesMvNormal(x::M) where {T,M<:AbstractMatrix{T}}
        μ = vec(mean(x, dims = 2))
        new{T,M,typeof(μ)}(size(x)..., x, μ)
    end
    function SamplesMvNormal(
        dim::Int,
        n_particles::Int,
        x::Tx,
        μ::Tμ,
    ) where {
        T,
        Tx<:AbstractMatrix{T},
        Tμ<:AbstractVector{T},
    }
        new{T,Tx,Tμ}(dim, n_particles, x, μ)
    end
end

Distributions.cov(d::SamplesMvNormal) = cov(d.x, dims = 2, )

@functor SamplesMvNormal

struct BlockMFSamplesMvNormal{
    T,
    Tx<:AbstractMatrix{T},
    Ti<:AbstractVector{<:Int},
    Tμ<:AbstractVector{T},
} <: AbstractSamplesMvNormal{T}
    dim::Int
    n_particles::Int
    K::Int
    id::Ti
    x::Tx
    μ::Tμ
    function BlockMFSamplesMvNormal(
        x::M,
        indices::AbstractVector{<:Int},
    ) where {T,M<:AbstractMatrix{T}}
        K = length(indices) - 1
        μ = vec(mean(x, dims = 2))
        return new{T,M,typeof(indices),typeof(μ)}(size(x)..., K, indices, x, μ)
    end
    function BlockMFSamplesMvNormal(
        dim::Int,
        n_particles::Int,
        K::Int,
        indices::Ti,
        x::Tx,
        μ::Tμ,
    ) where {T,Tx<:AbstractMatrix{T},Ti,Tμ<:AbstractVector{T}}
        return new{T,Tx,Ti,Tμ}(dim, n_particles, K, indices, x, μ)
    end
end

Distributions.cov(d::BlockSamplesMvNormal) =
    BlockDiagonal([cov(view(d.x, (d.id[i]+1):d.id[i+1], :), dims = 2) for i = 1:d.K])

@functor BlockMFSamplesMvNormal

struct MFSamplesMvNormal{
    T,
    Tx<:AbstractMatrix{T},
    Tμ<:AbstractVector{T},
} <: AbstractSamplesMvNormal{T}
    dim::Int
    n_particles::Int
    x::Tx
    μ::Tμ
    function MFSamplesMvNormal(
        x::M,
    ) where {T,M<:AbstractMatrix{T}}
        μ = vec(mean(x, dims = 2))
        return new{T,M,typeof(μ)}(size(x)..., x, μ)
    end
    function MFSamplesMvNormal(
        dim::Int,
        n_particles::Int,
        x::Tx,
        μ::Tμ,
    ) where {T,Tx<:AbstractMatrix{T},Ti,Tμ<:AbstractVector{T}}
        return new{T,Tx,Tμ}(dim, n_particles, x, μ)
    end
end

Distributions.cov(d::MFSamplesMvNormal) = Diagonal(var(d.x, dims = 2))
@functor MFSamplesMvNormal

const SampMvNormal = Union{
    MFSamplesMvNormal,
    BlockMFSamplesMvNormal,
    SamplesMvNormal,
    TransformedDistribution{<:AbstractSamplesMvNormal},
}