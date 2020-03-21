using StatsFuns
using DistributionsAD
using KernelFunctions
using Random: AbstractRNG, GLOBAL_RNG

struct SteinDistribution{T,M<:AbstractMatrix{T}} <: Distributions.ContinuousMultivariateDistribution
    dim::Int
    n_particles::Int
    x::M # Dimensions are nDim x nParticles
    function SteinDistribution(x::M) where {T, M<: AbstractMatrix{T}}
        new{T,M}(size(x)..., x)
    end
end

Base.length(d::SteinDistribution) = d.dim

# Random._rand!(d::SteinDistribution, v::AbstractVector) = d.x
Base.eltype(::SteinDistribution{T}) where {T} = T
function Distributions._rand!(rng::AbstractRNG, d::SteinDistribution, x::AbstractVector)
    nDim = length(x)
    @assert nDim == d.dim "Wrong dimensions"
    x .= d.x[:,rand(rng, 1:d.n_particles)]
end
function Distributions._rand!(rng::AbstractRNG, d::SteinDistribution, x::AbstractMatrix)
    nDim, nPoints = size(x)
    @assert nDim == d.dim "Wrong dimensions"
    x .= d.x[:,rand(rng, 1:d.n_particles, nPoints)]
end
Distributions.mean(d::SteinDistribution) = Statistics.mean(eachcol(d.x))
Distributions.cov(d::SteinDistribution) = Statistics.cov(eachcol(d.x))
Distributions.var(d::SteinDistribution) = Statistics.var(eachcol(d.x))
"""
    SteinVI(n_particles = 100, max_iters = 1000)

Stein Variational Inference (SteinVI) for a given model.
"""
struct SteinVI{AD} <: VariationalInference{AD}
    max_iters::Int        # maximum number of gradient steps used in optimization
    kernel::Kernel
end

# params(alg::SteinVI) = nothing;#params(alg.kernel)

SteinVI(args...) = SteinVI{ADBackend()}(args...)
SteinVI() = SteinVI(100, SqExponentialKernel())

alg_str(::SteinVI) = "SteinVI"

function vi(model::Turing.Model, alg::SteinVI, n_particles::Int ; optimizer = TruncatedADAGrad(), callback = nothing)
    logπ = make_logjoint(model)
    q = transformed(SteinDistribution(randn(n_particles, nVars)), bijector(model))
    vi(logπ, alg, q; optimizer = optimizer, callback = callback)
end

function vi(
    logπ::Function,
    alg::SteinVI,
    q::SteinDistribution;
    optimizer = TruncatedADAGrad(),
    callback = nothing,
) = vi(logπ, alg, transformed(q, Identity{length(q)}()), optimizer = optimizer, callback = callback)

function vi(
    logπ::Function,
    alg::SteinVI,
    q::TransformedDistribution{<:SteinDistribution};
    optimizer = TruncatedADAGrad(),
    callback = nothing,
)
    DEBUG && @debug "Optimizing SteinVI..."
    # Initial parameters for mean-field approx
    # Optimize
    optimize!(alg, q, logπ; optimizer = optimizer, callback = callback)

    # Return updated `Distribution`
    return q
end



function optimize!(
    alg::SteinVI,
    q::Transformed{<:SteinDistribution},
    logπ,
    θ::AbstractVector{<:Real};
    optimizer = TruncatedADAGrad(),
    callback = nothing
)
    alg_name = alg_str(alg)
    max_iters = alg.max_iters

    # diff_result = DiffResults.GradientResult(θ)
    alg.kernel.transform.s .= log(q.dist.n_particles) / sqrt( 0.5 * median(
    pairwise(SqEuclidean(), q.dist.x, dims = 1)))

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end

    time_elapsed = @elapsed while (i < max_iters) # & converged

        Δ = similar(q.dist.x) #Preallocate gradient
        K = kernelmatrix(alg.kernel, q.dist.x, obsdim = 2) #Compute kernel matrix
        gradlogp = ForwardDiff.gradient.(
        x -> forward(q.transform,x) |> (z,logdet)->logπ(z)+logdet,
        eachcol(q.dist.x))
        # Option 1 : Preallocate
        gradK = reshape(
            ForwardDiff.jacobian(
                    x -> kernelmatrix(alg.kernel, x, obsdim = 2),
                    q.dist.x),
                q.dist.n_particles, q.dist.n_particles, q.dist.n_particles, q.dist.dim)
        #grad!(vo, alg, q, model, θ, diff_result)
        for k in 1:q.dist.n_particles
            Δ[k,:] = sum(K[j, k] * gradlogp[j] + gradK[j, k, j, :]
                for j in 1:q.dist.n_particles) / q.dist.n_particles
        end
        # Option 2 : On time computations
        # for k in 1:q.n_particles
        #     Δ[k,:] = sum(
        #         K[j, k] * gradlogp[j] +
        #         ForwardDiff.gradient(x->KernelFunctions.kappa(alg.kernel,q.x[j,:],x), q.x[k,:])
        #         for j in 1:q.n_particles) / q.n_particles
        # end


        # apply update rule
        # Δ = DiffResults.gradient(diff_result)
        Δ = apply!(optimizer, q.dist.x, Δ)
        @. q.dist.x = q.dist.x + Δ
        alg.kernel.transform.s .=
            log(q.dist.n_particles) / sqrt( 0.5 * median(
            pairwise(SqEuclidean(), q.dist.x, dims = 2)))

        if !isnothing(callback)
            callback(q,i)
        end
        AdvancedVI.DEBUG && @debug "Step $i" Δ
        PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    return q
end
