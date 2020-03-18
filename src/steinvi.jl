using StatsFuns
using DistributionsAD
using KernelFunctions
using Random: AbstractRNG, GLOBAL_RNG

struct SteinDistribution{T,M<:AbstractMatrix{T}} <: Distributions.ContinuousMultivariateDistribution
    n_particles::Int
    dim::Int
    x::M
    transforms::BitArray
    function SteinDistribution(x::M, domains=falses(size(x,2))) where {T, M<: AbstractMatrix{T}}
        new{T,M}(size(x)..., x, domains)
    end
end

transform_particle(d::SteinDistribution, x::AbstractVector) =
    ifelse.(d.transforms,softplus.(x),x)

Base.length(d::SteinDistribution) = d.dim

# Random._rand!(d::SteinDistribution, v::AbstractVector) = d.x
Base.eltype(::SteinDistribution{T}) where {T} = T
function Distributions._rand!(rng::AbstractRNG, d::SteinDistribution, x::AbstractVector)
    nDim = length(x)
    @assert nDim == d.dim "Wrong dimensions"
    x .= d.x[rand(rng, 1:d.n_particles),:]
end
function Distributions._rand!(rng::AbstractRNG, d::SteinDistribution, x::AbstractMatrix)
    nDim, nPoints = size(x)
    @assert nDim == d.dim "Wrong dimensions"
    x .= d.x[rand(rng, 1:d.n_particles, nPoints),:]'
end
Distributions.mean(d::SteinDistribution) = Statistics.mean(transform_particle.(Ref(d),eachrow(d.x)))
Distributions.cov(d::SteinDistribution) = Statistics.cov(transform_particle.(Ref(d),eachrow(d.x)))
Distributions.var(d::SteinDistribution) = Statistics.var(transform_particle.(Ref(d),eachrow(d.x)))
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
    vars = Turing.VarInfo(model).metadata
    nVars = 0
    domains = Bool[]
    for v in vars
        nVars += length(v.vals)
        lbs = getproperty.(support.(v.dists), :lb)
        domains = vcat(domains, lbs.!=-Inf)
    end
    q = SteinDistribution(randn(n_particles, nVars), domains)
    logπ = make_logjoint(model)
    vi(logπ, alg, q; optimizer = optimizer, callback = callback)
end

function vi(logπ::Function, alg::SteinVI, q::SteinDistribution; optimizer = TruncatedADAGrad(), callback = nothing)
    DEBUG && @debug "Optimizing SteinVI..."
    # Initial parameters for mean-field approx
    # Optimize
    optimize!(alg, q, logπ, [0.0]; optimizer = optimizer, callback = callback)

    # Return updated `Distribution`
    return q
end

function optimize!(
    alg::SteinVI,
    q::SteinDistribution,
    logπ,
    θ::AbstractVector{<:Real};
    optimizer = TruncatedADAGrad(),
    callback = nothing
)
    alg_name = alg_str(alg)
    max_iters = alg.max_iters

    # diff_result = DiffResults.GradientResult(θ)
    alg.kernel.transform.s .= log(q.n_particles) / sqrt( 0.5 * median(
    pairwise(SqEuclidean(), q.x, dims = 1)))

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end

    time_elapsed = @elapsed while (i < max_iters) # & converged

        Δ = zeros(q.n_particles, q.dim) #Preallocate gradient
        K = kernelmatrix(alg.kernel, q.x, obsdim = 1) #Compute kernel matrix
        gradlogp = ForwardDiff.gradient.(
        z -> logπ(transform_particle(q,z)),
        eachrow(q.x))
        # Option 1 : Preallocate
        gradK = reshape(
            ForwardDiff.jacobian(
                    x -> kernelmatrix(alg.kernel, x, obsdim = 1),
                    q.x),
                q.n_particles, q.n_particles, q.n_particles, q.dim)
        #grad!(vo, alg, q, model, θ, diff_result)
        for k in 1:q.n_particles
            Δ[k,:] = sum(K[j, k] * gradlogp[j] + gradK[j, k, j, :]
                for j in 1:q.n_particles) / q.n_particles
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
        Δ = apply!(optimizer, q.x, Δ)
        @. q.x = q.x + Δ
        alg.kernel.transform.s .=
            log(q.n_particles) / sqrt( 0.5 * median(
            pairwise(SqEuclidean(), q.x, dims = 1)))

        if !isnothing(callback)
            callback(q,model,i)
        end
        AdvancedVI.DEBUG && @debug "Step $i" Δ
        PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    return q
end
