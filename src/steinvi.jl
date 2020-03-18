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
eltype(::SteinDistribution{T}) where {T} = T
Distributions.mean(d::SteinDistribution) = Statistics.mean(transform_particle.(Ref(d),eachrow(d.x)))
Distributions.cov(d::SteinDistribution) = Statistics.cov(transform_particle.(Ref(d),eachrow(d.x)))

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

function vi(model, alg::SteinVI, n_particles::Int ; optimizer = TruncatedADAGrad(), callback = nothing)
    vars = Turing.VarInfo(model).metadata
    nVars = 0
    domains = Bool[]
    for v in vars
        nVars += length(v.vals)
        lbs = getproperty.(support.(v.dists), :lb)
        domains = vcat(domains, lbs.!=-Inf)
    end
    q = SteinDistribution(randn(n_particles, nVars), domains)
    vi(model, alg, q; optimizer = optimizer, callback = callback)
end

function vi(model, alg::SteinVI, q::SteinDistribution; optimizer = TruncatedADAGrad(), callback = nothing)
    DEBUG && @debug "Optimizing SteinVI..."
    # Initial parameters for mean-field approx
    # Optimize
    optimize!(LogP(), alg, q, model, [0.0]; optimizer = optimizer, callback = callback)

    # Return updated `Distribution`
    return q
end

function optimize!(
    logp::LogP,
    alg::SteinVI,
    q::SteinDistribution,
    model,
    θ::AbstractVector{<:Real};
    optimizer = TruncatedADAGrad(),
    callback = nothing
)
    alg_name = alg_str(alg)
    max_iters = alg.max_iters
    logπ = make_logjoint(model)

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

        K = kernelmatrix(alg.kernel, q.x, obsdim = 1)
        # global gradK= reshape(
        #     ForwardDiff.jacobian(
        #             x -> kernelmatrix(alg.kernel, x, obsdim = 1),
        #             q.x),
        #         q.n_particles, q.n_particles, q.n_particles, q.dim)
        gradK_unshaped = ForwardDiff.jacobian(
                    x -> kernelmatrix(alg.kernel, x, obsdim = 1),
                    q.x)
        gradK = reshape(gradK_unshaped,q.n_particles, q.n_particles, q.n_particles, q.dim)
        gradlogp = ForwardDiff.gradient.(
                    z -> logπ(transform_particle(q,z)),
                    eachrow(q.x))
        #grad!(vo, alg, q, model, θ, diff_result)
        Δ = zeros(q.n_particles, q.dim)
        for k in 1:q.n_particles
            Δ[k,:] = sum(K[j, k] * gradlogp[j] + gradK[j, k, j, :]
                for j in 1:q.n_particles) / q.n_particles
        end
        # apply update rule
        # Δ = DiffResults.gradient(diff_result)
        global Δ = apply!(optimizer, q.x, Δ)
        @. q.x = q.x + Δ
        alg.kernel.transform.s .=
            log(q.n_particles) / sqrt( 0.5 * median(
            pairwise(SqEuclidean(), q.x, dims = 1)))

        if !isnothing(callback)
            callback(q,model,i)
        end
        # AdvancedVI.DEBUG && @debug "Step $i" Δ DiffResults.value(diff_result)
        # PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    return q
end

function (logp::LogP)(
    rng::AbstractRNG,
    alg::SteinVI,
    q::VariationalPosterior,
    model,
    θ
)
    vi = Turing.VarInfo(model)
    spl = Turing.SampleFromPrior()
    new_vi = Turing.VarINfo(vi, spl, θ)
    model(new_vi, spl)
    getlogp(new_vi)
end
