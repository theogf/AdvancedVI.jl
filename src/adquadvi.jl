"""
$(TYPEDEF)

Automatic Differentiation Variational Inference (ADVI) with automatic differentiation
backend `AD`.

# Fields

$(TYPEDFIELDS)
"""
struct ADQuadVI{AD, T} <: VariationalInference{AD}
    "Number of points used to estimate the ELBO expectation in each optimization step."
    nPoints::Int
    nodes::Vector{T}
    weights::Vector{T}
    "Maximum number of gradient steps."
    max_iters::Int
end

function ADQuadVI(nPoints::Int=100, max_iters::Int=1000)
    return ADQuadVI{ADBackend()}(nPoints, gausshermite(nPoints)..., max_iters)
end

alg_str(::ADQuadVI) = "ADQuadVI"

nSamples(alg::ADQuadVI) = alg.nPoints

function vi(model, alg::ADQuadVI, q, θ_init; optimizer = TruncatedADAGrad())
    θ = copy(θ_init)
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    # If `q` is a mean-field approx we use the specialized `update` function
    if q isa Distribution
        return update(q, θ)
    else
        # Otherwise we assume it's a mapping θ → q
        return q(θ)
    end
end


function optimize(elbo::ELBO, alg::ADQuadVI, q, model, θ_init; optimizer = TruncatedADAGrad())
    θ = copy(θ_init)

    # `model` assumed to be callable z ↦ p(x, z)
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    return θ
end

# WITHOUT updating parameters inside ELBO
function (elbo::ELBO)(
    rng::AbstractRNG,
    alg::ADQuadVI,
    q::VariationalPosterior,
    logπ::Function,
    num_samples
)
    @assert length(mean(q.dist)) == 1
    μ, σ² = first(mean(q.dist)), first(cov(q.dist))
    xs = alg.nodes*sqrt(σ²) .+ μ
    res = sum((x,w) -> _eval_logπ(q.transform, x)*w for (x,w) in zip(xs, alg.weights))

    if q isa TransformedDistribution
        res += entropy(q.dist)
    else
        res += entropy(q)
    end

    return res
end

function _eval_logπ(t, x)
    z, logjac = forward(t, x)
    (logπ(z) + logjac)
end
