using StatsFuns
using DistributionsAD
using KernelFunctions
using Random: AbstractRNG, GLOBAL_RNG


struct SteinDistribution{T,M<:AbstractMatrix{T}} <: Distributions.ContinuousMultivariateDistribution
    dim::Int
    n_particles::Int
    x::M
end

length(d::SteinDistribution) = d.dim
eltype(::SteinDistribution{T}) where {T} = T
mean(d::SteinDistribution) = mean(d.x, dims=2)
cov(d::SteinDistribution) = cov(d.x, dims=2)

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

function vi(model, alg::SteinVI, q::SteinDistribution; optimizer = TruncatedADAGrad())
    DEBUG && @debug "Optimizing SteinVI..."
    # Initial parameters for mean-field approx
    θ = [q.x]#params(alg)

    # Optimize
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    # Return updated `Distribution`
    return q
end

function optimize!(
    elbo::ELBO,
    alg::SteinVI,
    q::SteinDistribution,
    model,
    θ;
    optimizer = TruncatedADAGrad(),
)
    alg_name = alg_str(alg)
    max_iters = alg.max_iters

end
function vi(model, alg::SteinVI, nParticles, kernel; optimizer = TruncatedADAGrad())
    q = SteinDistribution(nParticles, kernel)
    DEBUG && @debug "Optimizing SteinVI..."
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


function optimize(elbo::ELBO, alg::SteinVI, q, model, θ_init; optimizer = TruncatedADAGrad())
    θ = copy(θ_init)

    # `model` assumed to be callable z ↦ p(x, z)
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    return θ
end

function (elbo::ELBO)(
    rng::AbstractRNG,
    alg::SteinVI,
    q::VariationalPosterior,
    logπ,
    num_samples
)
    #   𝔼_q(z)[log p(xᵢ, z)]
    # = ∫ log p(xᵢ, z) q(z) dz
    # = ∫ log p(xᵢ, f(ϕ)) q(f(ϕ)) |det J_f(ϕ)| dϕ   (since change of variables)
    # = ∫ log p(xᵢ, f(ϕ)) q̃(ϕ) dϕ                   (since q(f(ϕ)) |det J_f(ϕ)| = q̃(ϕ))
    # = 𝔼_q̃(ϕ)[log p(xᵢ, z)]

    #   𝔼_q(z)[log q(z)]
    # = ∫ q(f(ϕ)) log (q(f(ϕ))) |det J_f(ϕ)| dϕ     (since q(f(ϕ)) |det J_f(ϕ)| = q̃(ϕ))
    # = 𝔼_q̃(ϕ) [log q(f(ϕ))]
    # = 𝔼_q̃(ϕ) [log q̃(ϕ) - log |det J_f(ϕ)|]
    # = 𝔼_q̃(ϕ) [log q̃(ϕ)] - 𝔼_q̃(ϕ) [log |det J_f(ϕ)|]
    # = - ℍ(q̃(ϕ)) - 𝔼_q̃(ϕ) [log |det J_f(ϕ)|]

    # Finally, the ELBO is given by
    # ELBO = 𝔼_q(z)[log p(xᵢ, z)] - 𝔼_q(z)[log q(z)]
    #      = 𝔼_q̃(ϕ)[log p(xᵢ, z)] + 𝔼_q̃(ϕ) [log |det J_f(ϕ)|] + ℍ(q̃(ϕ))

    # If f: supp(p(z | x)) → ℝ then
    # ELBO = 𝔼[log p(x, z) - log q(z)]
    #      = 𝔼[log p(x, f⁻¹(z̃)) + logabsdet(J(f⁻¹(z̃)))] + ℍ(q̃(z̃))
    #      = 𝔼[log p(x, z) - logabsdetjac(J(f(z)))] + ℍ(q̃(z̃))

    # But our `forward(q)` is using f⁻¹: ℝ → supp(p(z | x)) going forward → `+ logjac`
    _, z, logjac, _ = forward(rng, q)
    res = (logπ(z) + logjac) / num_samples

    res += (q isa TransformedDistribution) ? entropy(q.dist) : entropy(q)

    for i = 2:num_samples
        _, z, logjac, _ = forward(rng, q)
        res += (logπ(z) + logjac) / num_samples
    end

    return res
end
