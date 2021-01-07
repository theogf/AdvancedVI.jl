"""
    SVGD(n_particles = 100, max_iters = 1000)

Stein Variational Inference (SVGD) for a given model.
"""
struct SVGD{AD} <: VariationalInference{AD}
    max_iters::Int        # maximum number of gradient steps used in optimization
    kernel::Kernel
end

SVGD(args...) = SVGD{ADBackend()}(args...)
SVGD() = SVGD(100, SqExponentialKernel())

alg_str(::SVGD) = "SVGD"

vi(
    logπ::Function,
    alg::SVGD,
    q::EmpiricalDistribution;
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing,
) = vi(
    logπ,
    alg,
    transformed(q, Identity{1}()),
    optimizer = optimizer,
    callback = callback,
    hyperparams = hyperparams,
    hp_optimizer = hp_optimizer,
)

function vi(
    logπ::Function,
    alg::SVGD,
    q::TransformedDistribution{<:EmpiricalDistribution};
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
        hyperparams = hyperparams,
        hp_optimizer = hp_optimizer,
    )
    # Return updated `Distribution`
    return q
end

function _logπ(logπ, x, tr)
    z, logdet = forward(tr, x)
    return logπ(z) + logdet
end


function optimize!(
    vo,
    alg::SVGD,
    q::TransformedDistribution{<:EmpiricalDistribution},
    logπ;
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing
)
    alg_name = alg_str(alg)
    max_iters = alg.max_iters

    # diff_result = DiffResults.GradientResult(θ)
    alg.kernel.transform.s .= log(q.dist.n_particles) / sqrt(median(
    pairwise(SqEuclidean(), q.dist.x, dims = 2)))

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end

    time_elapsed = @elapsed while (i < max_iters) # & converged

        logπbase = if !isnothing(hyperparams)
            logπ(hyperparams)
        else
            logπ
        end

        Δ = similar(q.dist.x) #Preallocate gradient
        K = kernelmatrix(alg.kernel, q.dist.x, obsdim = 2) #Compute kernel matrix
        gradlogp = ForwardDiff.gradient.(
            x -> eval_logπ(logπbase, q, x),
            eachcol(q.dist.x))
        # Option 1 : Preallocate
        # global gradK = reshape(
        #     ForwardDiff.jacobian(
        #             x -> kernelmatrix(alg.kernel, x, obsdim = 2),
        #             q.dist.x),
        #         q.dist.n_particles, q.dist.n_particles, q.dist.n_particles, q.dist.dim)
        # #grad!(vo, alg, q, model, θ, diff_result)
        # for k in 1:q.dist.n_particles
        #     Δ[:, k] = sum(K[j, k] * gradlogp[j] + gradK[j, k, j, :]
        #         for j in 1:q.dist.n_particles) / q.dist.n_particles
        # end
        # Option 2 : On time computations
        for k = 1:q.dist.n_particles
            Δ[:, k] =
                mean(
                    K[j, k] * gradlogp[j] + ForwardDiff.gradient(
                        x -> alg.kernel(x, q.dist.x[:, k]),
                        q.dist.x[:, j],
                    ) for j in 1:q.dist.n_particles
                )
        end


        # apply update rule
        # Δ = DiffResults.gradient(diff_result)
        Δ = apply!(optimizer, q.dist.x, Δ)
        @. q.dist.x = q.dist.x + Δ
        # alg.kernel.transform.s .=
        #     log(q.dist.n_particles) / sqrt(median(
        #     pairwise(SqEuclidean(), q.dist.x, dims = 2)))

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

function (elbo::ELBO)(
    rng::AbstractRNG,
    alg::SVGD,
    q::TransformedDistribution{<:EmpiricalDistribution},
    logπ::Function
)

    res = sum(mapslices(x -> eval_logπ(logπ, q, x), q.dist.x, dims = 1))

    if q isa TransformedDistribution
        res += entropy(q.dist)
    else
        res += entropy(q)
    end
    return res
end
