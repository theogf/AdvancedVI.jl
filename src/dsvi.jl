"""
    DSVI(n_particles = 100, max_iters = 1000)

Doubly Stochastic Variational Inference (DSVI) for a given model.
Can only work on the following distributions:
 - `CholMvNormal`
 - `MFMvNormal`
"""
struct DSVI{AD} <: VariationalInference{AD}
    max_iters::Int        # maximum number of gradient steps used in optimization
    nSamples::Int   # Number of samples per expectation
end

# params(alg::SteinVI) = nothing;#params(alg.kernel)

DSVI(args...) = DSVI{ADBackend()}(args...)
DSVI() = DSVI(100, 10)
nSamples(alg::DSVI) = alg.nSamples

alg_str(::DSVI) = "DSVI"

function vi(
    logπ::Function,
    alg::DSVI,
    q::AbstractPosteriorMvNormal;
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
    alg::DSVI,
    q::TransformedDistribution{<:AbstractPosteriorMvNormal};
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
    alg::DSVI{<:ForwardDiffAD},
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
    alg::DSVI,
    q::Bijectors.TransformedDistribution,
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

    z = zeros(length(q.dist), samples_per_step) # Storage for samples
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
        
        Distributions.randn!(z)
        reparametrize!(x, q.dist, z)

        grad!(vo, alg, q, _logπ, x, diff_result, samples_per_step)

        Δ = DiffResults.gradient(diff_result)
        
        Δμ .= apply!(optimizer[1], q.dist.μ, vec(mean(Δ, dims = 2)))
        ΔΓ .= typeof(q.dist.Γ)(apply!(optimizer[2], q.dist.Γ, updateΓ(Δ, z, q.dist.Γ)))
        # apply update rule
        q.dist.μ .-= Δμ
        q.dist.Γ .-= ΔΓ

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

function updateΓ(Δ, z, Γ::AbstractVector)
    vec(mean(Δ .* z, dims=2)) + inv.(Γ)
end

function updateΓ(Δ, z, Γ::LowerTriangular)
    LowerTriangular(Δ * z' / size(z, 2)) + inv(Diagonal(Γ))
end

function reparametrize!(x, q::CholMvNormal, z)
    x .= q.μ .+ q.Γ * z
end

function reparametrize!(x, q::MFMvNormal, z)
    x .= q.μ .+ q.Γ .* z
end