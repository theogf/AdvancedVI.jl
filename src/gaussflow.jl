using StatsFuns
using DistributionsAD
using Random: AbstractRNG, GLOBAL_RNG

"""
    GaussFlow(n_particles = 100, max_iters = 1000)

Gaussian Flow Inference (GaussFlow) for a given model.
Can only work on the following distributions:
 - `LowRankMvNormal`
 - `MFMvNormal`
 - `BlockMFMvNormal`
"""
struct GaussFlow{AD} <: VariationalInference{AD}
    max_iters::Int        # maximum number of gradient steps used in optimization
    nSamples::Int   # Number of samples per expectation
    precondΔ₁::Bool # Precondition the first gradient (mean)
    precondΔ₂::Bool # Precondition the second gradient (cov)
end

# params(alg::SteinVI) = nothing;#params(alg.kernel)

GaussFlow(args...) = GaussFlow{ADBackend()}(args...)
GaussFlow() = GaussFlow(100, 10, false, false)
nSamples(alg::GaussFlow) = alg.nSamples

alg_str(::GaussFlow) = "GaussFlow"

function vi(
    logπ::Function,
    alg::GaussFlow,
    q::PosteriorMvNormal;
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
    alg::GaussPFlow,
    q::TransformedDistribution{<:PosteriorMvNormal};
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
    alg::GaussFlow{<:ForwardDiffAD},
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
    alg::GaussFlow,
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

        Distributions.rand!(q, x) # Sample from q

        grad!(vo, alg, q, _logπ, x, diff_result, samples_per_step)

        Δ = DiffResults.gradient(diff_result)
        
        Δμ .= if alg.precondΔ₁
            q.dist.Γ * q.dist.Γ' * vec(mean(Δ, dims = 2))
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
    alg::GaussFlow,
)
    n_dim, n_samples = size(x)
    shift_x = x .- mean(q)
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

function compute_cov_part!(
    ΔΓ::AbstractMatrix,
    q::BlockMFLowRankMvNormal,
    x::AbstractMatrix,
    Δ::AbstractMatrix,
    alg::GaussFlow,
)
    n_dim, n_samples = size(x)
    shift_x = x .- mean(q)
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

function compute_cov_part!(
    ΔΓ::AbstractMatrix,
    q::MFMvNormal,
    x::AbstractMatrix,
    Δ::AbstractMatrix,
    alg::GaussFlow,
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