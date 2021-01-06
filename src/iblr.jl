"""
    IBLR(n_particles = 100, max_iters = 1000)

iBayes Learning Rule (IBLR) for a given model.
Can only work on the following distributions:
 - `Precision`
 - `MFMvNormal`
"""
struct IBLR{AD} <: VariationalInference{AD}
    max_iters::Int        # maximum number of gradient steps used in optimization
    nSamples::Int   # Number of samples per expectation
    hess_comp::Symbol
end

# params(alg::SteinVI) = nothing;#params(alg.kernel)

IBLR(args...) = IBLR{ADBackend()}(args...)
IBLR() = IBLR(100, 10, :hess)
nSamples(alg::IBLR) = alg.nSamples

alg_str(::IBLR) = "IBLR"

function vi(
    logπ::Function,
    alg::IBLR,
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
    alg::IBLR,
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
    alg::IBLR{<:ForwardDiffAD},
    q,
    logπ,
    x,
    out::DiffResults.MutableDiffResult,
    args...,
)
    f(x) = sum(z->phi(logπ, q, z), eachcol(x))
    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(x), chunk_size))
    config = ForwardDiff.GradientConfig(f, x, chunk)
    ForwardDiff.gradient!(out, f, x, config)
end

function hessian!( # Does not work currently... 
    vo,
    alg::IBLR{<:ForwardDiffAD},
    q,
    logπ,
    x,
    out::AbstractVector{<:DiffResults.MutableDiffResult},
    args...,
)
    f(x) = phi(logπ, q, x)
    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(x[:, 1]), chunk_size))
    config = ForwardDiff.HessianConfig.(f, eachcol(x), Ref(chunk))
    ForwardDiff.hessian!.(out, f, eachcol(x), config)
end

function optimize!(
    vo,
    alg::IBLR,
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

    z = zeros(dim(q.dist), samples_per_step) # Storage for samples
    x = zeros(dim(q.dist), samples_per_step) # Storage for samples
    diff_result = DiffResults.GradientResult(x)
    # hess_results = DiffResults.HessianResult.(eachcol(x)) 

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end
    Δμ = similar(q.dist.μ)
    G = similar(q.dist.S)
    gS = similar(q.dist.S)
    # optimizer isa Descent || error("IBLR only work with std. grad. descent")
    Δt = optimizer.eta
    
    time_elapsed = @elapsed while (i < max_iters) # & converged


        _logπ = if !isnothing(hyperparams)
            logπ(hyperparams)
        else
            logπ
        end
        
        Distributions.randn!(z)

        reparametrize!(x, q.dist, z)

        grad!(vo, alg, q, _logπ, x, diff_result, samples_per_step)
        # hessian!(vo, alg, q, _logπ, x, hess_results, samples_per_step)
        Δ = DiffResults.gradient(diff_result)
        

        update_dist!(q.dist, alg, Δ, x, z, Δt)
        
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

function update_dist!(d::PrecisionMvNormal, alg::IBLR, Δ, Δμ, G, gS, x, z, Δt)
    
    if alg.comp_hess == :hess
        gS .= mean(ForwardDiff.hessian.(z->phi(_logπ, q, z), eachcol(x)))
    elseif alg.comp_hess == :rep
        gS .= z * gμ'
        gS .= 0.5 * (gS + gS')
    end

    G .= d.S - gS
    Δμ .= d.S \ mean(Δ, dims=2)
    q.dist.μ .-= Δt * Δμ
    q.dist.S .= (1 - Δt) * q.dist.S + Δt * gS + 0.5 * Δt^2 * G * (q.dist.S \ G)
end

function update_dist!(d::MFMvNormal, alg::IBLR, Δ, Δμ, G, gS, x, z, Δt)
    
    if alg.comp_hess == :hess
        gS .= mean(diag.(ForwardDiff.hessian.(z->phi(_logπ, q, z), eachcol(x))))
    elseif alg.comp_hess == :rep
        gS .= S * (x - d.μ) * Δ'
        gS .= 0.5 * (gS + gS')
    end
    
    G .= d.S - gS
    Δμ .= d.S \ mean(Δ, dims=2)
    d.μ .-= Δt * Δμ
    d.S .= (1 - Δt) * d.S + Δt * gS + 0.5 * Δt^2 * G * (q.dist.S \ G)
end