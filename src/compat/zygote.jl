struct ZygoteAD <: ADBackend end
ADBackend(::Val{:zygote}) = ZygoteAD
function setadbackend(::Val{:zygote})
    ADBACKEND[] = :zygote
end

import .Zygote

export ZygoteAD

function AdvancedVI.grad!(
    vo,
    alg::VariationalInference{<:AdvancedVI.ZygoteAD},
    q,
    model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...
)
    f(θ) = if (q isa Distribution)
        - vo(alg, update(q, θ), model, args...)
    else
        - vo(alg, q(θ), model, args...)
    end
    y, back = Zygote.pullback(f, θ)
    dy = first(back(1.0))
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, dy)
    return out
end
function grad!(
    vo,
    alg::GaussPFlow{<:AdvancedVI.ZygoteAD},
    q,
    logπ,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...
)
    function logjoint(x)
        sum(map(axes(x, 2)) do i
            # phi(logπ, q, x[:, i])
            phi(logπ, q, view(x, :, i))
        end
        )
    end
    val, back = Zygote.pullback(logjoint, q.dist.x)
    dy = first(back(1.0))
    DiffResults.value!(out, val)
    DiffResults.gradient!(out, dy)
    return out
end