using Turing
using Distributions, DistributionsAD
using AdvancedVI; const AVI = AdvancedVI
using Makie, StatsMakie, Colors, MakieLayout
using KernelFunctions, Flux, KernelDensity

mu1 = -2.0;
mu2 = 2.0
mu_init = -10.0
sig_init = 1.0
d1 = Normal(mu1)
d2 = Normal(mu2)
d = MixtureModel([d1, d2],[1/3, 2/3])

x = rand(d,1000)
xrange = range(min(mu_init,mu1)-3.0,max(mu_init,mu2)+3.0,length=300)
scene, layout = layoutscene()
ax = layout[1, 1] = LAxis(scene)
plot!(ax,histogram(nbins=100,normalize=true), x, color=RGBA(0.0,0.0,0.0,0.5))
plot!(ax,xrange,pdf.(Ref(d),xrange),color=:red, linewidth=7.0)


@model model(x) = begin
    x ~ 0.3 * Normal(0.0) + 0.7 * Normal(3.0)
end

m = model(x)
Turing.VarInfo(model).metadata

nParticles = 100
max_iters = 2


##
# advi = Turing.Variational.ADVI(nParticles, max_iters)
advi = AVI.ADVI(nParticles, max_iters)
θ_init = [mu_init,sig_init]
adq = AVI.transformed(TuringDiagMvNormal([mu_init],[sig_init]),AVI.Bijectors.Identity{1}())



steinvi = AVI.SteinVI{AVI.TrackerAD}(max_iters, transform(SqExponentialKernel(), 1.0))
steinq = AVI.SteinDistribution(rand(Normal(mu_init, sqrt(sig_init)),nParticles,1))

gaussvi = AVI.PFlowVI(max_iters, false, false)
gaussq = AVI.transformed(SamplesMvNormal(rand(Normal(mu_init, sqrt(sig_init)),1,nParticles)),AVI.Bijectors.Identity{1}())

# logπ = Turing.Variational.make_logjoint(m)
setadbackend(:reverse_diff)
logπ_base(x) = log(1/3*pdf(d1,first(x)) + 2/3*pdf(d2,first(x)))

α = 1.0
optad = ADAGrad(α)
# optad = Flux.Optimiser(InvDecay(0.1),ADAGrad(α))
optstein = ADAGrad(α)
optgauss = ADAGrad(α)
t = Node(0)
pdfad = lift(t) do t
    adqn = Normal((AVI.params(adq.dist) |> x -> (first(x[1]),sqrt(first(x[2]))))...)
    pdfad = pdf.(Ref(adqn),xrange)
end
pdfstein = lift(t) do t
    steinqn = kde(steinq.x[:])
    pdfstein = pdf.(Ref(steinqn),xrange)
end
pdfgauss = lift(t) do t
    gaussqn = Normal(gaussq.dist.μ[1],sqrt(gaussq.dist.Σ[1,1]))
    pdfgauss = pdf.(Ref(gaussqn),xrange)
end

ls = [plot!(ax,xrange,pdf.(Ref(d),xrange),color=:red, linewidth=7.0),
    plot!(ax,xrange, pdfad, linewidth = 3.0, color = :green),
    plot!(ax,xrange, pdfstein, linewidth = 3.0, color = :blue),
    # plot!(ax,xrange, pdfgauss, linewidth = 3.0, color = :purple)
    ]
leg = layout[1,1] = LLegend(scene,ls,
                    ["p(x)",
                    "ADVI",
                    "Stein VI",
                    # "Gaussian Particles"
                    ],
                    width=Auto(false),height=Auto(false),
                    halign = :left, valign = :top, margin = (10,10,10,10))
##
record(scene,joinpath(@__DIR__,"path_particles_ad_vs_steinflow.gif"),framerate=25) do io
    for i in 1:500
        global adq = AVI.vi(logπ_base, advi, adq, θ_init, optimizer = optad)
        # global adq = Turing.Variational.vi(logπ_base, advi, adq, optimizer = optad)
        AVI.vi(logπ_base, steinvi, steinq, optimizer = optstein)
        # AVI.vi(logπ_base, gaussvi, gaussq, optimizer = optgauss)
        t[] = i
        recordframe!(io)
    end
end
optad
optgauss
optstein