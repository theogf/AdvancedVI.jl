using Turing
using AdvancedVI; const AVI = AdvancedVI
using KernelFunctions, Distances
using ForwardDiff
using LinearAlgebra
using Flux

x = randn(2000)

@model model(x) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0.0, sqrt(s))
    for i = 1:length(x)
        x[i] ~ Normal(m, sqrt(s))
    end
end
using Makie, Colors

##

max_iter = 100
m = model(x)
flowvi = AdvancedVI.PFlowVI(max_iter, true, true)
q = AVI.SamplesMvNormal(randn(100,2),[true,false])
q = AdvancedVI.vi(m, flowvi, 100, optimizer = ADAGrad(0.1))
@profiler q = AdvancedVI.vi(m, flowvi, 100, optimizer = ADAGrad(0.1))
# global q = AdvancedVI.vi(m, steinvi, q, optimizer = ADAGrad(0.1))
##

q = SamplesMvNormal(randn(100,2)*0.1,[true,false])
logπ = Turing.Variational.make_logjoint(m)
limits = FRect2D((.5,-.5),(1,1))
t = Node(1)
trajectories = [lift(t; init = [Point2f0(AVI.transform_particle(q,q.x[i,:]))]) do t
        push!(trajectories[i][], Point2f0(AVI.transform_particle(q,q.x[i,:])))
end  for i in 1:q.n_particles ]
# alpha = lift(t; init = [1.0]) do t
    # push!(alpha[],exp(-t))
# end
samples = lift(t; init = Point2f0.(AVI.transform_particle.([q],eachrow(q.x)))) do t
    samples = Point2f0.(AVI.transform_particle.([q],eachrow(q.x)))
end
scene = Scene(limits=limits)

colors = colormap("Reds",max_iter)
cc = lift(t->colors[1:t],t)
lines!.(trajectories,color=cc)
scatter!(samples, color=:red,markersize=0.01)
record(scene,joinpath(@__DIR__,"path_particles_pflow.gif"),framerate=10) do io
    function cb(q,i)
        t[] = i
        recordframe!(io)
    end
    global q = AdvancedVI.vi(logπ, flowvi, q, optimizer = ADAGrad(0.1), callback = cb)
end
