using Turing
using AdvancedVI
using KernelFunctions

x = randn(2000)

@model model(x) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0.0, sqrt(s))
    for i = 1:length(x)
        x[i] ~ Normal(m, sqrt(s))
    end
end

m = model(x)

steinvi = AdvancedVI.SteinVI(1000, SqExponentialKernel())
q = AdvancedVI.SteinDistribution(1, 100, rand(1,100))
q = AdvancedVI.vi(m, steinvi, q)
