using DataFrames
using CSV

data = CSV.read("goplayers.csv", DataFrame)

const P1=Int.(data.p11)
const P2=Int.(data.p22)

function loglik1(w1, w2)
    x = w1-w2
    log(1/(1 + exp(-x)))
end

init_w = rand(length(unique(vcat(P1, P2))))

function neg_log_lik(weights)
    -mapreduce(((p1, p2),)->loglik1(weights[p1], weights[p2]), +, zip(P1, P2))
end

using RCall

@rput P1 P2

# finishes in about 60s
@time R"""
w = runif(max(c(P1, P2)))
system.time(m <- optim(w, function(w) {
  x  = w[P1] - w[P2]
  p = 1/(1 + exp(-x))
  -sum(log(p))
}, method="BFGS"))

ping = m$par
""";

@rget ping

using Optim: optimize, BFGS
# take forever
@time opm = optimize(neg_log_lik, init_w, BFGS())
