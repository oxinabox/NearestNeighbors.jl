include("../src/gpu/testing.jl")
# Data
Nq = 10000;
Nr = 10000;
dim = 80;
data = rand(Float32, Nr, dim);
datap = data';
# query
query = rand(Float32, Nq, dim);
queryp = query';

# Precompile
pairwise(SqEuclidean(), rand(3,3), rand(3,3));
pairwise_GPU(rand(3,3), rand(3,3));

@time pw_cpu = pairwise(SqEuclidean(), datap, queryp);
@time pw_gpu = pairwise_GPU(data, query);
@test pw_cpu ≈ pw_gpu
