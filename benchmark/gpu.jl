using Distances

include("../src/gpu/brute.jl")

# Data
for T in [Float32, Float64]
    println("Type: $T")
    Nq = 10000;
    Nr = 10000;
    dim = 50;
    data = rand(T, Nr, dim);
    datap = data';
    # query
    query = rand(T, Nq, dim);
    queryp = query';

    m = size(datap, 2)
    n = size(queryp, 2)
    r = Array(T, (m, n))

    # Precompile

    pairwise(SqEuclidean(), rand(3,3), rand(3,3));
    pairwise2!(zeros(3,3), SqEuclidean(), rand(3,3), rand(3,3));
    pairwise_GPU(rand(3,3), rand(3,3));
    pairwise_GPU2(rand(3,3), rand(3,3));

    println("CPU")
    @time pw_cpu = pairwise(SqEuclidean(), datap, queryp);
    println("CPU + ger!")
    @time pw_cpu = pairwise2!(r, SqEuclidean(), datap, queryp);
    println("GPU + custom kernel")
    @time pw_gpu = pairwise_GPU(data, query);
    println("GPU + ger!")
    @time pw_gpu2 = pairwise_GPU2(data, query);
    #@test pw_cpu ≈ pw_gpu
end

function pairwise2!{T <: Union{Float32, Float64}}(r::AbstractMatrix{T}, dist::SqEuclidean, a::AbstractMatrix{T}, b::AbstractMatrix{T})
    BLAS.gemm!('T', 'N', T(-2.0), a, b, zero(T), r)
    sa2 = sumabs2(a, 1)
    onea = ones(eltype(sa2), size(sa2))

    sb2 = sumabs2(b, 1)
    oneb = ones(eltype(sb2), size(sb2))

    BLAS.ger!(one(T), vec(sa2), vec(oneb), r)
    BLAS.ger!(one(T), vec(onea), vec(sb2), r)
    r
end
