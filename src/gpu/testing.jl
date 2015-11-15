using CUDArt

using Distances
using Base.Test

include("cublas.jl")
include("NNCuda.jl")

function pairwise_GPU{T}(data::Matrix{T}, query::AbstractArray{T})
    result = devices(dev->true) do devlist
        NNCuda.init(devlist) do dev
            # Move data + query to device
            d_data = CudaPitchedArray(data)
            d_query = CudaPitchedArray(query)

            # TODO, split into smaller chunks if data is too big

            # Allocate on device
            d_RtQ = CudaPitchedArray(T, (size(data,1), size(query,1)))
            d_NR = CudaArray(T, size(data,1))
            d_ones_R = CudaArray(T, size(data,1))
            fill!(d_ones_R, T(1.0))
            d_NQ = CudaArray(T, size(query,1))
            d_ones_Q = CudaArray(T, size(query,1))
            fill!(d_ones_Q, T(1.0))

            gemm!('N','T', T(-2.0), d_data, d_query, zero(T), d_RtQ)
            NNCuda.colsumsq!(d_NR, d_data)
            NNCuda.colsumsq!(d_NQ, d_query)
            ger!(one(T), d_NR, d_ones_Q, d_RtQ)
            ger!(one(T), d_ones_R, d_NQ, d_RtQ)
            to_host(d_RtQ)
        end
    end
end
