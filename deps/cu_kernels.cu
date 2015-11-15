#include <stdint.h>

template <typename T>
__device__ void colsumsq_pitch(T *data, size_t width, size_t height, size_t pitch, T *colsumsq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < width) {
        T v, s = 0.0;
        for (int i = 0; i < height; i += 1) {
            v = data[i*pitch + idx];
            s += v * v;
        }
        colsumsq[idx] = s;
    }
}

template <typename T>
__device__ void add_vecs(T *data, size_t width, size_t height, size_t pitch, T *sa, T *sb) {

    unsigned int idxx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idxy = blockIdx.y * blockDim.y + threadIdx.y;
    // Read once from global memory and store in block shared
    __shared__ T sa_shared[16];
    __shared__ T sb_shared[16]; // Will use block size (16,16)
    if (threadIdx.x == 0 && idxy < height )
        sa_shared[threadIdx.y] = sa[idxy];
    if (threadIdx.y == 0 && idxx < width)
        sb_shared[threadIdx.x] = sb[idxx];
    __syncthreads();
     if (idxx < width && idxy < height) {
        data[idxy*pitch + idxx] += sa_shared[threadIdx.y] + sb_shared[threadIdx.x];
    }
}


extern "C" {
    void __global__ colsumsq_double(double *data, size_t width, size_t height, size_t pitch, double *colsumsq) {
            colsumsq_pitch(data, width, height, pitch, colsumsq);
    }

    void __global__ colsumsq_float(float *data, size_t width, size_t height, size_t pitch, float *colsumsq) {
            colsumsq_pitch(data, width, height, pitch, colsumsq);
    }

    void __global__ add_vecs_float(float *data, size_t width, size_t height, size_t pitch, float *sa, float *sb) {
            add_vecs(data, width, height, pitch, sa, sb);
    }

     void __global__ add_vecs_double(double *data, size_t width, size_t height, size_t pitch, double *sa, double *sb) {
            add_vecs(data, width, height, pitch, sa, sb);
    }
}
