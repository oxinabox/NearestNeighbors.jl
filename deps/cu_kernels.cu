#include <stdint.h>

template <typename T>
__device__ void colsumsq_pitch(T *data, size_t width, size_t height, size_t pitch, T *norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < width) {
        T v, s = 0.0;
        for (int i = 0; i < height; i += 1) {
            v = data[i*pitch + idx];
            s += v * v;
        }
        norm[idx] = s;
    }
}


extern "C" {
    void __global__ colsumsq_double(double *data, size_t width, size_t height, size_t pitch, double *norm) {
            colsumsq_pitch(data, width, height, pitch, norm);
    }

    void __global__ colsumsq_float(float *data, size_t width, size_t height, size_t pitch, float *norm) {
            colsumsq_pitch(data, width, height, pitch, norm);
    }
}
