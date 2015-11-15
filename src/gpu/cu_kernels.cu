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


// Copyright (c) 2014: Tim Holy.
template <typename T>
__device__ void fill_contiguous(T *data, size_t len, T val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < len; i += gridDim.x * blockDim.x) {
        data[i] = val;
    }
}


extern "C" {
    void __global__ fill_contiguous_double(double *data, size_t len, double val) {fill_contiguous(data, len, val);}
    void __global__ fill_contiguous_float(float *data, size_t len, float val)  {fill_contiguous(data, len, val);}

    void __global__ colsumsq_double(double *data, size_t width, size_t height, size_t pitch, double *norm) {
            colsumsq_pitch(data, width, height, pitch, norm);
    }

    void __global__ colsumsq_float(float *data, size_t width, size_t height, size_t pitch, float *norm) {
            colsumsq_pitch(data, width, height, pitch, norm);
    }
}
