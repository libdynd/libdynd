//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CUDA_CONFIG_HPP_
#define _DYND__CUDA_CONFIG_HPP_

#include <assert.h>
#include <cmath>

#ifdef DYND_CUDA
#include <cuda_runtime.h>
#endif // DYND_CUDA

#ifdef __CUDACC__ // We are compiling with NVIDIA's nvcc

#ifdef __CUDA_ARCH__ // We are compiling for the device
#define DYND_CUDA_DEVICE_ARCH
#else // We are compiling for the host
#define DYND_CUDA_HOST_ARCH
#endif // __CUDA_ARCH__

#define DYND_CUDA_DEVICE __device__ // A variable that resides on, or a function that is compiled for, the device
#define DYND_CUDA_HOST_DEVICE __host__ DYND_CUDA_DEVICE // A function that is compiled for both the host and the device
#define DYND_CUDA_GLOBAL __global__ // A function that is a CUDA kernel

#else // We are not compiling with NVIDIA's nvcc

#define DYND_CUDA_HOST_ARCH
#define DYND_CUDA_HOST_DEVICE

namespace dynd {
// Prevent isfinite from nvcc clashing with isfinite from cmath
template <typename T>
inline bool isfinite(T arg) {
    return std::isfinite(arg);
}
} // namespace dynd

#endif // __CUDACC_

#ifdef DYND_CUDA

namespace dynd {

    /**
     * Configuration of threads in a CUDA kernel, specialized to the number of grid
     * and block dimensions for efficiency.
     */
    template <int grid_ndim, int block_ndim>
    struct cuda_global_config;

    template <>
    struct cuda_global_config<1, 1> {
        unsigned int grid;
        unsigned int block;
        unsigned int threads;

        cuda_global_config() {}

        cuda_global_config(unsigned int grid, unsigned int block)
            : grid(grid), block(block), threads(grid * block) {}
    };

    template <int grid_ndim, int block_ndim>
    inline cuda_global_config<grid_ndim, block_ndim> make_cuda_global_config(size_t count);

    template <>
    inline cuda_global_config<1, 1> make_cuda_global_config(size_t DYND_UNUSED(count)) {
        // TODO: This should be chosen optimally depending on 'count'. For now, we default to a "good" configuration.
        return cuda_global_config<1, 1>(256, 256);
    }

#ifdef __CUDACC__
    /**
     * Returns the unique index of a thread in a CUDA kernel.
     */
    template <int grid_ndim, int block_ndim>
    DYND_CUDA_DEVICE inline unsigned int get_cuda_global_thread();

    template <>
    DYND_CUDA_DEVICE inline unsigned int get_cuda_global_thread<1, 1>() {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }
#endif // __CUDACC__

} // namespace dynd

#endif // DYND_CUDA

#endif // _DYND__CUDA_CONFIG_HPP_
