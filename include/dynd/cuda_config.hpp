//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CUDA_CONFIG_HPP_
#define _DYND__CUDA_CONFIG_HPP_

#include <cmath>

#ifdef DYND_CUDA
#include <cuda_runtime.h>
#endif // DYND_CUDA


#ifdef __CUDACC__
#define DYND_CUDA_DEVICE_CALLABLE __device__
#define DYND_CUDA_HOST_DEVICE_CALLABLE __host__ DYND_CUDA_DEVICE_CALLABLE
#define DYND_CUDA_GLOBAL_CALLABLE __global__
#ifdef __CUDA_ARCH__
#define DYND_CUDA_DEVICE_ARCH
#else
#define DYND_CUDA_HOST_ARCH 
#endif
#else

#define DYND_CUDA_HOST_ARCH

#define DYND_CUDA_HOST_DEVICE_CALLABLE

namespace dynd {
// Prevent isfinite from nvcc clashing with isfinite from cmath
template <typename T>
inline bool isfinite(T arg) {
    return std::isfinite(arg);
}
} // namespace dynd

#endif

#ifdef DYND_CUDA
namespace dynd {
    template <int grid_dim, int block_dim>
    struct cuda_global_config;

    template <>
    struct cuda_global_config<1, 1> {
        size_t grids;
        size_t blocks;
        size_t threads;

        cuda_global_config() {}

        cuda_global_config(int gr, int bl) : grids(gr), blocks(bl), threads(grids * blocks) {}
    };

    template <int grid_dim, int block_dim>
    inline cuda_global_config<grid_dim, block_dim> make_cuda_global_config(size_t count);

    template <>
    inline cuda_global_config<1, 1> make_cuda_global_config(size_t DYND_UNUSED(count)) {
        return cuda_global_config<1, 1>(256, 256);
    }

#ifdef __CUDACC__
    template <int grid_dim, int block_dim>
    DYND_CUDA_DEVICE_CALLABLE inline size_t get_cuda_global_thread();

    template <>
    DYND_CUDA_DEVICE_CALLABLE inline size_t get_cuda_global_thread<1, 1>() {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }
#endif
} // namespace dynd
#endif // DYND_CUDA

#endif // _DYND__CUDA_CONFIG_HPP_
