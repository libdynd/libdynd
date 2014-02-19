//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CUDA_CONFIG_HPP_
#define _DYND__CUDA_CONFIG_HPP_

#include <cmath>

#ifdef __CUDACC__
#define DYND_CUDA_DEVICE_CALLABLE __device__
#define DYND_CUDA_HOST_DEVICE_CALLABLE __host__ DYND_CUDA_DEVICE_CALLABLE
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

#endif // _DYND__CUDA_CONFIG_HPP_
