//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/ckernel_builder.hpp>

using namespace std;
using namespace dynd;

#ifdef __CUDACC__

ckernel_builder<kernel_request_cuda_device>::pooled_allocator ckernel_builder<kernel_request_cuda_device>::allocator;

__global__ void dynd::cuda_device_destroy(ckernel_prefix *self)
{
  self->destroy();
}

#endif