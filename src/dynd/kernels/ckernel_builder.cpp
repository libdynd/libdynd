//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/ckernel_builder.hpp>

#ifdef __CUDACC__

void *dynd::ckernel_builder<dynd::kernel_request_cuda_device>::pool = NULL;
size_t dynd::ckernel_builder<dynd::kernel_request_cuda_device>::pool_size = 0;

__global__ void dynd::cuda_device_destroy(ckernel_prefix *self) { self->destroy(); }

#endif