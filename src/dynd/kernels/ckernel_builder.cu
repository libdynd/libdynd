//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/ckernel_builder.hpp>

__global__ void dynd::cuda_device_destroy(ckernel_prefix *self) { self->destroy(); }