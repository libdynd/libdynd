//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifdef DYND_CUDA

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/types/cuda_device_type.hpp>

using namespace std;
using namespace dynd;

TEST(CudaDeviceArrayDType, Basic) {
    ndt::type d = ndt::make_cuda_device(ndt::make_type<int32_t>());
    EXPECT_EQ(cuda_device_type_id, d.get_type_id());
    EXPECT_EQ(memory_kind, d.get_kind());
}

#endif // DYND_CUDA
