//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifdef DYND_CUDA

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/types/cuda_host_type.hpp>
#include <dynd/types/strided_dim_type.hpp>

using namespace std;
using namespace dynd;

TEST(CUDAHostArrayDType, Basic) {
    ndt::type d = ndt::make_cuda_host(ndt::make_type<int32_t>());
    EXPECT_EQ(cuda_host_type_id, d.get_type_id());
    EXPECT_EQ(memory_kind, d.get_kind());
}

TEST(CUDAHostArrayDType, ShiftedMemory) {
    ndt::type d = ndt::make_cuda_host(ndt::make_strided_dim(ndt::make_type<int32_t>()));
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_cuda_host(ndt::make_type<int32_t>())), d.with_right_shifted_memory_type());
}




#endif // DYND_CUDA
