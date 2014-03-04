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
#include <dynd/types/cuda_host_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

TEST(CUDADeviceType, Basic) {
    ndt::type d = ndt::make_cuda_device(ndt::make_type<int32_t>());
    EXPECT_EQ(cuda_device_type_id, d.get_type_id());
    EXPECT_EQ(memory_kind, d.get_kind());
    EXPECT_EQ(ndt::make_type<int32_t>(), d.p("storage_type").as<ndt::type>());
    EXPECT_FALSE(d.is_expression());
	EXPECT_EQ(d, ndt::type("cuda_device[int32]"));

    // A memory type cannot have an array dimension type as storage
    EXPECT_THROW(ndt::make_cuda_device(ndt::make_strided_dim(ndt::make_type<int32_t>())), runtime_error);

    // Only built-in types can be allocated in CUDA global memory
    EXPECT_THROW(ndt::make_cuda_device(ndt::make_pointer<char>()), runtime_error);
}

TEST(CUDADeviceType, BuiltIn) {
    ndt::type d = ndt::make_cuda_device(ndt::make_type<float>());
    EXPECT_EQ(cuda_device_type_id, d.get_type_id());
    EXPECT_EQ(memory_kind, d.get_kind());
    EXPECT_EQ(sizeof(float), d.get_data_size());
    // CUDA host type and CUDA device type have the same data alignment
    EXPECT_EQ(ndt::make_cuda_host(ndt::make_type<float>()).get_data_alignment(), d.get_data_alignment());
	EXPECT_EQ(d, ndt::type("cuda_device[float32]"));
}

TEST(CUDADeviceType, IsTypeSubarray) {
    EXPECT_TRUE(ndt::make_cuda_device(ndt::make_type<int32_t>()).is_type_subarray(ndt::make_cuda_device(ndt::make_type<int32_t>())));
    EXPECT_TRUE(ndt::make_cuda_device(ndt::make_type<int32_t>()).is_type_subarray(ndt::make_type<int32_t>()));
    EXPECT_TRUE(ndt::make_strided_dim(ndt::make_cuda_device(ndt::make_type<int32_t>())).is_type_subarray(ndt::make_type<int32_t>()));
    EXPECT_FALSE(ndt::make_type<int32_t>().is_type_subarray(ndt::make_cuda_device(ndt::make_type<int32_t>())));
}

#endif // DYND_CUDA
