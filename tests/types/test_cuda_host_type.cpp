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
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

TEST(CUDAHostType, Basic) {
    ndt::type d = ndt::make_cuda_host(ndt::make_type<int32_t>());
    EXPECT_EQ(cuda_host_type_id, d.get_type_id());
    EXPECT_EQ(memory_kind, d.get_kind());
    EXPECT_EQ(ndt::make_type<int32_t>(), d.p("storage_type").as<ndt::type>());
    EXPECT_FALSE(d.is_expression());
    EXPECT_EQ((unsigned int)cudaHostAllocDefault, static_cast<const cuda_host_type *>(d.extended())->get_cuda_host_flags());
	EXPECT_EQ(d, ndt::type("cuda_host[int32]"));

    // A memory type cannot have an array dimension type as storage
    EXPECT_THROW(ndt::make_cuda_host(ndt::make_strided_dim(ndt::make_type<int32_t>())), runtime_error);

    d = ndt::make_cuda_host(ndt::make_type<float>(), cudaHostAllocMapped);
    EXPECT_EQ((unsigned int)cudaHostAllocMapped, static_cast<const cuda_host_type *>(d.extended())->get_cuda_host_flags());
}

TEST(CUDAHostType, BuiltIn) {
    ndt::type d = ndt::make_cuda_host(ndt::make_type<float>());
    EXPECT_EQ(cuda_host_type_id, d.get_type_id());
    EXPECT_EQ(memory_kind, d.get_kind());
    EXPECT_EQ(sizeof(float), d.get_data_size());
    // CUDA host type and CUDA device type have the same data alignment
    EXPECT_EQ(ndt::make_cuda_device(ndt::make_type<float>()).get_data_alignment(), d.get_data_alignment());
	EXPECT_EQ(d, ndt::type("cuda_host[float32]"));
}

TEST(CUDAHostType, IsTypeSubarray) {
    EXPECT_TRUE(ndt::make_cuda_host(ndt::make_type<int32_t>()).is_type_subarray(ndt::make_cuda_host(ndt::make_type<int32_t>())));
    EXPECT_TRUE(ndt::make_cuda_host(ndt::make_type<int32_t>()).is_type_subarray(ndt::make_type<int32_t>()));
    EXPECT_TRUE(ndt::make_strided_dim(ndt::make_cuda_host(ndt::make_type<int32_t>())).is_type_subarray(ndt::make_type<int32_t>()));
    EXPECT_FALSE(ndt::make_type<int32_t>().is_type_subarray(ndt::make_cuda_host(ndt::make_type<int32_t>())));
}

#endif // DYND_CUDA
