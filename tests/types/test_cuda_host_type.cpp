//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/types/cuda_host_type.hpp>
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

#ifdef DYND_CUDA

TEST(CUDAHostType, Basic)
{
  ndt::type d = ndt::make_cuda_host(ndt::make_type<int32_t>());
  EXPECT_EQ(cuda_host_id, d.get_id());
  EXPECT_EQ(memory_id, d.get_base_id());
  EXPECT_EQ(ndt::make_type<int32_t>(), d.p("storage_type").as<ndt::type>());
  EXPECT_FALSE(d.is_expression());
  EXPECT_EQ((unsigned int)cudaHostAllocDefault, d.extended<cuda_host_type>()->get_cuda_host_flags());
  EXPECT_EQ(d, ndt::type("cuda_host[int32]"));
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));
  // A memory type cannot be the element of an array dimension
  EXPECT_THROW(ndt::make_fixed_dimsym(ndt::make_cuda_host(ndt::make_type<int32_t>())), invalid_argument);
  d = ndt::make_cuda_host(ndt::make_type<float>(), cudaHostAllocMapped);
  EXPECT_EQ((unsigned int)cudaHostAllocMapped, d.extended<cuda_host_type>()->get_cuda_host_flags());
}

TEST(CUDAHostType, BuiltIn)
{
  ndt::type d = ndt::make_cuda_host(ndt::make_type<float>());
  EXPECT_EQ(cuda_host_id, d.get_id());
  EXPECT_EQ(memory_id, d.get_base_id());
  EXPECT_EQ(sizeof(float), d.get_data_size());
  // CUDA host type and CUDA device type have the same data alignment
  EXPECT_EQ(ndt::make_cuda_device(ndt::make_type<float>()).get_data_alignment(), d.get_data_alignment());
  EXPECT_EQ(d, ndt::type("cuda_host[float32]"));
}

TEST(CUDAHostType, FixedDim)
{
  ndt::type d = ndt::make_cuda_host(ndt::make_fixed_dimsym(ndt::make_type<float>()));
  EXPECT_EQ(cuda_host_id, d.get_id());
  EXPECT_EQ(memory_id, d.get_base_id());
  EXPECT_EQ(d.extended<base_memory_type>()->get_element_type().get_data_size(), d.get_data_size());
  EXPECT_EQ(d, ndt::type("cuda_host[Fixed * float32]"));
}

TEST(CUDAHostType, IsTypeSubarray)
{
  EXPECT_TRUE(
      ndt::make_cuda_host(ndt::make_type<int32_t>()).is_type_subarray(ndt::make_cuda_host(ndt::make_type<int32_t>())));
  EXPECT_TRUE(ndt::make_cuda_host(ndt::make_type<int32_t>()).is_type_subarray(ndt::make_type<int32_t>()));
  EXPECT_TRUE(ndt::make_cuda_host(ndt::make_fixed_dimsym(ndt::make_type<int32_t>()))
                  .is_type_subarray(ndt::make_type<int32_t>()));
  EXPECT_FALSE(ndt::make_type<int32_t>().is_type_subarray(ndt::make_cuda_host(ndt::make_type<int32_t>())));
}

#endif
