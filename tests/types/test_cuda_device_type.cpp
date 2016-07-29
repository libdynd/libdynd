//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <dynd/array.hpp>
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/types/cuda_host_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/gtest.hpp>

using namespace std;
using namespace dynd;

#ifdef DYND_CUDA

TEST(CUDADeviceType, Simple) {
  ndt::type d = ndt::make_cuda_device(ndt::make_type<int32_t>());
  EXPECT_EQ(cuda_device_id, d.get_id());
  EXPECT_EQ(memory_id, d.get_base_id());
  EXPECT_EQ(ndt::make_type<int32_t>(), d.p("storage_type").as<ndt::type>());
  EXPECT_FALSE(d.is_expression());
  EXPECT_EQ(d, ndt::type("cuda_device[int32]"));
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // A memory type cannot be the element of an array dimension
  EXPECT_THROW(ndt::make_fixed_dimsym(ndt::make_cuda_device(ndt::make_type<int32_t>())), invalid_argument);

  // Only built-in types can be allocated in CUDA global memory
  // Todo: Find a place to throw this exception
  //  EXPECT_THROW(ndt::make_cuda_device(ndt::make_pointer<char>()), runtime_error);
}

TEST(CUDADeviceType, BuiltIn) {
  ndt::type d = ndt::make_cuda_device(ndt::make_type<float>());
  EXPECT_EQ(cuda_device_id, d.get_id());
  EXPECT_EQ(memory_id, d.get_base_id());
  EXPECT_EQ(sizeof(float), d.get_data_size());
  // CUDA host type and CUDA device type have the same data alignment
  EXPECT_EQ(ndt::make_cuda_host(ndt::make_type<float>()).get_data_alignment(), d.get_data_alignment());
  EXPECT_EQ(d, ndt::type("cuda_device[float32]"));
}

TEST(CUDADeviceType, FixedDim) {
  ndt::type d = ndt::make_cuda_device(ndt::make_fixed_dimsym(ndt::make_type<float>()));
  EXPECT_EQ(cuda_device_id, d.get_id());
  EXPECT_EQ(memory_id, d.get_base_id());
  EXPECT_EQ(d.extended<base_memory_type>()->get_element_type().get_data_size(), d.get_data_size());
  EXPECT_EQ(d, ndt::type("cuda_device[Fixed * float32]"));
}

TEST(CUDADeviceType, IsTypeSubarray) {
  EXPECT_TRUE(ndt::make_cuda_device(ndt::make_type<int32_t>())
                  .is_type_subarray(ndt::make_cuda_device(ndt::make_type<int32_t>())));
  EXPECT_TRUE(ndt::make_cuda_device(ndt::make_type<int32_t>()).is_type_subarray(ndt::make_type<int32_t>()));
  EXPECT_TRUE(ndt::make_cuda_device(ndt::make_fixed_dimsym(ndt::make_type<int32_t>()))
                  .is_type_subarray(ndt::make_type<int32_t>()));
  EXPECT_FALSE(ndt::make_type<int32_t>().is_type_subarray(ndt::make_cuda_device(ndt::make_type<int32_t>())));
}

#endif
