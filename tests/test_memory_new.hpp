//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <stdexcept>
#include <cmath>

#include <inc_gtest.hpp>

#include <dynd/func/arrfunc.hpp>

template <typename T>
class Memory;

typedef std::integral_constant<kernel_request_t, kernel_request_host>
    HostKernelRequest;

template <>
class Memory<HostKernelRequest> : public ::testing::Test {
public:
  static const kernel_request_t KernelRequest = HostKernelRequest::value;

  // This is a workaround for a CUDA bug
  template <typename T>
  static nd::array To(const std::initializer_list<T> &a)
  {
    return nd::array(a);
  }

  static nd::array To(const nd::array &a) { return a; }
};

#ifdef DYND_CUDA

typedef integral_constant<kernel_request_t, kernel_request_cuda_device>
    CUDADeviceKernelRequest;

template <>
class Memory<CUDADeviceKernelRequest> : public ::testing::Test {
public:
  static const kernel_request_t KernelRequest = CUDADeviceKernelRequest::value;

  static nd::array To(const nd::array &a) { return a.to_cuda_device(); }

  // This is a workaround for a CUDA bug
  template <typename T>
  static nd::array To(const std::initializer_list<T> &a)
  {
    return nd::array(a).to_cuda_device();
  }
};

#endif