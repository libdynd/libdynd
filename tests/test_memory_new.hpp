//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <cmath>
#include <iostream>
#include <stdexcept>

#include <dynd/callable.hpp>
#include <dynd_assertions.hpp>

template <typename T>
class Memory;

typedef std::integral_constant<dynd::kernel_request_t, dynd::kernel_request_host> HostKernelRequest;

template <>
class Memory<HostKernelRequest> : public ::testing::Test {
public:
  static const dynd::kernel_request_t KernelRequest = HostKernelRequest::value;

  // This is a workaround for a CUDA bug
  template <typename T>
  static dynd::nd::array To(const std::initializer_list<T> &a) {
    return dynd::nd::array(a);
  }

  static dynd::nd::array To(const dynd::nd::array &a) { return a; }
};

#ifdef DYND_CUDA

typedef std::integral_constant<dynd::kernel_request_t, dynd::kernel_request_cuda_device> CUDADeviceKernelRequest;

template <>
class Memory<CUDADeviceKernelRequest> : public ::testing::Test {
public:
  static const dynd::kernel_request_t KernelRequest = CUDADeviceKernelRequest::value;

  static dynd::nd::array To(const dynd::nd::array &a) { return a.to_cuda_device(); }

  // This is a workaround for a CUDA bug
  template <typename T>
  static dynd::nd::array To(const std::initializer_list<T> &a) {
    return dynd::nd::array(a).to_cuda_device();
  }
};

#endif
