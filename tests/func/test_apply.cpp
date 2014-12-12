//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/array.hpp>
#include <dynd/func/apply_arrfunc.hpp>
#include <dynd/func/call_callable.hpp>
#include <dynd/types/cfixed_dim_type.hpp>

using namespace std;
using namespace dynd;

template <typename T>
class Apply;

typedef integral_constant<kernel_request_t, kernel_request_host>
    KernelRequestHost;

template <>
class Apply<KernelRequestHost> : public ::testing::Test {
public:
  static const kernel_request_t KernelRequest = KernelRequestHost::value;

  template <typename T>
  static nd::array To(const std::initializer_list<T> &a)
  {
    return nd::array(a);
  }

  static nd::array To(nd::array a) { return a; }
};

#ifdef DYND_CUDA

typedef integral_constant<kernel_request_t, kernel_request_cuda_device>
    KernelRequestCUDADevice;

template <>
class Apply<KernelRequestCUDADevice> : public ::testing::Test {
public:
  static const kernel_request_t KernelRequest = KernelRequestCUDADevice::value;

  static nd::array To(const nd::array &a) { return a.to_cuda_device(); }

  template <typename T>
  static nd::array To(const std::initializer_list<T> &a)
  {
    return nd::array(a).to_cuda_device();
  }
};

#endif

TYPED_TEST_CASE_P(Apply);

template <kernel_request_t kernreq, typename func_type>
struct func_wrapper;

#if !(defined(_MSC_VER) && _MSC_VER == 1800)
#define FUNC_WRAPPER(KERNREQ, ...)                                             \
  template <typename R, typename... A>                                         \
  struct func_wrapper<KERNREQ, R (*)(A...)> {                                  \
    R (*func)(A...);                                                           \
                                                                               \
    DYND_CUDA_HOST_DEVICE func_wrapper(R (*func)(A...)) : func(func) {}        \
                                                                               \
    __VA_ARGS__ R operator()(A... a) const { return (*func)(a...); }           \
  };
#else
// Workaround for MSVC 2013 variadic template bug
// https://connect.microsoft.com/VisualStudio/Feedback/Details/1034062
#define FUNC_WRAPPER(KERNREQ, ...)                                             \
  template <typename R, R (*func)()>                                           \
  struct func_wrapper<KERNREQ, R (*)()> {                                      \
    R (*func)();                                                               \
                                                                               \
    func_wrapper(R (*func)()) : func(func) {}                                  \
                                                                               \
    __VA_ARGS__ R operator()() const { return (*func)(); }                     \
  };                                                                           \
                                                                               \
  template <typename R, typename A0, R (*func)(A0)>                            \
  struct func_wrapper<KERNREQ, R (*)(A0)> {                                    \
    R (*func)(A0);                                                             \
                                                                               \
    func_wrapper(R (*func)(A0)) : func(func) {}                                \
                                                                               \
    __VA_ARGS__ R operator()(A0 a0) const { return (*func)(a0); }              \
  };                                                                           \
                                                                               \
  template <typename R, typename A0, typename A1, R (*func)(A0, A1)>           \
  struct func_wrapper<KERNREQ, R (*)(A0, A1)> {                                \
    R (*func)(A0, A1);                                                         \
                                                                               \
    func_wrapper(R (*func)(A0, A1)) : func(func) {}                            \
                                                                               \
    __VA_ARGS__ R operator()(A0 a0, A1 a1) const { return (*func)(a0, a1); }   \
  };                                                                           \
                                                                               \
  template <typename R, typename A0, typename A1, typename A2,                 \
            R (*func)(A0, A1, A2)>                                             \
  struct func_wrapper<KERNREQ, R (*)(A0, A1, A2)> {                            \
    R (*func)(A0, A1, A2);                                                     \
                                                                               \
    func_wrapper(R (*func)(A0, A1, A2)) : func(func) {}                        \
                                                                               \
    __VA_ARGS__ R operator()(A0 a0, A1 a1, A2 a2) const                        \
    {                                                                          \
      return (*func)(a0, a1, a2);                                              \
    }                                                                          \
  }
#endif

FUNC_WRAPPER(kernel_request_host);
FUNC_WRAPPER(kernel_request_cuda_device, __device__);

#undef FUNC_WRAPPER

#define GET_HOST_FUNC(NAME)                                                    \
  template <kernel_request_t kernreq>                                          \
  typename std::enable_if<kernreq == kernel_request_host,                      \
                          decltype(&NAME)>::type get_##NAME()                  \
  {                                                                            \
    return &NAME;                                                              \
  }

#define HOST_FUNC_AS_CALLABLE(NAME)                                            \
  template <kernel_request_t kernreq>                                          \
  struct NAME##_as_callable;                                                   \
                                                                               \
  template <>                                                                  \
  struct NAME##_as_callable<kernel_request_host>                               \
      : func_wrapper<kernel_request_host, decltype(&NAME)> {                   \
    NAME##_as_callable()                                                       \
        : func_wrapper<kernel_request_host, decltype(&NAME)>(                  \
              get_##NAME<kernel_request_host>())                               \
    {                                                                          \
    }                                                                          \
  }

#ifdef __CUDA_ARCH__
#define GET_CUDA_DEVICE_FUNC_BODY(NAME) return &NAME;
#else
#define GET_CUDA_DEVICE_FUNC_BODY(NAME)                                        \
  decltype(&NAME) res;                                                         \
  decltype(&NAME) *func, *cuda_device_func;                                    \
  throw_if_not_cuda_success(                                                   \
      cudaHostAlloc(&func, sizeof(decltype(&NAME)), cudaHostAllocMapped));     \
  throw_if_not_cuda_success(                                                   \
      cudaHostGetDevicePointer(&cuda_device_func, func, 0));                   \
  get_cuda_device_##NAME << <1, 1>>> (cuda_device_func);                       \
  throw_if_not_cuda_success(cudaDeviceSynchronize());                          \
  res = *func;                                                                 \
  throw_if_not_cuda_success(cudaFreeHost(func));                               \
                                                                               \
  return res;

#endif

#ifdef __CUDACC__

#define GET_CUDA_DEVICE_FUNC(NAME)                                             \
  __global__ void get_cuda_device_##NAME(void *res)                            \
  {                                                                            \
    *reinterpret_cast<decltype(&NAME) *>(res) = &NAME;                         \
  }                                                                            \
                                                                               \
  template <kernel_request_t kernreq>                                          \
  DYND_CUDA_HOST_DEVICE typename std::enable_if<                               \
      kernreq == kernel_request_cuda_device, decltype(&NAME)>::type            \
      get_##NAME()                                                             \
  {                                                                            \
    GET_CUDA_DEVICE_FUNC_BODY(NAME)                                            \
  }

#define CUDA_DEVICE_FUNC_AS_CALLABLE(NAME)                                     \
  template <kernel_request_t kernreq>                                          \
  struct NAME##_as_callable;                                                   \
                                                                               \
  template <>                                                                  \
  struct NAME##_as_callable<kernel_request_cuda_device>                        \
      : func_wrapper<kernel_request_cuda_device, decltype(&NAME)> {            \
    DYND_CUDA_HOST_DEVICE NAME##_as_callable()                                 \
        : func_wrapper<kernel_request_cuda_device, decltype(&NAME)>(           \
              get_##NAME<kernel_request_cuda_device>())                        \
    {                                                                          \
    }                                                                          \
  }

#endif

int func0(int x, int y) { return 2 * (x - y); }

GET_HOST_FUNC(func0)
HOST_FUNC_AS_CALLABLE(func0);

#ifdef __CUDACC__

__device__ double func1(double x, int y) { return x + 2.75 * y; }

GET_CUDA_DEVICE_FUNC(func1)
CUDA_DEVICE_FUNC_AS_CALLABLE(func1);

#endif

#undef GET_HOST_FUNC
#undef HOST_FUNC_AS_CALLABLE

#ifdef __CUDACC__

#undef GET_CUDA_DEVICE_FUNC
#undef CUDA_DEVICE_FUNC_AS_CALLABLE

#endif

TEST(Apply, Function)
{
  typedef Apply<KernelRequestHost> TestFixture;

  nd::arrfunc af;

  af = nd::apply::make<kernel_request_host, decltype(&func0), &func0>();
  EXPECT_ARR_EQ(TestFixture::To(4), af(TestFixture::To(5), TestFixture::To(3)));
}

TEST(Apply, FunctionWithKeywords)
{
  typedef Apply<KernelRequestHost> TestFixture;

  nd::arrfunc af;

  af = nd::apply::make<decltype(&func0), &func0>("y");
  EXPECT_ARR_EQ(TestFixture::To(4),
                af(TestFixture::To(5), kwds("y", TestFixture::To(3))));

  af = nd::apply::make<decltype(&func0), &func0>("x", "y");
  EXPECT_ARR_EQ(TestFixture::To(4),
                af(kwds("x", TestFixture::To(5), "y", TestFixture::To(3))));
}

TYPED_TEST_P(Apply, Callable)
{
  nd::arrfunc af;

  if (TestFixture::KernelRequest == kernel_request_host) {
    af = nd::apply::make<kernel_request_host>(
        get_func0<kernel_request_host>());
    EXPECT_ARR_EQ(TestFixture::To(4),
                  af(TestFixture::To(5), TestFixture::To(3)));

    af = nd::apply::make<kernel_request_host>(
        func0_as_callable<kernel_request_host>());
    EXPECT_ARR_EQ(TestFixture::To(4),
                  af(TestFixture::To(5), TestFixture::To(3)));

    af = nd::apply::make<kernel_request_host,
                                func0_as_callable<kernel_request_host>>();
    EXPECT_ARR_EQ(TestFixture::To(4),
                  af(TestFixture::To(5), TestFixture::To(3)));
  }

#ifdef __CUDACC__
  if (TestFixture::KernelRequest == kernel_request_cuda_device) {
    af = nd::apply::make<kernel_request_cuda_device>(
        get_func1<kernel_request_cuda_device>());
    EXPECT_ARR_EQ(TestFixture::To(58.25),
                  af(TestFixture::To(3.25), TestFixture::To(20)));

    af = nd::apply::make<kernel_request_cuda_device>(
        func1_as_callable<kernel_request_cuda_device>());
    EXPECT_ARR_EQ(TestFixture::To(58.25),
                  af(TestFixture::To(3.25), TestFixture::To(20)));

    af =
        nd::apply::make<kernel_request_cuda_device,
                               func1_as_callable<kernel_request_cuda_device>>();
    EXPECT_ARR_EQ(TestFixture::To(58.25),
                  af(TestFixture::To(3.25), TestFixture::To(20)));
  }
#endif
}

TYPED_TEST_P(Apply, CallableWithKeywords)
{
  nd::arrfunc af;

  if (TestFixture::KernelRequest == kernel_request_host) {
    af = nd::apply::make<kernel_request_host>(
        get_func0<kernel_request_host>(), "y");
    EXPECT_ARR_EQ(TestFixture::To(4),
                  af(TestFixture::To(5), kwds("y", TestFixture::To(3))));

    af = nd::apply::make<kernel_request_host>(
        get_func0<kernel_request_host>(), "x", "y");
    EXPECT_ARR_EQ(TestFixture::To(4),
                  af(kwds("x", TestFixture::To(5), "y", TestFixture::To(3))));

    af = nd::apply::make(func0_as_callable<kernel_request_host>(), "y");
    EXPECT_ARR_EQ(TestFixture::To(4), af(5, kwds("y", TestFixture::To(3))));

    af = nd::apply::make(func0_as_callable<kernel_request_host>(), "x",
                                "y");
    EXPECT_ARR_EQ(TestFixture::To(4),
                  af(kwds("x", TestFixture::To(5), "y", TestFixture::To(3))));
  }

#ifdef __CUDACC__
  if (TestFixture::KernelRequest == kernel_request_cuda_device) {
    af = nd::apply::make<kernel_request_cuda_device>(
        get_func1<kernel_request_cuda_device>(), "y");
    EXPECT_ARR_EQ(TestFixture::To(58.25),
                  af(TestFixture::To(3.25), kwds("y", TestFixture::To(20))));

    af = nd::apply::make<kernel_request_cuda_device>(
        get_func1<kernel_request_cuda_device>(), "x", "y");
    EXPECT_ARR_EQ(TestFixture::To(58.25), af(kwds("x", TestFixture::To(3.25),
                                                  "y", TestFixture::To(20))));

    af = nd::apply::make<kernel_request_cuda_device>(
        func1_as_callable<kernel_request_cuda_device>(), "y");
    EXPECT_ARR_EQ(TestFixture::To(58.25),
                  af(TestFixture::To(3.25), kwds("y", TestFixture::To(20))));

/*
    af = nd::apply::make<kernel_request_cuda_device>(
        func1_as_callable<kernel_request_cuda_device>(), "x", "y");
    EXPECT_ARR_EQ(TestFixture::To(58.25), af(kwds("x", TestFixture::To(3.25),
                                                  "y", TestFixture::To(20))));
*/
  }
#endif
}

REGISTER_TYPED_TEST_CASE_P(Apply, Callable, CallableWithKeywords);

INSTANTIATE_TYPED_TEST_CASE_P(HostMemory, Apply, KernelRequestHost);

#ifdef __CUDACC__
INSTANTIATE_TYPED_TEST_CASE_P(CUDADeviceMemory, Apply, KernelRequestCUDADevice);
#endif