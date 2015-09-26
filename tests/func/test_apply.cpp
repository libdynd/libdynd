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
#include "../test_memory_new.hpp"

#include <dynd/array.hpp>
#include <dynd/func/apply.hpp>

using namespace std;
using namespace dynd;

#if defined(_WIN32) && defined(DYND_CUDA)
// Workaround for CUDA NVCC bug where it deduces a reference to a function
// pointer instead of a function pointer in decltype.
template <typename T>
struct cuda_decltype_workaround {
  typedef std::remove_reference<std::remove_pointer<T>::type>::type *type;
};
#define CUDA_DECLTYPE_WORKAROUND(x) cuda_decltype_workaround<decltype(x)>::type
#else
#define CUDA_DECLTYPE_WORKAROUND(x) decltype(x)
#endif

template <typename T>
class Apply : public Memory<T> {
};

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
  template <typename R>                                                        \
  struct func_wrapper<KERNREQ, R (*)()> {                                      \
    R (*func)();                                                               \
                                                                               \
    DYND_CUDA_HOST_DEVICE func_wrapper(R (*func)()) : func(func) {}            \
                                                                               \
    __VA_ARGS__ R operator()() const { return (*func)(); }                     \
  };                                                                           \
                                                                               \
  template <typename R, typename A0>                                           \
  struct func_wrapper<KERNREQ, R (*)(A0)> {                                    \
    R (*func)(A0);                                                             \
                                                                               \
    DYND_CUDA_HOST_DEVICE func_wrapper(R (*func)(A0)) : func(func) {}          \
                                                                               \
    __VA_ARGS__ R operator()(A0 a0) const { return (*func)(a0); }              \
  };                                                                           \
                                                                               \
  template <typename R, typename A0, typename A1>                              \
  struct func_wrapper<KERNREQ, R (*)(A0, A1)> {                                \
    R (*func)(A0, A1);                                                         \
                                                                               \
    DYND_CUDA_HOST_DEVICE func_wrapper(R (*func)(A0, A1)) : func(func) {}      \
                                                                               \
    __VA_ARGS__ R operator()(A0 a0, A1 a1) const { return (*func)(a0, a1); }   \
  };                                                                           \
                                                                               \
  template <typename R, typename A0, typename A1, typename A2>                 \
  struct func_wrapper<KERNREQ, R (*)(A0, A1, A2)> {                            \
    R (*func)(A0, A1, A2);                                                     \
                                                                               \
    DYND_CUDA_HOST_DEVICE func_wrapper(R (*func)(A0, A1, A2)) : func(func) {}  \
                                                                               \
    __VA_ARGS__ R operator()(A0 a0, A1 a1, A2 a2) const                        \
    {                                                                          \
      return (*func)(a0, a1, a2);                                              \
    }                                                                          \
  }
#endif

FUNC_WRAPPER(kernel_request_host);

#ifdef DYND_CUDA
FUNC_WRAPPER(kernel_request_cuda_device, __device__);
#endif

#undef FUNC_WRAPPER

#define GET_HOST_FUNC(NAME)                                                    \
  template <kernel_request_t kernreq>                                          \
  typename std::enable_if<kernreq == kernel_request_host,                      \
                          CUDA_DECLTYPE_WORKAROUND(&NAME)>::type get_##NAME()  \
  {                                                                            \
    return &NAME;                                                              \
  }

#define HOST_FUNC_AS_CALLABLE(NAME)                                            \
  template <kernel_request_t kernreq>                                          \
  struct NAME##_as_callable;                                                   \
                                                                               \
  template <>                                                                  \
  struct NAME##_as_callable<kernel_request_host>                               \
      : func_wrapper<kernel_request_host, CUDA_DECLTYPE_WORKAROUND(&NAME)> {   \
    NAME##_as_callable()                                                       \
        : func_wrapper<kernel_request_host, CUDA_DECLTYPE_WORKAROUND(&NAME)>(  \
              get_##NAME<kernel_request_host>())                               \
    {                                                                          \
    }                                                                          \
  }

#ifdef __CUDA_ARCH__
#define GET_CUDA_DEVICE_FUNC_BODY(NAME) return &NAME;
#else
#define GET_CUDA_DEVICE_FUNC_BODY(NAME)                                        \
  CUDA_DECLTYPE_WORKAROUND(&NAME) func;                                        \
  CUDA_DECLTYPE_WORKAROUND(&NAME) * cuda_device_func;                          \
  cuda_throw_if_not_success(                                                   \
      cudaMalloc(&cuda_device_func, sizeof(CUDA_DECLTYPE_WORKAROUND(&NAME)))); \
  get_cuda_device_##NAME << <1, 1>>>                                           \
      (reinterpret_cast<void *>(cuda_device_func));                            \
  cuda_throw_if_not_success(cudaMemcpy(                                        \
      &func, cuda_device_func, sizeof(CUDA_DECLTYPE_WORKAROUND(&NAME)),        \
      cudaMemcpyDeviceToHost));                                                \
  cuda_throw_if_not_success(cudaFree(cuda_device_func));                       \
  return func;
#endif

#ifdef __CUDACC__

#define GET_CUDA_DEVICE_FUNC(NAME)                                             \
  __global__ void get_cuda_device_##NAME(void *res)                            \
  {                                                                            \
    *reinterpret_cast<CUDA_DECLTYPE_WORKAROUND(&NAME) *>(res) = &NAME;         \
  }                                                                            \
                                                                               \
  template <kernel_request_t kernreq>                                          \
  DYND_CUDA_HOST_DEVICE typename std::enable_if<                               \
      kernreq == kernel_request_cuda_device,                                   \
      CUDA_DECLTYPE_WORKAROUND(&NAME)>::type get_##NAME()                      \
  {                                                                            \
    GET_CUDA_DEVICE_FUNC_BODY(NAME)                                            \
  }

#define CUDA_DEVICE_FUNC_AS_CALLABLE(NAME)                                     \
  template <kernel_request_t kernreq>                                          \
  struct NAME##_as_callable;                                                   \
                                                                               \
  template <>                                                                  \
  struct NAME##_as_callable<kernel_request_cuda_device>                        \
      : func_wrapper<kernel_request_cuda_device,                               \
                     CUDA_DECLTYPE_WORKAROUND(&NAME)> {                        \
    DYND_CUDA_HOST_DEVICE NAME##_as_callable()                                 \
        : func_wrapper<kernel_request_cuda_device,                             \
                       CUDA_DECLTYPE_WORKAROUND(&NAME)>(                       \
              get_##NAME<kernel_request_cuda_device>())                        \
    {                                                                          \
    }                                                                          \
  }

#endif

#ifdef __CUDACC__
#define GET_CUDA_HOST_DEVICE_FUNC(NAME)                                        \
  GET_HOST_FUNC(NAME)                                                          \
  GET_CUDA_DEVICE_FUNC(NAME)
#define CUDA_HOST_DEVICE_FUNC_AS_CALLABLE(NAME)                                \
  HOST_FUNC_AS_CALLABLE(NAME);                                                 \
  CUDA_DEVICE_FUNC_AS_CALLABLE(NAME)
#else
#define GET_CUDA_HOST_DEVICE_FUNC GET_HOST_FUNC
#define CUDA_HOST_DEVICE_FUNC_AS_CALLABLE HOST_FUNC_AS_CALLABLE
#endif

int func0(int x, int y) { return 2 * (x - y); }

GET_HOST_FUNC(func0)
HOST_FUNC_AS_CALLABLE(func0);

#ifdef __CUDACC__

__device__ double func1(double x, int y) { return x + 2.75 * y; }

GET_CUDA_DEVICE_FUNC(func1)
CUDA_DEVICE_FUNC_AS_CALLABLE(func1);

#endif

DYND_CUDA_HOST_DEVICE float func2(const float (&x)[3])
{
  return x[0] + x[1] + x[2];
}

GET_CUDA_HOST_DEVICE_FUNC(func2)
CUDA_HOST_DEVICE_FUNC_AS_CALLABLE(func2);

DYND_CUDA_HOST_DEVICE unsigned int func3() { return 12U; }

GET_CUDA_HOST_DEVICE_FUNC(func3)
CUDA_HOST_DEVICE_FUNC_AS_CALLABLE(func3);

DYND_CUDA_HOST_DEVICE double func4(const double (&x)[3], const double (&y)[3])
{
  return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

GET_CUDA_HOST_DEVICE_FUNC(func4);
CUDA_HOST_DEVICE_FUNC_AS_CALLABLE(func4);

DYND_CUDA_HOST_DEVICE long func5(const long (&x)[2][3])
{
  return x[0][0] + x[0][1] + x[1][2];
}

GET_CUDA_HOST_DEVICE_FUNC(func5);
CUDA_HOST_DEVICE_FUNC_AS_CALLABLE(func5);

DYND_CUDA_HOST_DEVICE int func6(int x, int y, int z) { return x * y - z; }

GET_CUDA_HOST_DEVICE_FUNC(func6);
CUDA_HOST_DEVICE_FUNC_AS_CALLABLE(func6);

DYND_CUDA_HOST_DEVICE double func7(int x, int y, double z)
{
  return (x % y) * z;
}

GET_CUDA_HOST_DEVICE_FUNC(func7);
CUDA_HOST_DEVICE_FUNC_AS_CALLABLE(func7);

template <kernel_request_t kernreq>
class callable0 {
public:
  DYND_CUDA_HOST_DEVICE double operator()(double x) const { return 10 * x; }
};

template <kernel_request_t kernreq>
class callable1 {
  int m_x, m_y;

public:
  DYND_CUDA_HOST_DEVICE callable1(int x, int y) : m_x(x + 2), m_y(y + 3) {}
  DYND_CUDA_HOST_DEVICE int operator()(int z) const { return m_x * m_y - z; }
};

template <kernel_request_t kernreq>
class callable2 {
  int m_z;

public:
  DYND_CUDA_HOST_DEVICE callable2(int z = 7) : m_z(z) {}
  DYND_CUDA_HOST_DEVICE int operator()(int x, int y) const
  {
    return 2 * (x - y) + m_z;
  }
};

#undef GET_HOST_FUNC
#undef HOST_FUNC_AS_CALLABLE

#ifdef __CUDACC__

#undef GET_CUDA_DEVICE_FUNC
#undef CUDA_DEVICE_FUNC_AS_CALLABLE

#endif

#undef GET_CUDA_HOST_DEVICE_FUNC
#undef CUDA_HOST_DEVICE_FUNC_AS_CALLABLE

TEST(Apply, Function)
{
  typedef Apply<HostKernelRequest> TestFixture;

  nd::callable af;

  af = nd::functional::apply<kernel_request_host, decltype(&func0), &func0>();
  EXPECT_ARRAY_EQ(TestFixture::To(4), af(TestFixture::To(5), TestFixture::To(3)));

  af = nd::functional::apply<kernel_request_host, decltype(&func2), &func2>();
  EXPECT_ARRAY_EQ(TestFixture::To(13.6f),
                af(TestFixture::To({3.9f, -7.0f, 16.7f})));

  af = nd::functional::apply<kernel_request_host, decltype(&func3), &func3>();
  EXPECT_ARRAY_EQ(TestFixture::To(12U), af());

  /*
  af = nd::functional::apply<kernel_request_host, decltype(&func4), &func4>();
  std::cout << af.get_array_type() << std::endl;
  EXPECT_ARRAY_EQ(TestFixture::To(166.765), af(nd::array({9.14, -2.7, 15.32}),
                                             nd::array({0.0, 0.65, 11.0})));

    af = nd::functional::apply<kernel_request_host,
    decltype(&func5), &func5>();
    EXPECT_ARRAY_EQ(TestFixture::To(1251L), af(TestFixture::To({{1242L, 23L, -5L},
    {925L, -836L, -14L}})));
    */

  af = nd::functional::apply<kernel_request_host, decltype(&func6), &func6>();
  EXPECT_ARRAY_EQ(TestFixture::To(8),
                af(TestFixture::To(3), TestFixture::To(5), TestFixture::To(7)));

  af = nd::functional::apply<kernel_request_host, decltype(&func7), &func7>();
  EXPECT_ARRAY_EQ(
      TestFixture::To(36.3),
      af(TestFixture::To(38), TestFixture::To(5), TestFixture::To(12.1)));
}

TEST(Apply, FunctionWithKeywords)
{
  typedef Apply<HostKernelRequest> TestFixture;

  nd::callable af;

  af = nd::functional::apply<decltype(&func0), &func0>("y");
  EXPECT_ARRAY_EQ(TestFixture::To(4),
                af(TestFixture::To(5), kwds("y", TestFixture::To(3))));

  af = nd::functional::apply<decltype(&func0), &func0>("x", "y");
  EXPECT_ARRAY_EQ(TestFixture::To(4),
                af(kwds("x", TestFixture::To(5), "y", TestFixture::To(3))));

  af = nd::functional::apply<decltype(&func6), &func6>("z");
  EXPECT_ARRAY_EQ(TestFixture::To(8), af(TestFixture::To(3), TestFixture::To(5),
                                       kwds("z", TestFixture::To(7))));

  af = nd::functional::apply<decltype(&func6), &func6>("y", "z");
  EXPECT_ARRAY_EQ(TestFixture::To(8),
                af(TestFixture::To(3),
                   kwds("y", TestFixture::To(5), "z", TestFixture::To(7))));

  af = nd::functional::apply<decltype(&func6), &func6>("x", "y", "z");
  EXPECT_ARRAY_EQ(TestFixture::To(8),
                af(kwds("x", TestFixture::To(3), "y", TestFixture::To(5), "z",
                        TestFixture::To(7))));

  af = nd::functional::apply<decltype(&func7), &func7>("z");
  EXPECT_ARRAY_EQ(TestFixture::To(36.3),
                af(TestFixture::To(38), TestFixture::To(5),
                   kwds("z", TestFixture::To(12.1))));

  af = nd::functional::apply<decltype(&func7), &func7>("y", "z");
  EXPECT_ARRAY_EQ(TestFixture::To(36.3),
                af(TestFixture::To(38),
                   kwds("y", TestFixture::To(5), "z", TestFixture::To(12.1))));

  af = nd::functional::apply<decltype(&func7), &func7>("x", "y", "z");
  EXPECT_ARRAY_EQ(TestFixture::To(36.3),
                af(kwds("x", TestFixture::To(38), "y", TestFixture::To(5), "z",
                        TestFixture::To(12.1))));
}

struct struct0 {
  int func0(int x, int y, int z) { return x + y * z; }
};

TEST(Apply, MemberFunction)
{
  struct0 *s0 = new struct0;

  nd::callable af = nd::functional::apply(s0, &struct0::func0);
  EXPECT_ARRAY_EQ(nd::array(7), af(1, 2, 3));

  delete s0;
}

TYPED_TEST_P(Apply, Callable)
{
  nd::callable af;

  if (TestFixture::KernelRequest == kernel_request_host) {
    af = nd::functional::apply<kernel_request_host>(
        get_func0<kernel_request_host>());
    EXPECT_ARRAY_EQ(TestFixture::To(4),
                  af(TestFixture::To(5), TestFixture::To(3)));

    af = nd::functional::apply<kernel_request_host>(
        func0_as_callable<kernel_request_host>());
    EXPECT_ARRAY_EQ(TestFixture::To(4),
                  af(TestFixture::To(5), TestFixture::To(3)));

    af = nd::functional::apply<kernel_request_host,
                               func0_as_callable<kernel_request_host>>();
    EXPECT_ARRAY_EQ(TestFixture::To(4),
                  af(TestFixture::To(5), TestFixture::To(3)));
  }

#ifdef __CUDACC__
  if (TestFixture::KernelRequest == kernel_request_cuda_device) {
    af = nd::functional::apply<kernel_request_cuda_device>(
        get_func1<kernel_request_cuda_device>());
    EXPECT_ARRAY_EQ(TestFixture::To(58.25),
                  af(TestFixture::To(3.25), TestFixture::To(20)));

    af = nd::functional::apply<kernel_request_cuda_device>(
        func1_as_callable<kernel_request_cuda_device>());
    EXPECT_ARRAY_EQ(TestFixture::To(58.25),
                  af(TestFixture::To(3.25), TestFixture::To(20)));

    af = nd::functional::apply<kernel_request_cuda_device,
                               func1_as_callable<kernel_request_cuda_device>>();
    EXPECT_ARRAY_EQ(TestFixture::To(58.25),
                  af(TestFixture::To(3.25), TestFixture::To(20)));
  }
#endif

  af = nd::functional::apply<TestFixture::KernelRequest>(
      get_func2<TestFixture::KernelRequest>());
  EXPECT_ARRAY_EQ(TestFixture::To(13.6f),
                af(TestFixture::To({3.9f, -7.0f, 16.7f})));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      func2_as_callable<TestFixture::KernelRequest>());
  EXPECT_ARRAY_EQ(TestFixture::To(13.6f),
                af(TestFixture::To({3.9f, -7.0f, 16.7f})));

  af = nd::functional::apply<TestFixture::KernelRequest,
                             func2_as_callable<TestFixture::KernelRequest>>();
  EXPECT_ARRAY_EQ(TestFixture::To(13.6f),
                af(TestFixture::To({3.9f, -7.0f, 16.7f})));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      get_func3<TestFixture::KernelRequest>());
  EXPECT_ARRAY_EQ(TestFixture::To(12U), af());

  af = nd::functional::apply<TestFixture::KernelRequest>(
      func3_as_callable<TestFixture::KernelRequest>());
  EXPECT_ARRAY_EQ(TestFixture::To(12U), af());

  af = nd::functional::apply<TestFixture::KernelRequest,
                             func3_as_callable<TestFixture::KernelRequest>>();
  EXPECT_ARRAY_EQ(TestFixture::To(12U), af());

  af = nd::functional::apply<TestFixture::KernelRequest>(
      get_func4<TestFixture::KernelRequest>());
  EXPECT_ARRAY_EQ(TestFixture::To(167.451),
                af(TestFixture::To({9.25, -2.7, 15.375}),
                   TestFixture::To({0.0, 0.62, 11.0})));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      func4_as_callable<TestFixture::KernelRequest>());
  EXPECT_ARRAY_EQ(TestFixture::To(167.451),
                af(TestFixture::To({9.25, -2.7, 15.375}),
                   TestFixture::To({0.0, 0.62, 11.0})));

  af = nd::functional::apply<TestFixture::KernelRequest,
                             func4_as_callable<TestFixture::KernelRequest>>();
  EXPECT_ARRAY_EQ(TestFixture::To(167.451),
                af(TestFixture::To({9.25, -2.7, 15.375}),
                   TestFixture::To({0.0, 0.62, 11.0})));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      get_func6<TestFixture::KernelRequest>());
  EXPECT_ARRAY_EQ(TestFixture::To(8),
                af(TestFixture::To(3), TestFixture::To(5), TestFixture::To(7)));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      func6_as_callable<TestFixture::KernelRequest>());
  EXPECT_ARRAY_EQ(TestFixture::To(8),
                af(TestFixture::To(3), TestFixture::To(5), TestFixture::To(7)));

  af = nd::functional::apply<TestFixture::KernelRequest,
                             func6_as_callable<TestFixture::KernelRequest>>();
  EXPECT_ARRAY_EQ(TestFixture::To(8),
                af(TestFixture::To(3), TestFixture::To(5), TestFixture::To(7)));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      get_func7<TestFixture::KernelRequest>());
  EXPECT_ARRAY_EQ(
      TestFixture::To(36.3),
      af(TestFixture::To(38), TestFixture::To(5), TestFixture::To(12.1)));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      func7_as_callable<TestFixture::KernelRequest>());
  EXPECT_ARRAY_EQ(
      TestFixture::To(36.3),
      af(TestFixture::To(38), TestFixture::To(5), TestFixture::To(12.1)));

  af = nd::functional::apply<TestFixture::KernelRequest,
                             func7_as_callable<TestFixture::KernelRequest>>();
  EXPECT_ARRAY_EQ(
      TestFixture::To(36.3),
      af(TestFixture::To(38), TestFixture::To(5), TestFixture::To(12.1)));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      callable0<TestFixture::KernelRequest>());
  EXPECT_ARRAY_EQ(TestFixture::To(475.0), af(TestFixture::To(47.5)));

  af = nd::functional::apply<TestFixture::KernelRequest,
                             callable0<TestFixture::KernelRequest>>();
  EXPECT_ARRAY_EQ(TestFixture::To(475.0), af(TestFixture::To(47.5)));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      callable2<TestFixture::KernelRequest>());
  EXPECT_ARRAY_EQ(TestFixture::To(11),
                af(TestFixture::To(5), TestFixture::To(3)));

  af = nd::functional::apply<TestFixture::KernelRequest,
                             callable2<TestFixture::KernelRequest>>();
  EXPECT_ARRAY_EQ(TestFixture::To(11),
                af(TestFixture::To(5), TestFixture::To(3)));
}

TYPED_TEST_P(Apply, CallableWithKeywords)
{
  nd::callable af;

  if (TestFixture::KernelRequest == kernel_request_host) {
    af = nd::functional::apply<kernel_request_host>(
        get_func0<kernel_request_host>(), "y");
    EXPECT_ARRAY_EQ(TestFixture::To(4),
                  af(TestFixture::To(5), kwds("y", TestFixture::To(3))));

    af = nd::functional::apply<kernel_request_host>(
        get_func0<kernel_request_host>(), "x", "y");
    EXPECT_ARRAY_EQ(TestFixture::To(4),
                  af(kwds("x", TestFixture::To(5), "y", TestFixture::To(3))));

    af = nd::functional::apply(func0_as_callable<kernel_request_host>(), "y");
    EXPECT_ARRAY_EQ(TestFixture::To(4), af(5, kwds("y", TestFixture::To(3))));

    af = nd::functional::apply(func0_as_callable<kernel_request_host>(), "x",
                               "y");
    EXPECT_ARRAY_EQ(TestFixture::To(4),
                  af(kwds("x", TestFixture::To(5), "y", TestFixture::To(3))));
  }

#ifdef __CUDACC__
  if (TestFixture::KernelRequest == kernel_request_cuda_device) {
    af = nd::functional::apply<kernel_request_cuda_device>(
        get_func1<kernel_request_cuda_device>(), "y");
    EXPECT_ARRAY_EQ(TestFixture::To(58.25),
                  af(TestFixture::To(3.25), kwds("y", TestFixture::To(20))));

    af = nd::functional::apply<kernel_request_cuda_device>(
        get_func1<kernel_request_cuda_device>(), "x", "y");
    EXPECT_ARRAY_EQ(TestFixture::To(58.25), af(kwds("x", TestFixture::To(3.25),
                                                  "y", TestFixture::To(20))));

    af = nd::functional::apply<kernel_request_cuda_device>(
        func1_as_callable<kernel_request_cuda_device>(), "y");
    EXPECT_ARRAY_EQ(TestFixture::To(58.25),
                  af(TestFixture::To(3.25), kwds("y", TestFixture::To(20))));

    af = nd::functional::apply<kernel_request_cuda_device>(
        func1_as_callable<kernel_request_cuda_device>(), "x", "y");
    EXPECT_ARRAY_EQ(TestFixture::To(58.25), af(kwds("x", TestFixture::To(3.25),
                                                  "y", TestFixture::To(20))));
  }
#endif

  af = nd::functional::apply<TestFixture::KernelRequest>(
      get_func6<TestFixture::KernelRequest>(), "z");
  EXPECT_ARRAY_EQ(TestFixture::To(8), af(TestFixture::To(3), TestFixture::To(5),
                                       kwds("z", TestFixture::To(7))));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      get_func6<TestFixture::KernelRequest>(), "y", "z");
  EXPECT_ARRAY_EQ(TestFixture::To(8),
                af(TestFixture::To(3),
                   kwds("y", TestFixture::To(5), "z", TestFixture::To(7))));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      get_func6<TestFixture::KernelRequest>(), "x", "y", "z");
  EXPECT_ARRAY_EQ(TestFixture::To(8),
                af(kwds("x", TestFixture::To(3), "y", TestFixture::To(5), "z",
                        TestFixture::To(7))));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      func6_as_callable<TestFixture::KernelRequest>(), "z");
  EXPECT_ARRAY_EQ(TestFixture::To(8), af(TestFixture::To(3), TestFixture::To(5),
                                       kwds("z", TestFixture::To(7))));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      func6_as_callable<TestFixture::KernelRequest>(), "y", "z");
  EXPECT_ARRAY_EQ(TestFixture::To(8),
                af(TestFixture::To(3),
                   kwds("y", TestFixture::To(5), "z", TestFixture::To(7))));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      func6_as_callable<TestFixture::KernelRequest>(), "x", "y", "z");
  EXPECT_ARRAY_EQ(TestFixture::To(8),
                af(kwds("x", TestFixture::To(3), "y", TestFixture::To(5), "z",
                        TestFixture::To(7))));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      get_func7<TestFixture::KernelRequest>(), "z");
  EXPECT_ARRAY_EQ(TestFixture::To(36.3),
                af(TestFixture::To(38), TestFixture::To(5),
                   kwds("z", TestFixture::To(12.1))));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      get_func7<TestFixture::KernelRequest>(), "y", "z");
  EXPECT_ARRAY_EQ(TestFixture::To(36.3),
                af(TestFixture::To(38),
                   kwds("y", TestFixture::To(5), "z", TestFixture::To(12.1))));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      get_func7<TestFixture::KernelRequest>(), "x", "y", "z");
  EXPECT_ARRAY_EQ(TestFixture::To(36.3),
                af(kwds("x", TestFixture::To(38), "y", TestFixture::To(5), "z",
                        TestFixture::To(12.1))));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      func7_as_callable<TestFixture::KernelRequest>(), "z");
  EXPECT_ARRAY_EQ(TestFixture::To(36.3),
                af(TestFixture::To(38), TestFixture::To(5),
                   kwds("z", TestFixture::To(12.1))));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      func7_as_callable<TestFixture::KernelRequest>(), "y", "z");
  EXPECT_ARRAY_EQ(TestFixture::To(36.3),
                af(TestFixture::To(38),
                   kwds("y", TestFixture::To(5), "z", TestFixture::To(12.1))));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      func7_as_callable<TestFixture::KernelRequest>(), "x", "y", "z");
  EXPECT_ARRAY_EQ(TestFixture::To(36.3),
                af(kwds("x", TestFixture::To(38), "y", TestFixture::To(5), "z",
                        TestFixture::To(12.1))));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      callable0<TestFixture::KernelRequest>(), "x");
  EXPECT_ARRAY_EQ(TestFixture::To(475.0), af(kwds("x", TestFixture::To(47.5))));

  af = nd::functional::apply<TestFixture::KernelRequest,
                             callable1<TestFixture::KernelRequest>, int, int>(
      "x", "y");
  EXPECT_ARRAY_EQ(TestFixture::To(28),
                af(TestFixture::To(2),
                   kwds("x", TestFixture::To(1), "y", TestFixture::To(7))));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      callable2<TestFixture::KernelRequest>(), "y");
  EXPECT_ARRAY_EQ(TestFixture::To(11),
                af(TestFixture::To(5), kwds("y", TestFixture::To(3))));

  af = nd::functional::apply<TestFixture::KernelRequest>(
      callable2<TestFixture::KernelRequest>(), "x", "y");
  EXPECT_ARRAY_EQ(TestFixture::To(11),
                af(kwds("x", TestFixture::To(5), "y", TestFixture::To(3))));

  af = nd::functional::apply<TestFixture::KernelRequest,
                             callable2<TestFixture::KernelRequest>, int>("z");
  EXPECT_ARRAY_EQ(TestFixture::To(8), af(TestFixture::To(5), TestFixture::To(3),
                                       kwds("z", TestFixture::To(4))));
}

REGISTER_TYPED_TEST_CASE_P(Apply, Callable, CallableWithKeywords);

INSTANTIATE_TYPED_TEST_CASE_P(HostMemory, Apply, HostKernelRequest);
#ifdef DYND_CUDA
INSTANTIATE_TYPED_TEST_CASE_P(CUDADeviceMemory, Apply, CUDADeviceKernelRequest);
#endif
