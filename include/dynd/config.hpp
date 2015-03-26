//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/cmake_config.hpp>

#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <stdlib.h>
#include <initializer_list>
#include <limits>
#include <stdint.h>
#include <tuple>
#include <type_traits>
#include <utility>

#ifdef __NVCC__
#ifndef DYND_CUDA
#define DYND_CUDA
#endif
#endif

#ifdef DYND_CUDA
#include <cuda_runtime.h>
#endif

/** The number of elements to process at once when doing chunking/buffering */
#define DYND_BUFFER_CHUNK_SIZE 128

#ifdef __clang__

#if __has_feature(cxx_constexpr)
#define DYND_CONSTEXPR constexpr
#else
#define DYND_CONSTEXPR
#endif

#include <cmath>
#include <cctype>

// Workaround for a clang issue
#define DYND_ISSPACE std::isspace
#define DYND_TOLOWER std::tolower

#elif defined(__GNUC__)

// Hack trying to work around gcc isfinite problems
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#define DYND_IGNORE_UNUSED(NAME) NAME __attribute__((unused))

#define DYND_CONSTEXPR constexpr

#define DYND_ISSPACE isspace
#define DYND_TOLOWER tolower

#elif defined(_MSC_VER)

#define DYND_ISSPACE isspace
#define DYND_TOLOWER tolower

#include <float.h>

// If set, uses the FP status registers.
// On some compilers, there is no proper
// way to tell the compiler that these are
// important, and it reorders instructions
// so as to make them useless. On MSVC, there
// is #pragma fenv_access(on), which works.
//
// Update 2014-11-29: Found on 32-bit Windows, MSVC 2013 does not respect
//                    pragma fenv_access(on), and reorders incorrectly, so
//                    disabling this on 32-bit.
#ifdef _WIN64
#define DYND_USE_FPSTATUS
#endif

#define DYND_USE_STD_ATOMIC

#if _MSC_VER == 1800
// MSVC 2013 doesn't support nested initializer lists
// https://stackoverflow.com/questions/23965565/how-to-do-nested-initializer-lists-in-visual-c-2013
#define DYND_NESTED_INIT_LIST_BUG
#endif

// No DYND_CONSTEXPR yet, define it as nothing
#define DYND_CONSTEXPR

// Some C library calls will abort if given bad format strings, etc
// This RAII class temporarily disables that
class disable_invalid_parameter_handler {
  _invalid_parameter_handler m_saved;

  disable_invalid_parameter_handler(const disable_invalid_parameter_handler &);
  disable_invalid_parameter_handler &
  operator=(const disable_invalid_parameter_handler &);

  static void nop_parameter_handler(const wchar_t *, const wchar_t *,
                                    const wchar_t *, unsigned int, uintptr_t)
  {
  }

public:
  disable_invalid_parameter_handler()
  {
    m_saved = _set_invalid_parameter_handler(&nop_parameter_handler);
  }
  ~disable_invalid_parameter_handler()
  {
    _set_invalid_parameter_handler(m_saved);
  }
};

#endif // end of compiler vendor checks

#ifdef __CUDACC__
#define DYND_CUDA_HOST_DEVICE __host__ __device__
#else
#define DYND_CUDA_HOST_DEVICE
#endif

// If being run from the CLING C++ interpreter
#ifdef DYND_CLING
// Don't use the memcpy function (it has inline assembly).

inline void DYND_MEMCPY(char *dst, const char *src, intptr_t count)
{
  char *cdst = (char *)dst;
  const char *csrc = (const char *)src;
  while (count--) {
    *cdst++ = *csrc++;
  }
}
#else
#include <cstring>
#define DYND_MEMCPY(dst, src, count) std::memcpy(dst, src, count)
#endif

#include <dynd/type_sequence.hpp>

// These are small templates 'missing' from the standard library
namespace dynd {

template <typename T>
struct is_function_pointer {
  static const bool value =
      std::is_pointer<T>::value
          ? std::is_function<typename std::remove_pointer<T>::type>::value
          : false;
};

/**
 * Matches string literal arguments.
 */
template <typename T>
struct is_char_string_param {
  static const bool value = false;
};
template <>
struct is_char_string_param<const char *> {
  static const bool value = true;
};
template <>
struct is_char_string_param<char *> {
  static const bool value = true;
};
template <int N>
struct is_char_string_param<const char (&)[N]> {
  static const bool value = true;
};
template <int N>
struct is_char_string_param<const char (&&)[N]> {
  static const bool value = true;
};

/** Returns true if all the packed parameters are char strings */
template <typename... T>
struct all_char_string_params {
  static const bool value = false;
};
template <>
struct all_char_string_params<> {
  static const bool value = true;
};
template <typename T0>
struct all_char_string_params<T0> {
  static const bool value = is_char_string_param<T0>::value;
};
template <typename... T, typename T0>
struct all_char_string_params<T0, T...> {
  static const bool value =
      is_char_string_param<T0>::value && all_char_string_params<T...>::value;
};

template <typename T>
struct remove_all_pointers {
  typedef T type;
};

template <typename T>
struct remove_all_pointers<T *> {
  typedef typename remove_all_pointers<typename std::remove_cv<T>::type>::type
      type;
};

template <typename T, typename U>
struct as_ {
  typedef U type;
};

template <template <typename...> class C, typename T>
struct instantiate;

template <template <typename...> class C, typename... T>
struct instantiate<C, type_sequence<T...>> {
  typedef C<T...> type;
};

template <typename I>
struct integer_proxy;

template <typename T>
struct integer_proxy<integer_sequence<T>> {
  enum { size = 0 };

  template <typename... U, typename F, typename... A>
  static void for_each(F, A &&...)
  {
  }

  template <typename R, typename... A>
  static R make(A &&...)
  {
    return R();
  }
};

template <typename T, T I0>
struct integer_proxy<integer_sequence<T, I0>> {
  enum { size = 1 };

  template <typename... U, typename F, typename... A>
  static void for_each(F f, A &&... a)
  {
    f.template on_each<I0, U...>(std::forward<A>(a)...);
  }

  template <typename R, typename... A>
  static R make(A &&... a)
  {
    return R(get<I0>(std::forward<A>(a)...));
  }
};


template <typename T, T I0, T... I>
struct integer_proxy<integer_sequence<T, I0, I...>> {
  enum { size = dynd::integer_sequence<T, I0, I...>::size };

#if !(defined(_MSC_VER) && (_MSC_VER == 1800))
  template <typename R, typename... A>
  static R make(A &&... a)
  {
    return R(get<I0>(std::forward<A>(a)...), get<I>(std::forward<A>(a)...)...);
  }
#else
  // Workaround for MSVC 2013 compiler bug reported here:
  // https://connect.microsoft.com/VisualStudio/feedback/details/1045260/unpacking-std-forward-a-a-fails-when-nested-with-another-unpacking
  template <typename R>
  static R make()
  {
    return R();
  }
  template <typename R, typename A0>
  static R make(A0 &&a0)
  {
    return R(get<I0>(std::forward<A0>(a0)));
  }
  template <typename R, typename A0, typename A1>
  static R make(A0 &&a0, A1 &&a1)
  {
    return R(get<I0>(std::forward<A0>(a0), std::forward<A1>(a1)),
             get<I>(std::forward<A0>(a0), std::forward<A1>(a1))...);
  }
  template <typename R, typename A0, typename A1, typename A2>
  static R make(A0 &&a0, A1 &&a1, A2 &&a2)
  {
    return R(get<I0>(std::forward<A0>(a0), std::forward<A1>(a1),
                     std::forward<A2>(a2)),
             get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                    std::forward<A2>(a2))...);
  }
  template <typename R, typename A0, typename A1, typename A2, typename A3>
  static R make(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3)
  {
    return R(get<I0>(std::forward<A0>(a0), std::forward<A1>(a1),
                     std::forward<A2>(a2), std::forward<A3>(a3)),
             get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                    std::forward<A2>(a2), std::forward<A3>(a3))...);
  }
  template <typename R, typename A0, typename A1, typename A2, typename A3,
            typename A4>
  static R make(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4)
  {
    return R(get<I0>(std::forward<A0>(a0), std::forward<A1>(a1),
                     std::forward<A2>(a2), std::forward<A3>(a3),
                     std::forward<A4>(a4)),
             get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                    std::forward<A2>(a2), std::forward<A3>(a3),
                    std::forward<A4>(a4))...);
  }
  template <typename R, typename A0, typename A1, typename A2, typename A3,
            typename A4, typename A5>
  static R make(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5)
  {
    return R(get<I0>(std::forward<A0>(a0), std::forward<A1>(a1),
                     std::forward<A2>(a2), std::forward<A3>(a3),
                     std::forward<A4>(a4), std::forward<A5>(a5)),
             get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                    std::forward<A2>(a2), std::forward<A3>(a3),
                    std::forward<A4>(a4), std::forward<A5>(a5))...);
  }
  template <typename R, typename A0, typename A1, typename A2, typename A3,
            typename A4, typename A5, typename A6>
  static R make(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5, A6 &&a6)
  {
    return R(get<I0>(std::forward<A0>(a0), std::forward<A1>(a1),
                     std::forward<A2>(a2), std::forward<A3>(a3),
                     std::forward<A4>(a4), std::forward<A5>(a5)),
             get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                    std::forward<A2>(a2), std::forward<A3>(a3),
                    std::forward<A4>(a4), std::forward<A5>(a5),
                    std::forward<A6>(a6))...);
  }
#endif

  template <typename... U, typename F, typename... A>
  static void for_each(F f, A &&... a)
  {
    f.template on_each<I0, U...>(std::forward<A>(a)...);
    integer_proxy<integer_sequence<T, I...>>::template for_each<U...>(f, std::forward<A>(a)...);
  }
};

template <typename I>
using index_proxy = integer_proxy<I>;

/*
template <typename I>
struct index_proxy;

template <>
struct index_proxy<index_sequence<>> {
  enum { size = 0 };

  template <typename F, typename... A>
  static void for_each(F, A &&...)
  {
  }

  template <typename R, typename... A>
  static R make(A &&...)
  {
    return R();
  }
};

template <size_t I0>
struct index_proxy<index_sequence<I0>> {
  enum { size = 1 };

  template <typename F, typename... A>
  static void for_each(F f, A &&... a)
  {
    f.template on_each<I0>(std::forward<A>(a)...);
  }

  template <typename R, typename... A>
  static R make(A &&... a)
  {
    return R(get<I0>(std::forward<A>(a)...));
  }
};

template <size_t I0, size_t... I>
struct index_proxy<index_sequence<I0, I...>> {
  enum { size = index_sequence<I0, I...>::size };

#if !(defined(_MSC_VER) && (_MSC_VER == 1800))
  template <typename R, typename... A>
  static R make(A &&... a)
  {
    return R(get<I0>(std::forward<A>(a)...), get<I>(std::forward<A>(a)...)...);
  }
#else
  // Workaround for MSVC 2013 compiler bug reported here:
  // https://connect.microsoft.com/VisualStudio/feedback/details/1045260/unpacking-std-forward-a-a-fails-when-nested-with-another-unpacking
  template <typename R>
  static R make()
  {
    return R();
  }
  template <typename R, typename A0>
  static R make(A0 &&a0)
  {
    return R(get<I0>(std::forward<A0>(a0)));
  }
  template <typename R, typename A0, typename A1>
  static R make(A0 &&a0, A1 &&a1)
  {
    return R(get<I0>(std::forward<A0>(a0), std::forward<A1>(a1)),
             get<I>(std::forward<A0>(a0), std::forward<A1>(a1))...);
  }
  template <typename R, typename A0, typename A1, typename A2>
  static R make(A0 &&a0, A1 &&a1, A2 &&a2)
  {
    return R(get<I0>(std::forward<A0>(a0), std::forward<A1>(a1),
                     std::forward<A2>(a2)),
             get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                    std::forward<A2>(a2))...);
  }
  template <typename R, typename A0, typename A1, typename A2, typename A3>
  static R make(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3)
  {
    return R(get<I0>(std::forward<A0>(a0), std::forward<A1>(a1),
                     std::forward<A2>(a2), std::forward<A3>(a3)),
             get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                    std::forward<A2>(a2), std::forward<A3>(a3))...);
  }
  template <typename R, typename A0, typename A1, typename A2, typename A3,
            typename A4>
  static R make(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4)
  {
    return R(get<I0>(std::forward<A0>(a0), std::forward<A1>(a1),
                     std::forward<A2>(a2), std::forward<A3>(a3),
                     std::forward<A4>(a4)),
             get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                    std::forward<A2>(a2), std::forward<A3>(a3),
                    std::forward<A4>(a4))...);
  }
  template <typename R, typename A0, typename A1, typename A2, typename A3,
            typename A4, typename A5>
  static R make(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5)
  {
    return R(get<I0>(std::forward<A0>(a0), std::forward<A1>(a1),
                     std::forward<A2>(a2), std::forward<A3>(a3),
                     std::forward<A4>(a4), std::forward<A5>(a5)),
             get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                    std::forward<A2>(a2), std::forward<A3>(a3),
                    std::forward<A4>(a4), std::forward<A5>(a5))...);
  }
  template <typename R, typename A0, typename A1, typename A2, typename A3,
            typename A4, typename A5, typename A6>
  static R make(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5, A6 &&a6)
  {
    return R(get<I0>(std::forward<A0>(a0), std::forward<A1>(a1),
                     std::forward<A2>(a2), std::forward<A3>(a3),
                     std::forward<A4>(a4), std::forward<A5>(a5)),
             get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                    std::forward<A2>(a2), std::forward<A3>(a3),
                    std::forward<A4>(a4), std::forward<A5>(a5),
                    std::forward<A6>(a6))...);
  }
#endif

  template <typename F, typename... A>
  static void for_each(F f, A &&... a)
  {
    f.template on_each<I0>(std::forward<A>(a)...);
    index_proxy<index_sequence<I...>>::for_each(f, std::forward<A>(a)...);
  }
};
*/

} // namespace dynd

/////////////////////////////////////////
// Diagnostic configurations
//

/**
 * This preprocessor symbol enables or disables assertions
 * that pointers are aligned as they are supposed to be. This helps
 * test that alignment is being done correctly on platforms which
 * do not segfault on misaligned data.
 *
 * An exception is thrown if an improper unalignment is detected.
 *
 * See diagnostics.hpp for the macros which use this.
 */
#ifndef DYND_ALIGNMENT_ASSERTIONS
#define DYND_ALIGNMENT_ASSERTIONS 0
#endif

/**
 * This preprocessor symbol enables or disables tracing
 * of individual builtin type assignment operations.
 *
 * See diagnostics.hpp for the macros which use this.
 */
#ifndef DYND_ASSIGNMENT_TRACING
#define DYND_ASSIGNMENT_TRACING 0
#endif

/**
 * Preprocessor macro for marking variables unused, and suppressing
 * warnings for them.
 */
#define DYND_UNUSED(x)

#ifndef DYND_IGNORE_UNUSED
#define DYND_IGNORE_UNUSED(NAME) NAME
#endif

#define DYND_INC_IF_NOT_NULL(POINTER, OFFSET)                                  \
  ((POINTER == NULL) ? NULL : (POINTER + OFFSET))

namespace dynd {
// These are defined in git_version.cpp, generated from
// git_version.cpp.in by the CMake build configuration.
extern const char dynd_git_sha1[];
extern const char dynd_version_string[];
} // namespace dynd

// Check endian: define DYND_BIG_ENDIAN if big endian, otherwise assume little
#if defined(__GLIBC__)
#include <endian.h>
#if __BYTE_ORDER == __BIG_ENDIAN
#define DYND_BIG_ENDIAN
#elif __BYTE_ORDER != __LITTLE_ENDIAN
#error Unrecognized endianness from endian.h.
#endif
#endif

namespace dynd {

/**
 * Function to call for initializing dynd's global state, such
 * as cached ndt::type objects, the arrfunc registry, etc.
 */
inline int libdynd_init() { return 0; }

/**
 * Function to call to free all resources associated with
 * dynd's global state, that were initialized by libdynd_init.
 */
inline void libdynd_cleanup() {}

/**
  * A function which can be used at runtime to identify whether
  * the build of dynd being linked against was built with CUDA
  * support enabled.
  */
bool built_with_cuda();

} // namespace dynd

#ifdef DYND_CUDA

#define DYND_GET_CUDA_DEVICE_FUNC(NAME, FUNC)                                  \
  template <typename func_type>                                                \
  __global__ void NAME(void *res)                                              \
  {                                                                            \
    *reinterpret_cast<func_type *>(res) = static_cast<func_type>(&FUNC);       \
  }                                                                            \
                                                                               \
  template <typename func_type>                                                \
  func_type NAME()                                                             \
  {                                                                            \
    func_type func;                                                            \
    func_type *cuda_device_func;                                               \
    cuda_throw_if_not_success(                                                 \
        cudaMalloc(&cuda_device_func, sizeof(func_type)));                     \
    NAME<func_type> << <1, 1>>> (reinterpret_cast<void *>(cuda_device_func));  \
    cuda_throw_if_not_success(cudaMemcpy(                                      \
        &func, cuda_device_func, sizeof(func_type), cudaMemcpyDeviceToHost));  \
    cuda_throw_if_not_success(cudaFree(cuda_device_func));                     \
                                                                               \
    return func;                                                               \
  }

#endif

namespace dynd {
namespace detail {

  template <typename T, int N>
  class array_wrapper {
    T m_data[N];

  public:
    DYND_CUDA_HOST_DEVICE array_wrapper() = default;

    DYND_CUDA_HOST_DEVICE array_wrapper(const T *data)
    {
      memcpy(m_data, data, sizeof(m_data));
    }

    DYND_CUDA_HOST_DEVICE operator T *() { return m_data; }

    DYND_CUDA_HOST_DEVICE operator const T *() const { return m_data; }

    DYND_CUDA_HOST_DEVICE T &operator[](intptr_t i) { return m_data[i]; }

    DYND_CUDA_HOST_DEVICE const T &operator[](intptr_t i) const
    {
      return m_data[i];
    }
  };

  template <typename T>
  class array_wrapper<T, 0> {
  public:
    DYND_CUDA_HOST_DEVICE array_wrapper() = default;

    DYND_CUDA_HOST_DEVICE array_wrapper(const T *DYND_UNUSED(data)) {}

    DYND_CUDA_HOST_DEVICE operator T *() { return NULL; }

    DYND_CUDA_HOST_DEVICE operator const T *() const { return NULL; }
  };

  template <int N, typename T>
  array_wrapper<T, N> make_array_wrapper(const T *data)
  {
    return array_wrapper<T, N>(data);
  }

  template <typename T>
  class value_wrapper {
    T m_value;

  public:
    value_wrapper(const T &value) : m_value(value) {}

    DYND_CUDA_HOST_DEVICE operator T() const { return m_value; }
  };

  template <typename T>
  value_wrapper<T> make_value_wrapper(const T &value)
  {
    return value_wrapper<T>(value);
  }

#ifdef __CUDACC__

  template <intptr_t I>
  __device__ inline typename std::enable_if<I == 0, intptr_t>::type
  cuda_device_thread_id()
  {
    return blockIdx.x * blockDim.x + threadIdx.x;
  }

  template <intptr_t I>
  __device__ inline typename std::enable_if<I == 1, intptr_t>::type
  cuda_device_thread_id()
  {
    return blockIdx.y * blockDim.y + threadIdx.y;
  }

  template <intptr_t I>
  __device__ inline typename std::enable_if<I == 2, intptr_t>::type
  cuda_device_thread_id()
  {
    return blockIdx.z * blockDim.z + threadIdx.z;
  }

  template <intptr_t I>
  __device__ inline typename std::enable_if<I == -1, intptr_t>::type
  cuda_device_thread_id()
  {
    return 0;
  }

  template <intptr_t I>
  __device__ inline typename std::enable_if<I == 0, intptr_t>::type
  cuda_device_thread_count()
  {
    return gridDim.x * blockDim.x;
  }

  template <intptr_t I>
  __device__ inline typename std::enable_if<I == 1, intptr_t>::type
  cuda_device_thread_count()
  {
    return gridDim.y * blockDim.y;
  }

  template <intptr_t I>
  __device__ inline typename std::enable_if<I == 2, intptr_t>::type
  cuda_device_thread_count()
  {
    return gridDim.z * blockDim.z;
  }

  template <intptr_t I>
  __device__ inline typename std::enable_if<I == -1, intptr_t>::type
  cuda_device_thread_count()
  {
    return 1;
  }

  __device__ inline intptr_t cuda_device_thread_count()
  {
    return cuda_device_thread_count<0>() * cuda_device_thread_count<1>() *
           cuda_device_thread_count<2>();
  }

#endif

} // namespace dynd::detail
} // namespace dynd

#ifdef __CUDA_ARCH__
#define DYND_THREAD_ID(I) dynd::detail::cuda_device_thread_id<I>()
#define DYND_THREAD_COUNT(I) dynd::detail::cuda_device_thread_count<I>()
#else
#define DYND_THREAD_ID(I) 0
#define DYND_THREAD_COUNT(I) 1
#endif