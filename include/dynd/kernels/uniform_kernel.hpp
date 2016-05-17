//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <memory>
#include <random>

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {

/*
template <bool Condition, typename Type = int>
struct RequiresC_F {
  using type = Type;
};

template <typename Type>
struct RequiresC_F<false, Type> {};

template <bool Condition, typename Type>
using RequiresC = typename RequiresC_F<Condition, Type>::type;

template <typename Condition, typename Type = int>
using Requires = RequiresC<Condition::value, Type>;
*/

template <bool Value>
using Requires = std::enable_if_t<Value>;

inline std::shared_ptr<std::default_random_engine> &get_random_device() {
  static std::random_device random_device;
  static std::shared_ptr<std::default_random_engine> g(new std::default_random_engine(random_device()));

  return g;
}

namespace nd {
  namespace random {

    template <typename ReturnType, typename GeneratorType, typename Enable = void>
    struct uniform_kernel;

    template <typename ReturnType, typename GeneratorType>
    struct uniform_kernel<ReturnType, GeneratorType, Requires<is_integral<ReturnType>::value>>
        : base_strided_kernel<uniform_kernel<ReturnType, GeneratorType>, 0> {
      GeneratorType &g;
      std::uniform_int_distribution<ReturnType> d;

      uniform_kernel(GeneratorType *g, ReturnType a, ReturnType b) : g(*g), d(a, b) {}

      void single(char *dst, char *const *DYND_UNUSED(src)) { *reinterpret_cast<ReturnType *>(dst) = d(g); }
    };

    template <typename ReturnType, typename GeneratorType>
    struct uniform_kernel<ReturnType, GeneratorType, Requires<is_floating_point<ReturnType>::value>>
        : base_strided_kernel<uniform_kernel<ReturnType, GeneratorType>, 0> {
      GeneratorType &g;
      std::uniform_real_distribution<ReturnType> d;

      uniform_kernel(GeneratorType *g, ReturnType a, ReturnType b) : g(*g), d(a, b) {}

      void single(char *dst, char *const *DYND_UNUSED(src)) { *reinterpret_cast<ReturnType *>(dst) = d(g); }
    };

    template <typename ReturnType, typename GeneratorType>
    struct uniform_kernel<ReturnType, GeneratorType, Requires<is_complex<ReturnType>::value>>
        : base_strided_kernel<uniform_kernel<ReturnType, GeneratorType>, 0> {
      GeneratorType &g;
      std::uniform_real_distribution<typename ReturnType::value_type> d_real;
      std::uniform_real_distribution<typename ReturnType::value_type> d_imag;

      uniform_kernel(GeneratorType *g, ReturnType a, ReturnType b)
          : g(*g), d_real(a.real(), b.real()), d_imag(a.imag(), b.imag()) {}

      void single(char *dst, char *const *DYND_UNUSED(src)) {
        *reinterpret_cast<ReturnType *>(dst) = ReturnType(d_real(g), d_imag(g));
      }
    };

  } // namespace dynd::nd::random
} // namespace dynd::nd
} // namespace dynd
