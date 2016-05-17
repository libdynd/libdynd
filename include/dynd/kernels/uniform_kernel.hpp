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

inline std::shared_ptr<std::default_random_engine> &get_random_device() {
  static std::random_device random_device;
  static std::shared_ptr<std::default_random_engine> g(new std::default_random_engine(random_device()));

  return g;
}

namespace nd {
  namespace random {
    namespace detail {

      template <typename ReturnType, typename ReturnBaseType, typename GeneratorType>
      struct uniform_kernel;

      template <typename ReturnType, typename GeneratorType>
      struct uniform_kernel<ReturnType, ndt::int_kind_type, GeneratorType>
          : base_strided_kernel<uniform_kernel<ReturnType, ndt::int_kind_type, GeneratorType>, 0> {
        GeneratorType &g;
        std::uniform_int_distribution<ReturnType> d;

        uniform_kernel(GeneratorType *g, ReturnType a, ReturnType b) : g(*g), d(a, b) {}

        void single(char *dst, char *const *DYND_UNUSED(src)) { *reinterpret_cast<ReturnType *>(dst) = d(g); }
      };

      template <typename ReturnType, typename GeneratorType>
      struct uniform_kernel<ReturnType, ndt::uint_kind_type, GeneratorType>
          : uniform_kernel<ReturnType, ndt::int_kind_type, GeneratorType> {};

      template <typename ReturnType, typename GeneratorType>
      struct uniform_kernel<ReturnType, ndt::float_kind_type, GeneratorType>
          : base_strided_kernel<uniform_kernel<ReturnType, ndt::float_kind_type, GeneratorType>, 0> {
        GeneratorType &g;
        std::uniform_real_distribution<ReturnType> d;

        uniform_kernel(GeneratorType *g, ReturnType a, ReturnType b) : g(*g), d(a, b) {}

        void single(char *dst, char *const *DYND_UNUSED(src)) { *reinterpret_cast<ReturnType *>(dst) = d(g); }
      };

      template <typename ReturnType, typename GeneratorType>
      struct uniform_kernel<ReturnType, ndt::complex_kind_type, GeneratorType>
          : base_strided_kernel<uniform_kernel<ReturnType, ndt::complex_kind_type, GeneratorType>, 0> {
        GeneratorType &g;
        std::uniform_real_distribution<typename ReturnType::value_type> d_real;
        std::uniform_real_distribution<typename ReturnType::value_type> d_imag;

        uniform_kernel(GeneratorType *g, ReturnType a, ReturnType b)
            : g(*g), d_real(a.real(), b.real()), d_imag(a.imag(), b.imag()) {}

        void single(char *dst, char *const *DYND_UNUSED(src)) {
          *reinterpret_cast<ReturnType *>(dst) = ReturnType(d_real(g), d_imag(g));
        }
      };

    } // namespace dynd::nd::random::detail

    template <typename ReturnType, typename GeneratorType>
    using uniform_kernel = detail::uniform_kernel<ReturnType, ndt::base_of_t<ReturnType>, GeneratorType>;

  } // namespace dynd::nd::random
} // namespace dynd::nd
} // namespace dynd
