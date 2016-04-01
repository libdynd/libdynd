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

inline std::shared_ptr<std::default_random_engine> &get_random_device()
{
  static std::random_device random_device;
  static std::shared_ptr<std::default_random_engine> g(new std::default_random_engine(random_device()));

  return g;
}

namespace nd {
  namespace random {
    namespace detail {

      template <type_id_t ResID, type_id_t ResBaseID, typename GeneratorType>
      struct uniform_kernel;

      template <type_id_t ResID, typename GeneratorType>
      struct uniform_kernel<ResID, int_kind_id, GeneratorType>
          : base_strided_kernel<uniform_kernel<ResID, int_kind_id, GeneratorType>, 0> {
        typedef typename type_of<ResID>::type R;

        GeneratorType &g;
        std::uniform_int_distribution<R> d;

        uniform_kernel(GeneratorType *g, R a, R b) : g(*g), d(a, b) {}

        void single(char *dst, char *const *DYND_UNUSED(src)) { *reinterpret_cast<R *>(dst) = d(g); }
      };

      template <type_id_t ResID, typename GeneratorType>
      struct uniform_kernel<ResID, uint_kind_id, GeneratorType> : uniform_kernel<ResID, int_kind_id, GeneratorType> {
      };

      template <type_id_t ResID, typename GeneratorType>
      struct uniform_kernel<ResID, float_kind_id, GeneratorType>
          : base_strided_kernel<uniform_kernel<ResID, float_kind_id, GeneratorType>, 0> {
        typedef typename type_of<ResID>::type R;

        GeneratorType &g;
        std::uniform_real_distribution<R> d;

        uniform_kernel(GeneratorType *g, R a, R b) : g(*g), d(a, b) {}

        void single(char *dst, char *const *DYND_UNUSED(src)) { *reinterpret_cast<R *>(dst) = d(g); }
      };

      template <type_id_t ResID, typename GeneratorType>
      struct uniform_kernel<ResID, complex_kind_id, GeneratorType>
          : base_strided_kernel<uniform_kernel<ResID, complex_kind_id, GeneratorType>, 0> {
        typedef typename type_of<ResID>::type R;

        GeneratorType &g;
        std::uniform_real_distribution<typename R::value_type> d_real;
        std::uniform_real_distribution<typename R::value_type> d_imag;

        uniform_kernel(GeneratorType *g, R a, R b) : g(*g), d_real(a.real(), b.real()), d_imag(a.imag(), b.imag()) {}

        void single(char *dst, char *const *DYND_UNUSED(src)) { *reinterpret_cast<R *>(dst) = R(d_real(g), d_imag(g)); }
      };

    } // namespace dynd::nd::random::detail

    template <type_id_t ResID, typename GeneratorType>
    using uniform_kernel = detail::uniform_kernel<ResID, base_id_of<ResID>::value, GeneratorType>;

  } // namespace dynd::nd::random
} // namespace dynd::nd
} // namespace dynd
