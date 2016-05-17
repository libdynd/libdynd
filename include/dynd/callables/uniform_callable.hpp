//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/uniform_kernel.hpp>
#include <dynd/types/callable_type.hpp>
#include <dynd/types/option_type.hpp>

namespace dynd {
namespace nd {
  namespace random {
    namespace detail {

      template <typename ReturnType, typename ReturnBaseType, typename GeneratorType>
      class uniform_callable;

      template <typename ReturnType, typename GeneratorType>
      class uniform_callable<ReturnType, ndt::int_kind_type, GeneratorType> : public base_callable {
      public:
        uniform_callable()
            : base_callable(ndt::make_type<ndt::callable_type>(
                  ndt::make_type<ReturnType>(), {},
                  {{ndt::make_type<ndt::option_type>(ndt::make_type<ReturnType>()), "a"},
                   {ndt::make_type<ndt::option_type>(ndt::make_type<ReturnType>()), "b"}})) {}

        ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                          const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                          size_t DYND_UNUSED(nkwd), const array *kwds,
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
          std::shared_ptr<GeneratorType> g = get_random_device();

          ReturnType a;
          if (kwds[0].is_na()) {
            a = 0;
          } else {
            a = kwds[0].as<ReturnType>();
          }

          ReturnType b;
          if (kwds[1].is_na()) {
            b = std::numeric_limits<ReturnType>::max();
          } else {
            b = kwds[1].as<ReturnType>();
          }

          cg.emplace_back([g, a, b](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
                                    const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
                                    const char *const *DYND_UNUSED(src_arrmeta)) {
            kb.emplace_back<uniform_kernel<ReturnType, ndt::int_kind_type, GeneratorType>>(kernreq, g.get(), a, b);
          });

          return dst_tp;
        }
      };

      template <typename ReturnType, typename GeneratorType>
      class uniform_callable<ReturnType, ndt::uint_kind_type, GeneratorType>
          : public uniform_callable<ReturnType, ndt::int_kind_type, GeneratorType> {};

      template <typename ReturnType, typename GeneratorType>
      class uniform_callable<ReturnType, ndt::float_kind_type, GeneratorType> : public base_callable {
      public:
        uniform_callable()
            : base_callable(ndt::make_type<ndt::callable_type>(
                  ndt::make_type<ReturnType>(), {},
                  {{ndt::make_type<ndt::option_type>(ndt::make_type<ReturnType>()), "a"},
                   {ndt::make_type<ndt::option_type>(ndt::make_type<ReturnType>()), "b"}})) {}

        ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                          const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                          size_t DYND_UNUSED(nkwd), const array *kwds,
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
          std::shared_ptr<GeneratorType> g = get_random_device();

          ReturnType a;
          if (kwds[0].is_na()) {
            a = 0;
          } else {
            a = kwds[0].as<ReturnType>();
          }

          ReturnType b;
          if (kwds[1].is_na()) {
            b = 1;
          } else {
            b = kwds[1].as<ReturnType>();
          }

          cg.emplace_back([g, a, b](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
                                    const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
                                    const char *const *DYND_UNUSED(src_arrmeta)) {
            kb.emplace_back<uniform_kernel<ReturnType, ndt::float_kind_type, GeneratorType>>(kernreq, g.get(), a, b);
          });

          return dst_tp;
        }
      };

      template <typename ReturnType, typename GeneratorType>
      class uniform_callable<ReturnType, ndt::complex_kind_type, GeneratorType> : public base_callable {
      public:
        uniform_callable()
            : base_callable(ndt::make_type<ndt::callable_type>(
                  ndt::make_type<ReturnType>(), {},
                  {{ndt::make_type<ndt::option_type>(ndt::make_type<ReturnType>()), "a"},
                   {ndt::make_type<ndt::option_type>(ndt::make_type<ReturnType>()), "b"}})) {}

        ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                          const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                          size_t DYND_UNUSED(nkwd), const array *kwds,
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
          std::shared_ptr<GeneratorType> g = get_random_device();

          ReturnType a;
          if (kwds[0].is_na()) {
            a = ReturnType(0, 0);
          } else {
            a = kwds[0].as<ReturnType>();
          }

          ReturnType b;
          if (kwds[1].is_na()) {
            b = ReturnType(1, 1);
          } else {
            b = kwds[1].as<ReturnType>();
          }

          cg.emplace_back([g, a, b](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
                                    const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
                                    const char *const *DYND_UNUSED(src_arrmeta)) {
            kb.emplace_back<uniform_kernel<ReturnType, ndt::complex_kind_type, GeneratorType>>(kernreq, g.get(), a, b);
          });

          return dst_tp;
        }
      };

    } // namespace dynd::nd::detail

    template <typename ReturnType, typename GeneratorType>
    using uniform_callable = detail::uniform_callable<ReturnType, ndt::base_of_t<ReturnType>, GeneratorType>;

  } // namespace dynd::nd::random
} // namespace dynd::nd
} // namespace dynd
