//
// Copyright (C) 2011-15 DyND Developers
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

      template <type_id_t ResID, type_id_t ResBaseID, typename GeneratorType>
      class uniform_callable;

      template <type_id_t ResID, typename GeneratorType>
      class uniform_callable<ResID, int_kind_id, GeneratorType> : public base_callable {
        typedef typename type_of<ResID>::type R;

      public:
        uniform_callable()
            : base_callable(
                  ndt::callable_type::make(ResID, {}, {"a", "b"}, {ndt::make_type<ndt::option_type>(ResID),
                                                                   ndt::make_type<ndt::option_type>(ResID)})) {}

        const ndt::type &resolve(call_graph &cg, const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                                 const ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                                 const array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
          cg.emplace_back(this);
          return dst_tp;
        }

        void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                         const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                         const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                         kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                         const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
          std::shared_ptr<GeneratorType> g = get_random_device();

          R a;
          if (kwds[0].is_na()) {
            a = 0;
          } else {
            a = kwds[0].as<R>();
          }

          R b;
          if (kwds[1].is_na()) {
            b = std::numeric_limits<R>::max();
          } else {
            b = kwds[1].as<R>();
          }

          ckb->emplace_back<uniform_kernel<ResID, int_kind_id, GeneratorType>>(kernreq, g.get(), a, b);
        }
      };

      template <type_id_t ResID, typename GeneratorType>
      class uniform_callable<ResID, uint_kind_id, GeneratorType>
          : public uniform_callable<ResID, int_kind_id, GeneratorType> {};

      template <type_id_t ResID, typename GeneratorType>
      class uniform_callable<ResID, float_kind_id, GeneratorType> : public base_callable {
        typedef typename type_of<ResID>::type R;

      public:
        uniform_callable()
            : base_callable(
                  ndt::callable_type::make(ResID, {}, {"a", "b"}, {ndt::make_type<ndt::option_type>(ResID),
                                                                   ndt::make_type<ndt::option_type>(ResID)})) {}

        const ndt::type &resolve(call_graph &cg, const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                                 const ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                                 const array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
          cg.emplace_back(this);
          return dst_tp;
        }

        void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                         const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                         const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                         kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                         const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
          std::shared_ptr<GeneratorType> g = get_random_device();

          R a;
          if (kwds[0].is_na()) {
            a = 0;
          } else {
            a = kwds[0].as<R>();
          }

          R b;
          if (kwds[1].is_na()) {
            b = 1;
          } else {
            b = kwds[1].as<R>();
          }

          ckb->emplace_back<uniform_kernel<ResID, float_kind_id, GeneratorType>>(kernreq, g.get(), a, b);
        }
      };

      template <type_id_t ResID, typename GeneratorType>
      class uniform_callable<ResID, complex_kind_id, GeneratorType> : public base_callable {
        typedef typename type_of<ResID>::type R;

      public:
        uniform_callable()
            : base_callable(
                  ndt::callable_type::make(ResID, {}, {"a", "b"}, {ndt::make_type<ndt::option_type>(ResID),
                                                                   ndt::make_type<ndt::option_type>(ResID)})) {}

        const ndt::type &resolve(call_graph &cg, const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                                 const ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                                 const array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
          cg.emplace_back(this);
          return dst_tp;
        }

        void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                         const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                         const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                         kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                         const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
          std::shared_ptr<GeneratorType> g = get_random_device();

          R a;
          if (kwds[0].is_na()) {
            a = R(0, 0);
          } else {
            a = kwds[0].as<R>();
          }

          R b;
          if (kwds[1].is_na()) {
            b = R(1, 1);
          } else {
            b = kwds[1].as<R>();
          }

          ckb->emplace_back<uniform_kernel<ResID, complex_kind_id, GeneratorType>>(kernreq, g.get(), a, b);
        }
      };

    } // namespace dynd::nd::detail

    template <type_id_t ResID, typename GeneratorType>
    using uniform_callable = detail::uniform_callable<ResID, base_id_of<ResID>::value, GeneratorType>;

  } // namespace dynd::nd::random
} // namespace dynd::nd
} // namespace dynd
