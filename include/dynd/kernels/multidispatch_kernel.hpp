//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <set>
#include <unordered_map>

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>
#include <dynd/type.hpp>
#include <dynd/func/callable.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * Placeholder hard-coded function for determining allowable
     * implicit conversions during dispatch. Allowing conversions based
     * on ``kind`` of the following forms:
     *
     * uint -> uint, where the size is nondecreasing
     * uint -> int, where the size is increasing
     * int -> int, where the size is nondecreasing
     * uint -> real, where the size is increasing
     * int -> real, where the size is increasing
     * real -> real, where the size is nondecreasing
     * real -> complex, where the size of the real component is nondecreasing
     *
     */
    inline bool can_implicitly_convert(const ndt::type &src, const ndt::type &dst,
                                       std::map<std::string, ndt::type> &typevars)
    {
      if (src == dst) {
        return true;
      }
      if (src.get_ndim() > 0 || dst.get_ndim() > 0) {
        ndt::type src_dtype, dst_dtype;
        if (src.match(dst, typevars)) {
          return can_implicitly_convert(src.get_dtype(), dst.get_dtype(), typevars);
        }
        else {
          return false;
        }
      }

      if (src.get_kind() == uint_kind &&
          (dst.get_kind() == uint_kind || dst.get_kind() == sint_kind || dst.get_kind() == real_kind)) {
        return src.get_data_size() < dst.get_data_size();
      }
      if (src.get_kind() == sint_kind && (dst.get_kind() == sint_kind || dst.get_kind() == real_kind)) {
        return src.get_data_size() < dst.get_data_size();
      }
      if (src.get_kind() == real_kind) {
        if (dst.get_kind() == real_kind) {
          return src.get_data_size() < dst.get_data_size();
        }
        else if (dst.get_kind() == complex_kind) {
          return src.get_data_size() * 2 <= dst.get_data_size();
        }
      }
      return false;
    }

    struct DYND_API old_multidispatch_ck : base_virtual_kernel<old_multidispatch_ck> {
      static void resolve_dst_type(char *static_data, char *data, ndt::type &dst_tp, intptr_t nsrc,
                                   const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                                   const std::map<std::string, ndt::type> &tp_vars);

      static intptr_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars);
    };

    template <typename DispatcherType>
    struct multidispatch_kernel : base_virtual_kernel<multidispatch_kernel<DispatcherType>> {
      typedef DispatcherType static_data_type;

/*
      static char *data_init(char *static_data, const ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                             intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                             const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        DispatcherType &dispatcher = *reinterpret_cast<static_data_type *>(static_data);
        callable &child = dispatcher(dst_tp, nsrc, src_tp);

        std::cout << child << std::endl;

        return NULL;
      }
*/

      static void resolve_dst_type(char *static_data, char *data, ndt::type &dst_tp, intptr_t nsrc,
                                   const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                                   const std::map<std::string, ndt::type> &tp_vars)
      {
        DispatcherType &dispatcher = *reinterpret_cast<static_data_type *>(static_data);

        callable &child = dispatcher(dst_tp, nsrc, src_tp);
        const ndt::type &child_dst_tp = child.get_type()->get_return_type();
        if (child_dst_tp.is_symbolic()) {
          child->resolve_dst_type(child->static_data(), data, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
        }
        else {
          dst_tp = child_dst_tp;
        }
      }

      static intptr_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t nkwd, const array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars)
      {
        DispatcherType &dispatcher = *reinterpret_cast<static_data_type *>(static_data);

        callable &child = dispatcher(dst_tp, nsrc, src_tp);
        return child->instantiate(child->static_data(), data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
                                  src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
