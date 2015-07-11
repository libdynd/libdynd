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
#include <dynd/func/arrfunc.hpp>

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
    inline bool
    can_implicitly_convert(const ndt::type &src, const ndt::type &dst,
                           std::map<nd::string, ndt::type> &typevars)
    {
      if (src == dst) {
        return true;
      }
      if (src.get_ndim() > 0 || dst.get_ndim() > 0) {
        ndt::type src_dtype, dst_dtype;
        if (src.match(dst, typevars)) {
          return can_implicitly_convert(src.get_dtype(), dst.get_dtype(),
                                        typevars);
        } else {
          return false;
        }
      }

      if (src.get_kind() == uint_kind &&
          (dst.get_kind() == uint_kind || dst.get_kind() == sint_kind ||
           dst.get_kind() == real_kind)) {
        return src.get_data_size() < dst.get_data_size();
      }
      if (src.get_kind() == sint_kind &&
          (dst.get_kind() == sint_kind || dst.get_kind() == real_kind)) {
        return src.get_data_size() < dst.get_data_size();
      }
      if (src.get_kind() == real_kind) {
        if (dst.get_kind() == real_kind) {
          return src.get_data_size() < dst.get_data_size();
        } else if (dst.get_kind() == complex_kind) {
          return src.get_data_size() * 2 <= dst.get_data_size();
        }
      }
      return false;
    }

    struct old_multidispatch_ck : base_virtual_kernel<old_multidispatch_ck> {
      static void resolve_dst_type(
          const arrfunc_type_data *af_self, const ndt::arrfunc_type *af_tp,
          char *static_data, size_t data_size, char *data, ndt::type &dst_tp,
          intptr_t nsrc, const ndt::type *src_tp, const nd::array &kwds,
          const std::map<nd::string, ndt::type> &tp_vars);

      static intptr_t
      instantiate(const arrfunc_type_data *af_self,
                  const ndt::arrfunc_type *af_tp, const char *static_data,
                  size_t data_size, char *data, void *ckb, intptr_t ckb_offset,
                  const ndt::type &dst_tp, const char *dst_arrmeta,
                  intptr_t nsrc, const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *ectx, const nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type> &tp_vars);
    };

    template <typename StaticDataType>
    struct multidispatch_kernel
        : base_virtual_kernel<multidispatch_kernel<StaticDataType>> {
      typedef StaticDataType static_data;

      struct data {
        const arrfunc &child;

        data(const arrfunc &child) : child(child) {}
      };

      /*
            static void
            data_init(const arrfunc_type_data *self,
                      const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                      const char *DYND_UNUSED(static_data),
                      size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
                      intptr_t nsrc, const ndt::type *src_tp, array
         &DYND_UNUSED(kwds),
                      const std::map<nd::string, ndt::type>
         &DYND_UNUSED(tp_vars))
            {
              static_data &static_data =
                  *self->get_data_as<std::shared_ptr<StaticDataType>>()->get();

              const arrfunc &child = static_data(dst_tp, nsrc, src_tp);
              if (child->data_init != NULL) {
                child->data_init(child, self_tp, NULL, 0, NULL, nsrc, src_tp,
         kwds,
                                 tp_vars);
              }
            }
      */

      static void
      resolve_dst_type(const arrfunc_type_data *self,
                       const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                       char *DYND_UNUSED(static_data),
                       size_t DYND_UNUSED(data_size), char *data,
                       ndt::type &dst_tp, intptr_t nsrc,
                       const ndt::type *src_tp, const dynd::nd::array &kwds,
                       const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        static_data &static_data =
            *self->get_data_as<std::shared_ptr<StaticDataType>>()->get();

        arrfunc &child = const_cast<arrfunc &>(static_data(dst_tp, nsrc, src_tp));

        const ndt::type &child_dst_tp = child.get_type()->get_return_type();
        if (child_dst_tp.is_symbolic()) {
          child.get()->resolve_dst_type(child.get(), child.get_type(),
                                        child.get()->static_data,
                                        child.get()->data_size, data, dst_tp,
                                        nsrc, src_tp, kwds, tp_vars);
        } else {
          dst_tp = child_dst_tp;
        }
      }

      static intptr_t instantiate(
          const arrfunc_type_data *self,
          const ndt::arrfunc_type *DYND_UNUSED(self_tp),
          const char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
          char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
          const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
          const char *const *src_arrmeta, kernel_request_t kernreq,
          const eval::eval_context *ectx, const dynd::nd::array &kwds,
          const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        static_data &static_data =
            *self->get_data_as<std::shared_ptr<StaticDataType>>()->get();

        const arrfunc &child = static_data(dst_tp, nsrc, src_tp);
        return child.get()->instantiate(
            child.get(), child.get_type(), child.get()->static_data,
            child.get()->data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd