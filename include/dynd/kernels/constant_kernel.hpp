//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/func/assignment.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    struct constant_kernel : base_strided_kernel<constant_kernel, 0> {
      char *data;

      constant_kernel(char *data) : data(data) {}

      ~constant_kernel() { get_child()->destroy(); }

      void single(char *dst, char *const *DYND_UNUSED(src)) { get_child()->single(dst, &data); }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        static std::intptr_t data_stride[1] = {0};

        get_child()->strided(dst, dst_stride, &data, data_stride, count);
      }

      static void instantiate(char *static_data, char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp,
                              const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                              const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                              intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &tp_vars)
      {
        const array &val = *reinterpret_cast<array *>(static_data);

        ckb->emplace_back<constant_kernel>(kernreq, const_cast<char *>(val.cdata()));

        nd::array error_mode = assign_error_default;
        const char *child_src_metadata = val.get()->metadata();
        assign::get()->instantiate(assign::get()->static_data(), NULL, ckb, dst_tp, dst_arrmeta, 1, &dst_tp,
                                   &child_src_metadata, kernreq | kernel_request_data_only, 1, &error_mode, tp_vars);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
