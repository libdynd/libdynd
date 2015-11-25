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

    // All methods are inlined, so this does not need to be declared DYND_API.
    struct constant_kernel : base_kernel<constant_kernel> {
      static const std::size_t data_size = 0;

      char *data;

      constant_kernel(char *data) : data(data)
      {
      }

      ~constant_kernel()
      {
        get_child()->destroy();
      }

      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        get_child()->single(dst, &data);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        static std::intptr_t data_stride[1] = {0};

        get_child()->strided(dst, dst_stride, &data, data_stride, count);
      }

      static intptr_t instantiate(char *static_data, char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                                  const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        const array &val = *reinterpret_cast<array *>(static_data);

        make(ckb, kernreq, ckb_offset, const_cast<char *>(val.cdata()));
        return make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, dst_tp, val.get()->metadata(), kernreq,
                                      ectx);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
