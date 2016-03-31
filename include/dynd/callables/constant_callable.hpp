//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/assignment.hpp>
#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/constant_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    class constant_callable : public base_callable {
      array m_val;

    public:
      constant_callable(const nd::array &val)
          : base_callable(
                ndt::callable_type::make(val.get_type(), ndt::tuple_type::make(true), ndt::struct_type::make(true))),
            m_val(val) {}

      void resolve(call_graph &cg) { cg.emplace_back(this); }

      void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                       intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                       const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                       intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                       const std::map<std::string, ndt::type> &tp_vars) {
        ckb->emplace_back<constant_kernel>(kernreq, const_cast<char *>(m_val.cdata()));

        nd::array error_mode = assign_error_default;
        const char *child_src_metadata = m_val.get()->metadata();
        assign->instantiate(NULL, ckb, dst_tp, dst_arrmeta, 1, &dst_tp, &child_src_metadata,
                            kernreq | kernel_request_data_only, 1, &error_mode, tp_vars);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
