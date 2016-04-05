//
// Copyright (C) 2011-16 DyND Developers
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

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                        size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                        const std::map<std::string, ndt::type> &tp_vars) {
        cg.push_back([val = m_val](call_node * &DYND_UNUSED(node), kernel_builder * ckb, kernel_request_t kernreq,
                                   const char *dst_arrmeta, size_t DYND_UNUSED(nsrc),
                                   const char *const *DYND_UNUSED(src_arrmeta)) {
          ckb->emplace_back<constant_kernel>(kernreq, const_cast<char *>(val.cdata()));

          const char *child_src_metadata = val.get()->metadata();
          ckb->instantiate(kernreq | kernel_request_data_only, dst_arrmeta, 1, &child_src_metadata);
        });

        nd::array error_mode = assign_error_default;
        assign->resolve(this, nullptr, cg, dst_tp, 1, &dst_tp, 1, &error_mode, tp_vars);

        return dst_tp;
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
