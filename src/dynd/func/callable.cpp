//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/assignment.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>
#include <dynd/types/expr_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/type.hpp>

using namespace std;
using namespace dynd;

namespace {

////////////////////////////////////////////////////////////////
// Functions for the unary assignment as an callable

struct unary_assignment_ck : nd::base_virtual_kernel<unary_assignment_ck> {
  static intptr_t instantiate(char *static_data, char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                              const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                              const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                              const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                              const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    assign_error_mode errmode = *reinterpret_cast<assign_error_mode *>(static_data);
    if (errmode == ectx->errmode) {
      return make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq, ectx);
    }
    else {
      eval::eval_context ectx_tmp(*ectx);
      ectx_tmp.errmode = errmode;
      return make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq,
                                    &ectx_tmp);
    }
  }
};

////////////////////////////////////////////////////////////////
// Functions for property access as an callable

struct property_kernel : nd::base_virtual_kernel<property_kernel> {
  static intptr_t instantiate(char *static_data, char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                              const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                              const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                              const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                              const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    ndt::type prop_src_tp = *reinterpret_cast<ndt::type *>(static_data);

    if (dst_tp.value_type() == prop_src_tp.value_type()) {
      if (src_tp[0] == prop_src_tp.operand_type()) {
        return make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, prop_src_tp, src_arrmeta[0], kernreq, ectx);
      }
      else if (src_tp[0].value_type() == prop_src_tp.operand_type()) {
        return make_assignment_kernel(
            ckb, ckb_offset, dst_tp, dst_arrmeta,
            prop_src_tp.extended<ndt::base_expr_type>()->with_replaced_storage_type(src_tp[0]), src_arrmeta[0], kernreq,
            ectx);
      }
    }

    stringstream ss;
    ss << "Cannot instantiate callable for assigning from ";
    ss << " using input type " << src_tp[0];
    ss << " and output type " << dst_tp;
    throw type_error(ss.str());
  }
};

} // anonymous namespace

nd::callable dynd::make_callable_from_assignment(const ndt::type &dst_tp, const ndt::type &src_tp,
                                                 assign_error_mode errmode)
{
  return nd::callable::make<unary_assignment_ck>(ndt::callable_type::make(dst_tp, src_tp), errmode);
}

void nd::detail::check_narg(const ndt::callable_type *af_tp, intptr_t narg)
{
  if (!af_tp->is_pos_variadic() && narg != af_tp->get_npos()) {
    std::stringstream ss;
    ss << "callable expected " << af_tp->get_npos() << " positional arguments, but received " << narg;
    throw std::invalid_argument(ss.str());
  }
}

void nd::detail::check_arg(const ndt::callable_type *af_tp, intptr_t i, const ndt::type &actual_tp,
                           const char *actual_arrmeta, std::map<std::string, ndt::type> &tp_vars)
{
  if (af_tp->is_pos_variadic()) {
    return;
  }

  ndt::type expected_tp = af_tp->get_pos_type(i);
  if (!expected_tp.match(NULL, actual_tp.value_type(), actual_arrmeta, tp_vars)) {
    std::stringstream ss;
    ss << "positional argument " << i << " to callable does not match, ";
    ss << "expected " << expected_tp << ", received " << actual_tp;
    throw std::invalid_argument(ss.str());
  }
}
