//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/assignment.hpp>
#include <dynd/callable.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/type.hpp>

using namespace std;
using namespace dynd;

namespace {

////////////////////////////////////////////////////////////////
// Functions for the unary assignment as an callable

struct unary_assignment_ck : nd::base_kernel<unary_assignment_ck, 1> {
  static void instantiate(char *static_data, char *DYND_UNUSED(data), nd::kernel_builder *ckb, const ndt::type &dst_tp,
                          const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                          const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                          const nd::array *DYND_UNUSED(kwds),
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    assign_error_mode errmode = *reinterpret_cast<assign_error_mode *>(static_data);
    eval::eval_context ectx_tmp;
    ectx_tmp.errmode = errmode;
    make_assignment_kernel(ckb, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq, &ectx_tmp);
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
  ndt::type candidate_tp = actual_tp;
  if (actual_tp.get_id() != pointer_id) {
    candidate_tp = candidate_tp.value_type();
  }

  if (!expected_tp.match(NULL, candidate_tp, actual_arrmeta, tp_vars)) {
    std::stringstream ss;
    ss << "positional argument " << i << " to callable does not match, ";
    ss << "expected " << expected_tp << ", received " << actual_tp;
    throw std::invalid_argument(ss.str());
  }
}
