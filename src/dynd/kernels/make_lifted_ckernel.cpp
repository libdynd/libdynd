//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/make_lifted_ckernel.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/kernels/elwise.hpp>

using namespace std;
using namespace dynd;

size_t dynd::make_lifted_expr_ckernel(
    const arrfunc_type_data *elwise_handler,
    const arrfunc_type *elwise_handler_tp, void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, const ndt::type *src_tp,
    const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  intptr_t src_count = elwise_handler_tp->get_npos();

  // Check if no lifting is required
  intptr_t dst_ndim =
      dst_tp.get_ndim() - elwise_handler_tp->get_return_type().get_ndim();
  if (dst_ndim == 0) {
    intptr_t i = 0;
    for (; i < src_count; ++i) {
      intptr_t src_ndim =
          src_tp[i].get_ndim() - elwise_handler_tp->get_arg_type(i).get_ndim();
      if (src_ndim != 0) {
        break;
      }
    }
    if (i == src_count) {
      // No dimensions to lift, call the elementwise instantiate directly
      return elwise_handler->instantiate(
          elwise_handler, elwise_handler_tp, ckb, ckb_offset, dst_tp,
          dst_arrmeta, src_tp, src_arrmeta, kernreq, ectx, nd::array());
    } else {
      intptr_t src_ndim =
          src_tp[i].get_ndim() - elwise_handler_tp->get_arg_type(i).get_ndim();
      stringstream ss;
      ss << "Trying to broadcast " << src_ndim << " dimensions of " << src_tp[i]
         << " into 0 dimensions of " << dst_tp
         << ", the destination dimension count must be greater";
      throw broadcast_error(ss.str());
    }
  }

  // Do a pass through the src types to classify them
  bool src_all_strided = true, src_all_strided_or_var = true;
  for (intptr_t i = 0; i < src_count; ++i) {
    intptr_t src_ndim =
        src_tp[i].get_ndim() - elwise_handler_tp->get_arg_type(i).get_ndim();
    switch (src_tp[i].without_memory_type().get_type_id()) {
    case fixed_dim_type_id:
    case cfixed_dim_type_id:
      break;
    case var_dim_type_id:
      src_all_strided = false;
      break;
    default:
      // If it's a scalar, allow it to broadcast like
      // a strided dimension
      if (src_ndim > 0) {
        src_all_strided_or_var = false;
      }
      break;
    }
  }

  // Call to some special-case functions based on the
  // destination type
  switch (dst_tp.without_memory_type().get_type_id()) {
  case fixed_dim_type_id:
  case cfixed_dim_type_id:
    if (src_all_strided) {
      return kernels::instantiate_elwise_ck<fixed_dim_type_id,
                                            fixed_dim_type_id>(
          elwise_handler, elwise_handler_tp, ckb, ckb_offset, dst_tp,
          dst_arrmeta, src_tp, src_arrmeta, kernreq, ectx);
    } else if (src_all_strided_or_var) {
      return kernels::instantiate_elwise_ck<fixed_dim_type_id, var_dim_type_id>(
          elwise_handler, elwise_handler_tp, ckb, ckb_offset, dst_tp,
          dst_arrmeta, src_tp, src_arrmeta, kernreq, ectx);
    } else {
      // TODO
    }
    break;
  case var_dim_type_id:
    if (src_all_strided_or_var) {
      return kernels::instantiate_elwise_ck<var_dim_type_id, fixed_dim_type_id>(
          elwise_handler, elwise_handler_tp, ckb, ckb_offset, dst_tp,
          dst_arrmeta, src_tp, src_arrmeta, kernreq, ectx);
    } else {
      // TODO
    }
    break;
  case offset_dim_type_id:
    // TODO
    break;
  default:
    break;
  }

  stringstream ss;
  ss << "Cannot process lifted elwise expression from (";
  for (intptr_t i = 0; i < src_count; ++i) {
    ss << src_tp[i];
    if (i != src_count - 1) {
      ss << ", ";
    }
  }
  ss << ") to " << dst_tp;
  throw runtime_error(ss.str());
}
