#include <dynd/kernels/elwise.hpp>

size_t dynd::kernels::make_lifted_expr_ckernel(
    const arrfunc_type_data *child, const arrfunc_type *child_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    dynd::kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &kwds)
{
  intptr_t src_count = child_tp->get_npos();

  // Check if no lifting is required
  intptr_t dst_ndim =
      dst_tp.get_ndim() - child_tp->get_return_type().get_ndim();
  if (dst_ndim == 0) {
    intptr_t i = 0;
    for (; i < src_count; ++i) {
      intptr_t src_ndim =
          src_tp[i].get_ndim() - child_tp->get_arg_type(i).get_ndim();
      if (src_ndim != 0) {
        break;
      }
    }
    if (i == src_count) {
      // No dimensions to lift, call the elementwise instantiate directly
      return child->instantiate(child, child_tp, ckb, ckb_offset, dst_tp,
                                dst_arrmeta, src_tp, src_arrmeta, kernreq, ectx,
                                kwds);
    } else {
      intptr_t src_ndim =
          src_tp[i].get_ndim() - child_tp->get_arg_type(i).get_ndim();
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
        src_tp[i].get_ndim() - child_tp->get_arg_type(i).get_ndim();
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
          child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
          src_arrmeta, kernreq, ectx, kwds);
    } else if (src_all_strided_or_var) {
      return kernels::instantiate_elwise_ck<fixed_dim_type_id, var_dim_type_id>(
          child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
          src_arrmeta, kernreq, ectx, kwds);
    } else {
      // TODO
    }
    break;
  case var_dim_type_id:
    if (src_all_strided_or_var) {
      return kernels::instantiate_elwise_ck<var_dim_type_id, fixed_dim_type_id>(
          child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
          src_arrmeta, kernreq, ectx, kwds);
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