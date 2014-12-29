//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/elwise.hpp>

using namespace std;
using namespace dynd;

int decl::nd::elwise::resolve_dst_type(const arrfunc_type_data *child_af,
                                       const arrfunc_type *child_af_tp,
                                       intptr_t nsrc, const ndt::type *src_tp,
                                       int throw_on_error,
                                       ndt::type &out_dst_tp,
                                       const dynd::nd::array &kwds)
{
  intptr_t ndim = 0;
  // First get the type for the child arrfunc
  ndt::type child_dst_tp;
  if (child_af->resolve_dst_type) {
    std::vector<ndt::type> child_src_tp(nsrc);
    for (intptr_t i = 0; i < nsrc; ++i) {
      intptr_t child_ndim_i = child_af_tp->get_pos_type(i).get_ndim();
      if (child_ndim_i < src_tp[i].get_ndim()) {
        child_src_tp[i] = src_tp[i].get_dtype(child_ndim_i);
        ndim = std::max(ndim, src_tp[i].get_ndim() - child_ndim_i);
      } else {
        child_src_tp[i] = src_tp[i];
      }
    }
    if (!child_af->resolve_dst_type(child_af, child_af_tp, nsrc,
                                    &child_src_tp[0], throw_on_error,
                                    child_dst_tp, kwds)) {
      return 0;
    }
  } else {
    // TODO: Should pattern match the source types here
    for (intptr_t i = 0; i < nsrc; ++i) {
      ndim = std::max(ndim, src_tp[i].get_ndim() -
                                child_af_tp->get_pos_type(i).get_ndim());
    }
    child_dst_tp = child_af_tp->get_return_type();
  }
  // Then build the type for the rest of the dimensions
  if (ndim > 0) {
    dimvector shape(ndim), tmp_shape(ndim);
    for (intptr_t i = 0; i < ndim; ++i) {
      shape[i] = -1;
    }
    for (intptr_t i = 0; i < nsrc; ++i) {
      intptr_t ndim_i =
          src_tp[i].get_ndim() - child_af_tp->get_pos_type(i).get_ndim();
      if (ndim_i > 0) {
        ndt::type tp = src_tp[i];
        intptr_t *shape_i = shape.get() + (ndim - ndim_i);
        intptr_t shape_at_j;
        for (intptr_t j = 0; j < ndim_i; ++j) {
          switch (tp.get_type_id()) {
          case fixed_dim_type_id:
            shape_at_j = tp.extended<fixed_dim_type>()->get_fixed_dim_size();
            if (shape_i[j] < 0 || shape_i[j] == 1) {
              if (shape_at_j != 1) {
                shape_i[j] = shape_at_j;
              }
            } else if (shape_i[j] != shape_at_j) {
              if (throw_on_error) {
                throw broadcast_error(ndim, shape.get(), ndim_i, shape_i);
              } else {
                return 0;
              }
            }
            break;
          case cfixed_dim_type_id:
            shape_at_j = tp.extended<cfixed_dim_type>()->get_fixed_dim_size();
            if (shape_i[j] < 0 || shape_i[j] == 1) {
              if (shape_at_j != 1) {
                shape_i[j] = shape_at_j;
              }
            } else if (shape_i[j] != shape_at_j) {
              if (throw_on_error) {
                throw broadcast_error(ndim, shape.get(), ndim_i, shape_i);
              } else {
                return 0;
              }
            }
            break;
          case var_dim_type_id:
            break;
          default:
            if (throw_on_error) {
              throw broadcast_error(ndim, shape.get(), ndim_i, shape_i);
            } else {
              return 0;
            }
          }
          tp = tp.get_dtype(tp.get_ndim() - 1);
        }
      }
    }
    for (intptr_t i = ndim - 1; i >= 0; --i) {
      if (shape[i] == -1) {
        child_dst_tp = ndt::make_var_dim(child_dst_tp);
      } else {
        child_dst_tp = ndt::make_fixed_dim(shape[i], child_dst_tp);
      }
    }
  }
  out_dst_tp = child_dst_tp;

  return 1;
}

intptr_t decl::nd::elwise::instantiate(
    const arrfunc_type_data *child, const arrfunc_type *child_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    dynd::kernel_request_t kernreq, const eval::eval_context *ectx,
    const dynd::nd::array &kwds)
{
  intptr_t src_count = child_tp->get_npos();

  // Check if no lifting is required
  intptr_t dst_ndim =
      dst_tp.get_ndim() - child_tp->get_return_type().get_ndim();
  if (dst_ndim == 0) {
    intptr_t i = 0;
    for (; i < src_count; ++i) {
      intptr_t src_ndim =
          src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
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
          src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
      stringstream ss;
      ss << "Trying to broadcast " << src_ndim << " dimensions of " << src_tp[i]
         << " into 0 dimensions of " << dst_tp
         << ", the destination dimension count must be greater. The element "
            "arrfunc type is \"" << ndt::type(child_tp, true) << "\"";
      throw broadcast_error(ss.str());
    }
  }

  // Do a pass through the src types to classify them
  bool src_all_strided = true, src_all_strided_or_var = true;
  for (intptr_t i = 0; i < src_count; ++i) {
    intptr_t src_ndim =
        src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
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
      return instantiate<fixed_dim_type_id, fixed_dim_type_id>(
          child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
          src_arrmeta, kernreq, ectx, kwds);
    } else if (src_all_strided_or_var) {
      return instantiate<fixed_dim_type_id, var_dim_type_id>(
          child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
          src_arrmeta, kernreq, ectx, kwds);
    } else {
      // TODO
    }
    break;
  case var_dim_type_id:
    if (src_all_strided_or_var) {
      return instantiate<var_dim_type_id, fixed_dim_type_id>(
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

ndt::type decl::nd::elwise::make_lifted_type(const arrfunc_type *child_tp)
{
  const ndt::type *param_types = child_tp->get_pos_types_raw();
  intptr_t param_count = child_tp->get_narg();
  dynd::nd::array out_param_types =
      dynd::nd::empty(param_count, ndt::make_type());
  dynd::nd::string dimsname("Dims");
  ndt::type *pt =
      reinterpret_cast<ndt::type *>(out_param_types.get_readwrite_originptr());
  for (intptr_t i = 0, i_end = child_tp->get_npos(); i != i_end; ++i) {
    pt[i] = ndt::make_ellipsis_dim(dimsname, param_types[i]);
  }
  for (intptr_t i = child_tp->get_npos(), i_end = child_tp->get_narg();
       i != i_end; ++i) {
    pt[i] = param_types[i];
  }
  out_param_types.flag_as_immutable();
  return ndt::make_arrfunc(
      ndt::make_tuple(out_param_types),
      ndt::make_ellipsis_dim(dimsname, child_tp->get_return_type()));
}

decl::nd::elwise nd::elwise;