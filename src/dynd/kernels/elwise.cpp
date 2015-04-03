//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/elwise.hpp>
#include <dynd/types/dim_fragment_type.hpp>

using namespace std;
using namespace dynd;

void nd::functional::elwise_virtual_ck::resolve_dst_type(
    const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
    char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t nsrc,
    const ndt::type *src_tp, const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, ndt::type> &tp_vars)
{
  const arrfunc_type_data *child_af =
      self->get_data_as<dynd::nd::arrfunc>()->get();
  const arrfunc_type *child_af_tp =
      self->get_data_as<dynd::nd::arrfunc>()->get_type();

  intptr_t ndim = 0;
  // First get the type for the child arrfunc
  ndt::type child_dst_tp;
  if (child_af->resolve_dst_type != NULL) {
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
    child_af->resolve_dst_type(
        child_af, child_af_tp, NULL, child_dst_tp, nsrc,
        child_src_tp.empty() ? NULL : child_src_tp.data(), kwds, tp_vars);
  } else {
    // TODO: Should pattern match the source types here
    for (intptr_t i = 0; i < nsrc; ++i) {
      ndim = std::max(ndim, src_tp[i].get_ndim() -
                                child_af_tp->get_pos_type(i).get_ndim());
    }
    child_dst_tp =
        ndt::substitute(child_af_tp->get_return_type(), tp_vars, false);
  }
  if (nsrc == 0) {
    dst_tp = tp_vars.at("Dims").extended<dim_fragment_type>()->apply_to_dtype(
        child_dst_tp.without_memory_type());
    if (child_dst_tp.get_kind() == memory_kind) {
      dst_tp =
          child_dst_tp.extended<base_memory_type>()->with_replaced_storage_type(
              dst_tp);
    }

    return;
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
        ndt::type tp = src_tp[i].without_memory_type();
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
              throw broadcast_error(ndim, shape.get(), ndim_i, shape_i);
            }
            break;
          case cfixed_dim_type_id:
            shape_at_j = tp.extended<cfixed_dim_type>()->get_fixed_dim_size();
            if (shape_i[j] < 0 || shape_i[j] == 1) {
              if (shape_at_j != 1) {
                shape_i[j] = shape_at_j;
              }
            } else if (shape_i[j] != shape_at_j) {
              throw broadcast_error(ndim, shape.get(), ndim_i, shape_i);
            }
            break;
          case var_dim_type_id:
            break;
          default:
            throw broadcast_error(ndim, shape.get(), ndim_i, shape_i);
          }
          tp = tp.get_dtype(tp.get_ndim() - 1);
        }
      }
    }

    ndt::type tp = child_dst_tp.without_memory_type();
    for (intptr_t i = ndim - 1; i >= 0; --i) {
      if (shape[i] == -1) {
        tp = ndt::make_var_dim(tp);
      } else {
        tp = ndt::make_fixed_dim(shape[i], tp);
      }
    }
    if (child_dst_tp.get_kind() == memory_kind) {
      child_dst_tp =
          child_dst_tp.extended<base_memory_type>()->with_replaced_storage_type(
              tp);
    } else {
      child_dst_tp = tp;
    }
  }
  dst_tp = child_dst_tp;
}

void nd::functional::elwise_virtual_ck::resolve_option_values(
    const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
    char *DYND_UNUSED(data), intptr_t nsrc, const ndt::type *src_tp,
    nd::array &kwds, const std::map<nd::string, ndt::type> &tp_vars)
{
  const arrfunc_type_data *child =
      self->get_data_as<dynd::nd::arrfunc>()->get();
  const arrfunc_type *child_tp =
      self->get_data_as<dynd::nd::arrfunc>()->get_type();

  return child->resolve_option_values(child, child_tp, NULL, nsrc, src_tp, kwds,
                                      tp_vars);
}

intptr_t nd::functional::elwise_virtual_ck::instantiate(
    const arrfunc_type_data *self, const arrfunc_type *self_tp,
    char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    dynd::kernel_request_t kernreq, const eval::eval_context *ectx,
    const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, ndt::type> &tp_vars)
{
  const arrfunc_type_data *child =
      self->get_data_as<dynd::nd::arrfunc>()->get();
  const arrfunc_type *child_tp =
      self->get_data_as<dynd::nd::arrfunc>()->get_type();

  intptr_t src_count = child_tp->get_npos();

  // Check if no lifting is required
  intptr_t dst_ndim = dst_tp.get_ndim();
  if (!child_tp->get_return_type().is_symbolic()) {
    dst_ndim -= child_tp->get_return_type().get_ndim();
  }

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
      return child->instantiate(child, child_tp, NULL, ckb, ckb_offset, dst_tp,
                                dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq,
                                ectx, kwds, tp_vars);
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
    switch (src_tp[i].get_type_id()) {
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
  switch (dst_tp.get_type_id()) {
  case fixed_dim_type_id:
  case cfixed_dim_type_id:
    if (src_all_strided) {
      return elwise_virtual_ck::instantiate<fixed_dim_type_id,
                                            fixed_dim_type_id>(
          self, self_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
          src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
    } else if (src_all_strided_or_var) {
      return elwise_virtual_ck::instantiate<fixed_dim_type_id, var_dim_type_id>(
          self, self_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
          src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
    } else {
      // TODO
    }
    break;
  case var_dim_type_id:
    if (src_all_strided_or_var) {
      return elwise_virtual_ck::instantiate<var_dim_type_id, fixed_dim_type_id>(
          self, self_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
          src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
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

template <type_id_t dst_type_id, type_id_t src_type_id>
intptr_t nd::functional::elwise_virtual_ck::instantiate(
    const arrfunc_type_data *self, const arrfunc_type *self_tp,
    char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, ndt::type> &tp_vars)
{
  const arrfunc_type *child_tp =
      self->get_data_as<dynd::nd::arrfunc>()->get_type();

  switch (child_tp->get_npos()) {
  case 0:
    return elwise_ck<dst_type_id, src_type_id, 0>::instantiate(
        self, self_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
        src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case 1:
    return elwise_ck<dst_type_id, src_type_id, 1>::instantiate(
        self, self_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
        src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case 2:
    return elwise_ck<dst_type_id, src_type_id, 2>::instantiate(
        self, self_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
        src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case 3:
    return elwise_ck<dst_type_id, src_type_id, 3>::instantiate(
        self, self_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
        src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case 4:
    return elwise_ck<dst_type_id, src_type_id, 4>::instantiate(
        self, self_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
        src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case 5:
    return elwise_ck<dst_type_id, src_type_id, 5>::instantiate(
        self, self_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
        src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case 6:
    return elwise_ck<dst_type_id, src_type_id, 6>::instantiate(
        self, self_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
        src_arrmeta, kernreq, ectx, kwds, tp_vars);
  default:
    throw runtime_error("elwise with src_count > 6 not implemented yet");
  }
}