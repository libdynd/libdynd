//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/elwise.hpp>
#include <dynd/types/dim_fragment_type.hpp>

using namespace std;
using namespace dynd;

void nd::functional::elwise_resolve_option_values(
    const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
    intptr_t nsrc, const ndt::type *src_tp, nd::array &kwds,
    const std::map<nd::string, ndt::type> &tp_vars)
{
  const arrfunc_type_data *child =
      self->get_data_as<dynd::nd::arrfunc>()->get();
  const arrfunc_type *child_tp =
      self->get_data_as<dynd::nd::arrfunc>()->get_type();

  return child->resolve_option_values(child, child_tp, nsrc, src_tp, kwds,
                                      tp_vars);
}

int nd::functional::elwise_resolve_dst_type_with_child(
    const arrfunc_type_data *child_af, const arrfunc_type *child_af_tp,
    intptr_t nsrc, const ndt::type *src_tp, int throw_on_error,
    ndt::type &out_dst_tp, const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, ndt::type> &tp_vars)
{
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
    if (!child_af->resolve_dst_type(
            child_af, child_af_tp, nsrc,
            child_src_tp.empty() ? NULL : child_src_tp.data(), throw_on_error,
            child_dst_tp, kwds, tp_vars)) {
      return 0;
    }
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
    out_dst_tp =
        tp_vars.at("Dims").extended<dim_fragment_type>()->apply_to_dtype(
            child_dst_tp);
    return 1;
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
  out_dst_tp = child_dst_tp;

  return 1;
}

#ifdef __CUDACC__
static void *create_cuda_device_trampoline(void *ckb, intptr_t ckb_offset,
                                           intptr_t src_count,
                                           dynd::kernel_request_t kernreq,
                                           unsigned int blocks,
                                           unsigned int threads)
{
  switch (src_count) {
  case 1: {
    typedef kernels::cuda_parallel_ck<1> self_type;
    self_type *self =
        self_type::create(ckb, kernreq, ckb_offset, blocks, threads);
    return &self->ckb;
  }
  case 2: {
    typedef kernels::cuda_parallel_ck<2> self_type;
    self_type *self =
        self_type::create(ckb, kernreq, ckb_offset, blocks, threads);
    return &self->ckb;
  }
  case 3: {
    typedef kernels::cuda_parallel_ck<3> self_type;
    self_type *self =
        self_type::create(ckb, kernreq, ckb_offset, blocks, threads);
    return &self->ckb;
  }
  case 4: {
    typedef kernels::cuda_parallel_ck<4> self_type;
    self_type *self =
        self_type::create(ckb, kernreq, ckb_offset, blocks, threads);
    return &self->ckb;
  }
  case 5: {
    typedef kernels::cuda_parallel_ck<5> self_type;
    self_type *self =
        self_type::create(ckb, kernreq, ckb_offset, blocks, threads);
    return &self->ckb;
  }
  case 6: {
    typedef kernels::cuda_parallel_ck<6> self_type;
    self_type *self =
        self_type::create(ckb, kernreq, ckb_offset, blocks, threads);
    return &self->ckb;
  }
  default:
    throw runtime_error("elwise with src_count > 6 not implemented yet");
  }
}
#endif // __CUDACC__

ndt::type nd::functional::elwise_make_type(const arrfunc_type *child_tp)
{
  const ndt::type *param_types = child_tp->get_pos_types_raw();
  intptr_t param_count = child_tp->get_npos();
  dynd::nd::array out_param_types =
      dynd::nd::empty(param_count, ndt::make_type());
  dynd::nd::string dimsname("Dims");
  ndt::type *pt =
      reinterpret_cast<ndt::type *>(out_param_types.get_readwrite_originptr());

  for (intptr_t i = 0, i_end = child_tp->get_npos(); i != i_end; ++i) {
    if (param_types[i].get_kind() == memory_kind) {
      pt[i] = pt[i].extended<base_memory_type>()->with_replaced_storage_type(
          ndt::make_ellipsis_dim(dimsname,
                                 param_types[i].without_memory_type()));
    } else {
      pt[i] = ndt::make_ellipsis_dim(dimsname, param_types[i]);
    }
  }

  ndt::type kwd_tp = child_tp->get_kwd_struct();
  /*
    if (true) {
      intptr_t old_field_count =
          kwd_tp.extended<base_struct_type>()->get_field_count();
      nd::array names =
          nd::empty(ndt::make_fixed_dim(old_field_count + 2,
    ndt::make_string()));
      nd::array fields =
          nd::empty(ndt::make_fixed_dim(old_field_count + 2, ndt::make_type()));
      for (intptr_t i = 0; i < old_field_count; ++i) {
        names(i)
            .val_assign(kwd_tp.extended<base_struct_type>()->get_field_name(i));
        fields(i)
            .val_assign(kwd_tp.extended<base_struct_type>()->get_field_type(i));
      }
      names(old_field_count).val_assign("threads");
      fields(old_field_count)
          .val_assign(ndt::make_option(ndt::make_type<int>()));
      names(old_field_count + 1).val_assign("blocks");
      fields(old_field_count + 1)
          .val_assign(ndt::make_option(ndt::make_type<int>()));
      kwd_tp = ndt::make_struct(names, fields);
    }
  */

  ndt::type ret_tp = child_tp->get_return_type();
  if (ret_tp.get_kind() == memory_kind) {
    ret_tp = ret_tp.extended<base_memory_type>()->with_replaced_storage_type(
        ndt::make_ellipsis_dim(dimsname, ret_tp.without_memory_type()));
  } else {
    ret_tp = ndt::make_ellipsis_dim(dimsname, ret_tp);
  }

  return ndt::make_arrfunc(ndt::make_tuple(out_param_types), kwd_tp, ret_tp);
}

nd::arrfunc nd::functional::elwise(const arrfunc &child)
{
  const arrfunc_type *child_tp = child.get_type();

  return dynd::nd::arrfunc(
      elwise_make_type(child.get_type()), child, &elwise_instantiate,
      (child.get()->resolve_option_values == NULL)
          ? NULL
          : &elwise_resolve_option_values,
      child_tp->has_kwd("dst_tp") ? NULL : &elwise_resolve_dst_type);
}

template <int I>
    typename std::enable_if <
    I<10, intptr_t>::type nd::functional::elwise_instantiate_with_child(
        const arrfunc_type_data *child, const arrfunc_type *child_tp, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        const ndt::type *src_tp, const char *const *src_arrmeta,
        dynd::kernel_request_t kernreq, const eval::eval_context *ectx,
        const dynd::nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &tp_vars)
{
  intptr_t src_count = child_tp->get_npos();

#ifdef __CUDACC__
  if (dst_tp.get_type_id() == cuda_device_type_id) {
    // If everything is CUDA device memory, then instantiate a CUDA
    // proxy ckernel, and add the cuda_device request flag to the kernreq.
    bool src_all_device = true;
    for (intptr_t i = 0; i < src_count; ++i) {
      src_all_device =
          src_all_device && (src_tp[i].get_type_id() == cuda_device_type_id);
    }

    if (src_all_device) {
      if ((kernreq & kernel_request_memory) != kernel_request_host) {
        throw invalid_argument(
            "got CUDA device_types, but not kernel_request_host");
      }
      /*
            int blocks, threads;
            try {
              if (kwds.p("blocks").is_missing()) {
                blocks = 256;
              } else {
                blocks = kwds.p("blocks").as<int>();
              }
            } catch (...) {
              blocks = 256;
            }
            try {
              if (kwds.p("threads").is_missing()) {
                threads = 256;
              } else {
                threads = kwds.p("threads").as<int>();
              }
            } catch (...) {
              threads = 256;
            }
      */
      void *cuda_ckb = create_cuda_device_trampoline(ckb, ckb_offset, src_count,
                                                     kernreq, 128, 128);
      ndt::type new_dst_tp =
          dst_tp.extended<base_memory_type>()->get_element_type();
      vector<ndt::type> new_src_tp(src_count);
      for (intptr_t i = 0; i < src_count; ++i) {
        new_src_tp[i] =
            src_tp[i].extended<base_memory_type>()->get_element_type();
      }
      elwise_instantiate_with_child<I>(child, child_tp, cuda_ckb, 0, new_dst_tp,
                                       dst_arrmeta, &new_src_tp[0], src_arrmeta,
                                       kernreq | kernel_request_cuda_device,
                                       ectx, kwds, tp_vars);
      // The return is the ckb_offset for the ckb that was passed in,
      // not the CUDA ckb we just created for the CUDA memory.
      return ckb_offset;
    }
  }
#endif // __CUDACC__

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
      return child->instantiate(child, child_tp, ckb, ckb_offset, dst_tp,
                                dst_arrmeta, src_tp, src_arrmeta, kernreq, ectx,
                                kwds, tp_vars);
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
      return elwise_instantiate_with_child<fixed_dim_type_id, fixed_dim_type_id,
                                           I>(
          child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
          src_arrmeta, kernreq, ectx, kwds, tp_vars);
    } else if (src_all_strided_or_var) {
      return elwise_instantiate_with_child<fixed_dim_type_id, var_dim_type_id,
                                           I>(
          child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
          src_arrmeta, kernreq, ectx, kwds, tp_vars);
    } else {
      // TODO
    }
    break;
  case var_dim_type_id:
    if (src_all_strided_or_var) {
      return elwise_instantiate_with_child<var_dim_type_id, fixed_dim_type_id,
                                           I>(
          child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
          src_arrmeta, kernreq, ectx, kwds, tp_vars);
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

template <int I>
typename std::enable_if<I == 10, intptr_t>::type
nd::functional::elwise_instantiate_with_child(
    const arrfunc_type_data *DYND_UNUSED(child),
    const arrfunc_type *DYND_UNUSED(child_tp), void *DYND_UNUSED(ckb),
    intptr_t DYND_UNUSED(ckb_offset), const ndt::type &DYND_UNUSED(dst_tp),
    const char *DYND_UNUSED(dst_arrmeta), const ndt::type *DYND_UNUSED(src_tp),
    const char *const *DYND_UNUSED(src_arrmeta),
    dynd::kernel_request_t DYND_UNUSED(kernreq),
    const eval::eval_context *DYND_UNUSED(ectx),
    const dynd::nd::array &DYND_UNUSED(kwds),
    const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  throw std::runtime_error("ERROR");
}

template <type_id_t dst_type_id, type_id_t src_type_id, int I>
intptr_t nd::functional::elwise_instantiate_with_child(
    const arrfunc_type_data *child, const arrfunc_type *child_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, ndt::type> &tp_vars)
{
  switch (child_tp->get_npos()) {
  case 0:
    return kernels::elwise_ck<dst_type_id, src_type_id, 0, I>::instantiate(
        child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
        src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case 1:
    return kernels::elwise_ck<dst_type_id, src_type_id, 1, I>::instantiate(
        child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
        src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case 2:
    return kernels::elwise_ck<dst_type_id, src_type_id, 2, I>::instantiate(
        child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
        src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case 3:
    return kernels::elwise_ck<dst_type_id, src_type_id, 3, I>::instantiate(
        child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
        src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case 4:
    return kernels::elwise_ck<dst_type_id, src_type_id, 4, I>::instantiate(
        child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
        src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case 5:
    return kernels::elwise_ck<dst_type_id, src_type_id, 5, I>::instantiate(
        child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
        src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case 6:
    return kernels::elwise_ck<dst_type_id, src_type_id, 6, I>::instantiate(
        child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
        src_arrmeta, kernreq, ectx, kwds, tp_vars);
  default:
    throw runtime_error("elwise with src_count > 6 not implemented yet");
  }
}

template <int I>
intptr_t nd::functional::elwise_instantiate(
    const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type *src_tp,
    const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
    const eval::eval_context *ectx, const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, ndt::type> &tp_vars)
{
  const arrfunc_type_data *child =
      self->get_data_as<dynd::nd::arrfunc>()->get();
  const arrfunc_type *child_tp =
      self->get_data_as<dynd::nd::arrfunc>()->get_type();

  return elwise_instantiate_with_child<I>(
      child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
      src_arrmeta, kernreq, ectx, kwds, tp_vars);
}

template intptr_t nd::functional::elwise_instantiate<0>(
    const arrfunc_type_data *child, const arrfunc_type *child_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    dynd::kernel_request_t kernreq, const eval::eval_context *ectx,
    const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, ndt::type> &tp_vars);

template intptr_t nd::functional::elwise_instantiate_with_child<0>(
    const arrfunc_type_data *child, const arrfunc_type *child_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    dynd::kernel_request_t kernreq, const eval::eval_context *ectx,
    const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, ndt::type> &tp_vars);

int nd::functional::elwise_resolve_dst_type(
    const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
    intptr_t nsrc, const ndt::type *src_tp, int throw_on_error,
    ndt::type &dst_tp, const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, ndt::type> &tp_vars)
{
  const arrfunc_type_data *child =
      self->get_data_as<dynd::nd::arrfunc>()->get();
  const arrfunc_type *child_tp =
      self->get_data_as<dynd::nd::arrfunc>()->get_type();

  return elwise_resolve_dst_type_with_child(
      child, child_tp, nsrc, src_tp, throw_on_error, dst_tp, kwds, tp_vars);
}
