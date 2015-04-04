//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/arrmeta_holder.hpp>
#include <dynd/kernels/elwise.hpp>
#include <dynd/kernels/outer.hpp>

using namespace std;
using namespace dynd;

intptr_t nd::functional::outer_ck::instantiate(
    const arrfunc_type_data *child, const arrfunc_type *child_tp,
    char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    dynd::kernel_request_t kernreq, const eval::eval_context *ectx,
    const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, ndt::type> &tp_vars)
{
  intptr_t ndim = 0;
  for (intptr_t i = 0; i < child_tp->get_npos(); ++i) {
    ndim += src_tp[i].get_ndim();
  }

  std::vector<ndt::type> new_src_tp(child_tp->get_npos());
  std::vector<const char *> new_src_arrmeta;

  arrmeta_holder *new_src_arrmeta_holder =
      new arrmeta_holder[child_tp->get_npos()];
  for (intptr_t i = 0, j = 0; i < child_tp->get_npos(); ++i) {
    ndt::type old_tp = src_tp[i];
    new_src_tp[i] = old_tp.with_new_axis(0, j);
    new_src_tp[i] = new_src_tp[i].with_new_axis(
        new_src_tp[i].get_ndim(), ndim - new_src_tp[i].get_ndim());
    ndt::type new_tp = new_src_tp[i];

    new (new_src_arrmeta_holder + i) arrmeta_holder(new_tp);
    char *new_arrmeta = new_src_arrmeta_holder[i].get();

    intptr_t k;
    for (k = 0; k < j; ++k) {
      size_stride_t *smd = reinterpret_cast<size_stride_t *>(new_arrmeta);
      smd->dim_size = 1;
      smd->stride = 0;
      new_tp = new_tp.get_type_at_dimension(&new_arrmeta, 1);
    }
    j += old_tp.get_ndim();
    for (; old_tp.get_ndim(); ++k) {
      if (new_tp.get_kind() == memory_kind) {
        new_tp.extended<base_memory_type>()
            ->get_element_type()
            .extended<base_dim_type>()
            ->arrmeta_copy_construct_onedim(new_arrmeta, src_arrmeta[i], NULL);
      } else {
        new_tp.extended<base_dim_type>()->arrmeta_copy_construct_onedim(
            new_arrmeta, src_arrmeta[i], NULL);
      }
      old_tp =
          old_tp.get_type_at_dimension(const_cast<char **>(src_arrmeta + i), 1);
      new_tp = new_tp.get_type_at_dimension(&new_arrmeta, 1);
    }
    for (; new_tp.get_ndim();) {
      size_stride_t *smd = reinterpret_cast<size_stride_t *>(new_arrmeta);
      smd->dim_size = 1;
      smd->stride = 0;
      new_tp = new_tp.get_type_at_dimension(&new_arrmeta, 1);
    }

    new_src_arrmeta.push_back(new_src_arrmeta_holder[i].get());
  }

  ckb_offset = elwise_virtual_ck::instantiate(
      child, child_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
      new_src_tp.data(), new_src_arrmeta.data(), kernreq, ectx, kwds, tp_vars);
  delete[] new_src_arrmeta_holder;

  return ckb_offset;
}

void nd::functional::outer_ck::resolve_dst_type(
    const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
    char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t nsrc,
    const ndt::type *src_tp, const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, ndt::type> &tp_vars)
{
  const arrfunc_type_data *child =
      self->get_data_as<dynd::nd::arrfunc>()->get();
  const arrfunc_type *child_tp =
      self->get_data_as<dynd::nd::arrfunc>()->get_type();

  if (child->resolve_dst_type != NULL) {
    child->resolve_dst_type(child, child_tp, NULL, dst_tp, nsrc, src_tp, kwds,
                            tp_vars);
  } else {
    dst_tp = ndt::substitute(child_tp->get_return_type(), tp_vars, false);
  }

  ndt::type tp = dst_tp.without_memory_type();
  for (intptr_t i = nsrc - 1; i >= 0; --i) {
    if (!src_tp[i].without_memory_type().is_scalar()) {
      tp = src_tp[i].without_memory_type().with_replaced_dtype(tp);
    }
  }
  if (dst_tp.get_kind() == memory_kind) {
    dst_tp =
        dst_tp.extended<base_memory_type>()->with_replaced_storage_type(tp);
  } else {
    dst_tp = tp;
  }
}