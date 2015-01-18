//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//


#include <dynd/arrmeta_holder.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/outer.hpp>

nd::arrfunc nd::functional::outer(const arrfunc &child)
{
  return nd::arrfunc(outer_make_type(child.get_type()), child,
                     &outer_instantiate, NULL, &outer_resolve_dst_type);
}

ndt::type nd::functional::outer_make_type(const arrfunc_type *child_tp)
{
  return elwise_make_type(child_tp);
}

intptr_t nd::functional::outer_instantiate(
    const arrfunc_type_data *child, const arrfunc_type *child_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
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

  arrmeta_holder *new_src_arrmeta_holder = new arrmeta_holder[child_tp->get_npos()];
  for (intptr_t i = 0, j = 0; i < child_tp->get_npos(); ++i) {
    ndt::type old_tp = src_tp[i];
    new_src_tp[i] = old_tp.with_new_axis(0, j);
    new_src_tp[i] = new_src_tp[i].with_new_axis(new_src_tp[i].get_ndim(), ndim - new_src_tp[i].get_ndim());
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
      new_tp.extended<base_dim_type>()->arrmeta_copy_construct_onedim(
          new_arrmeta, src_arrmeta[i], NULL);
      old_tp = old_tp.get_type_at_dimension(const_cast<char **>(src_arrmeta + i), 1);
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

  return elwise_instantiate(child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta,
                            new_src_tp.data(), new_src_arrmeta.data(), kernreq, ectx, kwds, tp_vars);
}

int nd::functional::outer_resolve_dst_type_with_child(
    const arrfunc_type_data *DYND_UNUSED(child), const arrfunc_type *child_tp,
    intptr_t nsrc, const ndt::type *src_tp, int DYND_UNUSED(throw_on_error),
    ndt::type &dst_tp, const dynd::nd::array &DYND_UNUSED(kwds),
    const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  std::vector<intptr_t> shape;
  for (intptr_t i = 0; i < nsrc; ++i) {
    shape.push_back(src_tp[i].get_dim_size(NULL, NULL));
  }

  dst_tp = child_tp->get_return_type();
  for (size_t i = 0; i < shape.size(); ++i) {
    dst_tp = ndt::make_fixed_dim(shape[i], dst_tp);
  }

  return 1;
}

int nd::functional::outer_resolve_dst_type(
    const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
    intptr_t nsrc, const ndt::type *src_tp, int throw_on_error,
    ndt::type &dst_tp, const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, ndt::type> &tp_vars)
{
  const arrfunc_type_data *child =
      self->get_data_as<dynd::nd::arrfunc>()->get();
  const arrfunc_type *child_tp =
      self->get_data_as<dynd::nd::arrfunc>()->get_type();

  return outer_resolve_dst_type_with_child(
      child, child_tp, nsrc, src_tp, throw_on_error, dst_tp, kwds, tp_vars);
}
