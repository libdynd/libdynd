//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

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
    const ndt::type *src_tp, const char *const *,
    dynd::kernel_request_t kernreq, const eval::eval_context *ectx,
    const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, ndt::type> &tp_vars)
{
//  intptr_t new_ndim = child_tp->get_npos();
  ndt::type new_src_tp[2] = {src_tp[0].with_new_axis(1), src_tp[1].with_new_axis(0)};

  size_stride_t new_src_arrmeta0[2];
  new_src_arrmeta0[0].dim_size = 2;
  new_src_arrmeta0[0].stride = sizeof(int);
  new_src_arrmeta0[1].dim_size = 1;
  new_src_arrmeta0[1].stride = 0;

  size_stride_t new_src_arrmeta1[2];
  new_src_arrmeta1[1].dim_size = 2;
  new_src_arrmeta1[1].stride = sizeof(int);
  new_src_arrmeta1[0].dim_size = 1;
  new_src_arrmeta1[0].stride = 0;

  char *new_src_arrmeta[2] = {reinterpret_cast<char *>(new_src_arrmeta0), reinterpret_cast<char *>(new_src_arrmeta1)};

//  std::cout << new_src_tp[0] << std::endl;
//  std::cout << new_src_tp[1] << std::endl;

  return elwise_instantiate(child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta,
                            new_src_tp, new_src_arrmeta, kernreq, ectx, kwds, tp_vars);
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
