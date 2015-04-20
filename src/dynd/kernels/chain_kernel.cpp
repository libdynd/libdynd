//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/chain_kernel.hpp>

using namespace std;
using namespace dynd;

/**
 * Instantiate the chaining of arrfuncs ``first`` and ``second``, using
 * ``buffer_tp`` as the intermediate type, without creating a temporary chained
 * arrfunc.
 */
intptr_t nd::functional::chain_kernel::instantiate(
    const arrfunc_type_data *af_self,
    const ndt::arrfunc_type *DYND_UNUSED(af_tp), char *data, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds,
    const std::map<nd::string, ndt::type> &tp_vars)
{
  const static_data *static_data = af_self->get_data_as<struct static_data>();

  const arrfunc_type_data *first = static_data->first.get();
  const ndt::arrfunc_type *first_tp = static_data->first.get_type();

  const arrfunc_type_data *second = static_data->second.get();
  const ndt::arrfunc_type *second_tp = static_data->second.get_type();

  const ndt::type &buffer_tp = static_data->buffer_tp;

  intptr_t root_ckb_offset = ckb_offset;
  chain_kernel *self = make(ckb, kernreq, ckb_offset, static_data->buffer_tp);
  ckb_offset =
      first->instantiate(first, first_tp, data, ckb, ckb_offset, buffer_tp,
                         self->buffer_arrmeta.get(), 1, src_tp, src_arrmeta,
                         kernreq, ectx, kwds, tp_vars);
  self = get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
                  root_ckb_offset);
  self->second_offset = ckb_offset - root_ckb_offset;
  const char *buffer_arrmeta = self->buffer_arrmeta.get();
  return second->instantiate(second, second_tp, data + first->data_size, ckb,
                             ckb_offset, dst_tp, dst_arrmeta, 1, &buffer_tp,
                             &buffer_arrmeta, kernreq, ectx, kwds, tp_vars);
}

void nd::functional::chain_kernel::resolve_dst_type(
    const arrfunc_type_data *DYND_UNUSED(self),
    const ndt::arrfunc_type *self_tp, char *DYND_UNUSED(data),
    ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
    const ndt::type *DYND_UNUSED(src_tp), const nd::array &DYND_UNUSED(kwds),
    const std::map<nd::string, ndt::type> &tp_vars)
{
  dst_tp = ndt::substitute(self_tp->get_return_type(), tp_vars, true);
}