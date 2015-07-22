//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/chain_kernel.hpp>

using namespace std;
using namespace dynd;

void nd::functional::chain_kernel::resolve_dst_type(
    char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
    char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
    const ndt::type *DYND_UNUSED(src_tp), const nd::array &DYND_UNUSED(kwds),
    const std::map<nd::string, ndt::type> &tp_vars)
{
  dst_tp = ndt::substitute(dst_tp, tp_vars, true);
}

/**
 * Instantiate the chaining of callables ``first`` and ``second``, using
 * ``buffer_tp`` as the intermediate type, without creating a temporary chained
 * callable.
 */
intptr_t nd::functional::chain_kernel::instantiate(
    char *static_data, size_t data_size, char *data, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds,
    const std::map<nd::string, ndt::type> &tp_vars)
{
  const struct static_data *static_data_x =
      reinterpret_cast<struct static_data *>(static_data);

  callable_type_data *first =
      const_cast<callable_type_data *>(static_data_x->first.get());
  callable_type_data *second =
      const_cast<callable_type_data *>(static_data_x->second.get());

  const ndt::type &buffer_tp = static_data_x->buffer_tp;

  intptr_t root_ckb_offset = ckb_offset;
  chain_kernel *self = make(ckb, kernreq, ckb_offset, static_data_x->buffer_tp);
  ckb_offset =
      first->instantiate(first->static_data, data_size, data, ckb, ckb_offset,
                         buffer_tp, self->buffer_arrmeta.get(), 1, src_tp,
                         src_arrmeta, kernreq, ectx, kwds, tp_vars);
  self = get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
                  root_ckb_offset);
  self->second_offset = ckb_offset - root_ckb_offset;
  const char *buffer_arrmeta = self->buffer_arrmeta.get();
  return second->instantiate(second->static_data, data_size - first->data_size,
                             data + first->data_size, ckb, ckb_offset, dst_tp,
                             dst_arrmeta, 1, &buffer_tp, &buffer_arrmeta,
                             kernreq, ectx, kwds, tp_vars);
}