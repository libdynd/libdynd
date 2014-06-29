//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND_FUNC_CHAIN_ARRFUNC_HPP
#define DYND_FUNC_CHAIN_ARRFUNC_HPP

#include <dynd/func/arrfunc.hpp>

namespace dynd {

/**
 * Returns an arrfunc which chains the two arrfuncs together.
 * The buffer used to connect them is made out of the provided ``buf_tp``.
 */
nd::arrfunc make_chain_arrfunc(const nd::arrfunc &first,
                               const nd::arrfunc &second,
                               const ndt::type &buf_tp = ndt::type());

void make_chain_arrfunc(const nd::arrfunc &first, const nd::arrfunc &second,
                        const ndt::type &buf_tp, arrfunc_type_data *out_af);

/**
 * Instantiate the chaining of arrfuncs ``first`` and ``second``, using
 * ``buf_tp`` as the intermediate type, without creating a temporary chained
 * arrfunc.
 */
intptr_t make_chain_buf_tp_ckernel(
    const arrfunc_type_data *first, const arrfunc_type_data *second,
    const ndt::type &buf_tp, dynd::ckernel_builder *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx);

} // namespace dynd

#endif // DYND_FUNC_CHAIN_ARRFUNC_HPP
