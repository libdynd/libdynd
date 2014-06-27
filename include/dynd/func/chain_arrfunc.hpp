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

} // namespace dynd

#endif // DYND_FUNC_CHAIN_ARRFUNC_HPP
