//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ROLLING_ARRFUNC_HPP_
#define _DYND__ROLLING_ARRFUNC_HPP_

#include <dynd/config.hpp>
#include <dynd/array.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/types/arrfunc_type.hpp>

namespace dynd {

/**
 * Create an arrfunc which applies a given window_op in a
 * rolling window fashion.
 *
 * \param out_af  The output arrfunc which is filled.
 * \param window_op  A arrfunc object which should be applied to each
 *                   window. The types of this ckernel must match appropriately
 *                   with `dst_tp` and `src_tp`.
 * \param window_size  The size of the rolling window.
 */
void make_rolling_arrfunc(arrfunc_type_data *out_af, const nd::arrfunc &window_op,
                          intptr_t window_size);

inline nd::arrfunc make_rolling_arrfunc(const nd::arrfunc &window_op,
                                        intptr_t window_size)
{
    nd::array af = nd::empty(ndt::make_arrfunc());
    make_rolling_arrfunc(
        reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()),
        window_op, window_size);
    af.flag_as_immutable();
    return af;
}

} // namespace dynd

#endif // _DYND__ROLLING_ARRFUNC_HPP_
