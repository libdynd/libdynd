//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ROLLING_CKERNEL_DEFERRED_HPP_
#define _DYND__ROLLING_CKERNEL_DEFERRED_HPP_

#include <dynd/config.hpp>
#include <dynd/array.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/types/arrfunc_type.hpp>

namespace dynd {

/**
 * Create a deferred ckernel which applies a given window_op in a
 * rolling window fashion.
 *
 * \param out_ckd  The output arrfunc which is filled.
 * \param dst_tp  The destination type for the resulting ckernel_deferred.
 * \param src_tp  The source type for the resulting ckernel_deferred.
 * \param window_op  A arrfunc object which should be applied to each
 *                   window. The types of this ckernel must match appropriately
 *                   with `dst_tp` and `src_tp`.
 * \param window_size  The size of the rolling window.
 */
void make_rolling_ckernel_deferred(arrfunc *out_ckd,
                                   const ndt::type &dst_tp,
                                   const ndt::type &src_tp,
                                   const nd::array &window_op, intptr_t window_size);

inline nd::array make_rolling_ckernel_deferred(const ndt::type &dst_tp,
                                               const ndt::type &src_tp,
                                               const nd::array &window_op,
                                               intptr_t window_size)
{
    nd::array ckd = nd::empty(ndt::make_arrfunc());
    make_rolling_ckernel_deferred(
        reinterpret_cast<arrfunc *>(ckd.get_readwrite_originptr()),
        dst_tp, src_tp, window_op, window_size);
    return ckd;
}

} // namespace dynd

#endif // _DYND__ROLLING_CKERNEL_DEFERRED_HPP_
