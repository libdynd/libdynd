//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__TAKE_ARRFUNC_HPP_
#define _DYND__TAKE_ARRFUNC_HPP_

#include <dynd/config.hpp>
#include <dynd/array.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/kernels/expr_kernels.hpp>

namespace dynd { namespace kernels {

/**
 * Create a deferred ckernel which applies a given window_op in a
 * rolling window fashion.
 *
 * \param out_af  The output arrfunc which is filled.
 * \param dst_tp  The destination type for the resulting arrfunc.
 * \param src_tp  The source type for the resulting arrfunc.
 * \param window_size  The size of the rolling window.
 */
void make_take_arrfunc(arrfunc *out_af, const ndt::type &dst_tp,
                       const ndt::type &src_tp, const ndt::type &mask_tp);

inline nd::array make_take_arrfunc(const ndt::type &dst_tp,
                                   const ndt::type &src_tp,
                                   const ndt::type &mask_tp)
{
    nd::array af = nd::empty(ndt::make_arrfunc());
    make_take_arrfunc(reinterpret_cast<arrfunc *>(af.get_readwrite_originptr()),
                      dst_tp, src_tp, mask_tp);
    return af;
}

}} // namespace dynd::kernels

#endif // _DYND__TAKE_ARRFUNC_HPP_
