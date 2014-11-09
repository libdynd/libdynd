//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/strided_vals.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {

/**
 * Create an arrfunc which applies a given window_op in a
 * rolling window fashion.
 *
 * \param out_af  The output arrfunc which is filled.
 * \param neighborhood_op  An arrfunc object which transforms a neighborhood into
 *                         a single output value. Signature
 *                         '(fixed * fixed * NH, fixed * fixed * MSK) -> OUT',
 */
void make_neighborhood_arrfunc(arrfunc_old_type_data *out_af, const nd::arrfunc &neighborhood_op,
                               intptr_t nh_ndim);

inline nd::arrfunc make_neighborhood_arrfunc(const nd::arrfunc &neighborhood_op,
                            intptr_t nh_ndim)
{
    nd::array af = nd::empty(ndt::make_arrfunc());
    make_neighborhood_arrfunc(reinterpret_cast<arrfunc_old_type_data *>(af.get_readwrite_originptr()),
        neighborhood_op, nh_ndim);
    af.flag_as_immutable();
    return af;
}

inline nd::arrfunc make_neighborhood_arrfunc(const nd::arrfunc &neighborhood_op, const nd::array &mask);

} // namespace dynd
