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
 * Create an arrfunc which applies either a boolean masked or
 * an indexed take/"fancy indexing" operation.
 *
 * \param out_af  The arrfunc to fill.
 */
void make_take_arrfunc(arrfunc_type_data *out_af);

inline nd::arrfunc make_take_arrfunc()
{
    nd::array af = nd::empty(ndt::make_arrfunc());
    make_take_arrfunc(
        reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()));
    af.flag_as_immutable();
    return af;
}

}} // namespace dynd::kernels

#endif // _DYND__TAKE_ARRFUNC_HPP_
