//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND_FUNC_NEIGHBORHOOD_ARRFUNC_HPP
#define DYND_FUNC_NEIGHBORHOOD_ARRFUNC_HPP

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
 * \param neighborhood_op  An arrfunc object which transforms a neighborhood into
 *                         a single output value. Signature
 *                         '(strided * strided * NH, strided * strided * MSK) -> OUT',
 * \param window_size  The size of the rolling window.
 */
void make_neighborhood_arrfunc(arrfunc_type_data *out_af,
                                 const nd::arrfunc &neighborhood_op,
                                 intptr_t nh_ndim,
                                 const intptr_t *nh_shape,
                                 const intptr_t *nh_centre);

inline nd::arrfunc make_neighborhood_arrfunc(const nd::arrfunc &neighborhood_op,
                            intptr_t nh_ndim,
                            const intptr_t *nh_shape,
                            const intptr_t *nh_centre)
{
  nd::array af = nd::empty(ndt::make_arrfunc());
  make_neighborhood_arrfunc(
      reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()),
      neighborhood_op, nh_ndim, nh_shape, nh_centre);
  af.flag_as_immutable();
  return af;
}

} // namespace dynd

#endif // DYND_FUNC_NEIGHBORHOOD_ARRFUNC_HPP
