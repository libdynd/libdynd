//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/array.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {

/**
 * Lifts the provided arrfunc, so it broadcasts all the arguments.
 *
 * \param child_af  The arrfunc to be lifted.
 */
nd::arrfunc lift_arrfunc(const nd::arrfunc &child_af);

int resolve_lifted_dst_type(const arrfunc_type_data *self,
                            const arrfunc_type *af_tp, intptr_t nsrc,
                            const ndt::type *src_tp, int throw_on_error,
                            ndt::type &out_dst_tp, const nd::array &kwds);

} // namespace dynd
