//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/shape_tools.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/pointer_type.hpp>

namespace dynd {

/**
 * Create an arrfunc which applies an indexed take/"fancy indexing" operation,
 * but stores the pointers.
 *
 * \param out_af  The arrfunc to fill.
 */
void make_take_by_pointer_arrfunc(arrfunc_type_data *out_af);

nd::arrfunc make_take_by_pointer_arrfunc();

} // namespace dynd
