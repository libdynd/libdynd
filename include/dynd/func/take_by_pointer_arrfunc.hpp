//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND_FUNC_APPLY_ARRFUNC_HPP
#define DYND_FUNC_APPLY_ARRFUNC_HPP

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/expr_kernels.hpp>

namespace dynd {

void make_take_by_pointer_arrfunc(arrfunc_type_data *out_af);

nd::arrfunc make_take_by_pointer_arrfunc();

} // namespace dynd

#endif // DYND_FUNC_APPLY_ARRFUNC_HPP
