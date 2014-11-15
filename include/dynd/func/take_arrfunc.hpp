//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/array.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/types/arrfunc_old_type.hpp>
#include <dynd/kernels/expr_kernels.hpp>

namespace dynd { namespace kernels {

/**
 * Create an arrfunc which applies either a boolean masked or
 * an indexed take/"fancy indexing" operation.
 */
nd::arrfunc make_take_arrfunc();

}} // namespace dynd::kernels
