//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/comparison_kernels.hpp>

namespace dynd {

/**
 * Makes a kernel which does a comparison when
 * at least one of src0_dt and src1_dt is an
 * expr_kind type.
 */
DYND_API size_t make_expression_comparison_kernel(
                void *ckb, intptr_t ckb_offset,
                const ndt::type& src0_dt, const char *src0_arrmeta,
                const ndt::type& src1_dt, const char *src1_arrmeta,
                comparison_type_t comptype,
                const eval::eval_context *ectx);

} // namespace dynd
