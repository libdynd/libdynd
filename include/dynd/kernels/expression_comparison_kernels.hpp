//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__EXPRESSION_COMPARISON_KERNELS_HPP_
#define _DYND__EXPRESSION_COMPARISON_KERNELS_HPP_

#include <dynd/kernels/comparison_kernels.hpp>

namespace dynd {

/**
 * Makes a kernel which does a comparison when
 * at least one of src0_dt and src1_dt is an
 * expr_kind type.
 */
size_t make_expression_comparison_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                const ndt::type& src0_dt, const char *src0_arrmeta,
                const ndt::type& src1_dt, const char *src1_arrmeta,
                comparison_type_t comptype,
                const eval::eval_context *ectx);

} // namespace dynd

#endif // _DYND__EXPRESSION_COMPARISON_KERNELS_HPP_

