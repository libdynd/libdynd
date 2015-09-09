//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/comparison_kernels.hpp>
#include <dynd/typed_data_assign.hpp>

namespace dynd {

/**
 * Makes a kernel which lexicographically compares two
 * instances of the same struct/cstruct.
 */
DYND_API size_t make_tuple_comparison_kernel(void *ckb, intptr_t ckb_offset,
                                            const ndt::type &src_tp,
                                            const char *src0_arrmeta,
                                            const char *src1_arrmeta,
                                            comparison_type_t comptype,
                                            const eval::eval_context *ectx);

} // namespace dynd
