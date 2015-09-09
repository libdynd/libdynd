//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/type_id.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/math.hpp>

namespace dynd {

namespace ndt {
    class type;
} // namespace ndt

enum comparison_type_t {
    /**
     * A less than operation suitable for sorting
     * (one of a < b or b < a must be true when a != b).
     */
    comparison_type_sorting_less,
    /** Standard comparisons */
    comparison_type_less,
    comparison_type_less_equal,
    comparison_type_equal,
    comparison_type_not_equal,
    comparison_type_greater_equal,
    comparison_type_greater
};

/**
 * Creates a comparison kernel for two type/arrmeta
 * pairs. This adds the kernel at the 'ckb_offset' position
 * in 'ckb's data, as part of a hierarchy matching the
 * type's hierarchy.
 *
 * This function should always be called with this == src0_dt first,
 * and types which don't support the particular assignment should
 * then call the corresponding function with this == src1_dt.
 *
 * \param ckb  The hierarchical assignment kernel being constructed.
 * \param ckb_offset  The offset within 'ckb'.
 * \param src0_dt  The first dynd type.
 * \param src0_arrmeta  Arrmeta for the first data.
 * \param src1_dt  The second dynd type.
 * \param src1_arrmeta  Arrmeta for the second data
 * \param comptype  The type of comparison to do.
 * \param ectx  DyND evaluation context.
 *
 * \returns  The offset within 'out' immediately after the
 *           created kernel.
 */
DYND_API size_t make_comparison_kernel(void *ckb, intptr_t ckb_offset,
                                       const ndt::type &src0_dt,
                                       const char *src0_arrmeta,
                                       const ndt::type &src1_dt,
                                       const char *src1_arrmeta,
                                       comparison_type_t comptype,
                                       const eval::eval_context *ectx);

/**
 * Creates a comparison kernel that compares the two builtin
 * types.
 *
 * \param ckb  The hierarchical assignment kernel being constructed.
 * \param ckb_offset  The offset within 'ckb'.
 * \param src0_type_id  The first dynd type id.
 * \param src1_type_id  The second dynd type id.
 * \param comptype  The type of comparison to do.
 */
DYND_API size_t make_builtin_type_comparison_kernel(void *ckb,
                                                    intptr_t ckb_offset,
                                                    type_id_t src0_type_id,
                                                    type_id_t src1_type_id,
                                                    comparison_type_t comptype);

} // namespace dynd
