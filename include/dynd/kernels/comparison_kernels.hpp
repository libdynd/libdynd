//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__COMPARISON_KERNELS_HPP_
#define _DYND__COMPARISON_KERNELS_HPP_

#include <dynd/types/type_id.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/eval/eval_context.hpp>

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


// The predicate function type uses 'int' instead of 'bool' so
// that it is a well-specified C function pointer.
typedef int (*binary_single_predicate_t)(const char *src0, const char *src1,
                        ckernel_prefix *extra);

/**
 * See the ckernel_builder class documentation
 * for details about how kernels can be built and
 * used.
 *
 * This kernel type is for kernels which perform
 * a comparison between one type/arrmeta value
 * and a different type/arrmeta value.
 */
class comparison_ckernel_builder : public ckernel_builder {
public:
    comparison_ckernel_builder()
        : ckernel_builder()
    {
    }

    inline binary_single_predicate_t get_function() const {
        return get()->get_function<binary_single_predicate_t>();
    }

    /** Calls the function to do the comparison */
    inline bool operator()(const char *src0, const char *src1) {
        ckernel_prefix *kdp = get();
        binary_single_predicate_t fn = kdp->get_function<binary_single_predicate_t>();
        return fn(src0, src1, kdp) != 0;
    }
};

/**
 * Creates a comparison kernel for two type/arrmeta
 * pairs. This adds the kernel at the 'out_offset' position
 * in 'out's data, as part of a hierarchy matching the
 * type's hierarchy.
 *
 * This function should always be called with this == src0_dt first,
 * and types which don't support the particular assignment should
 * then call the corresponding function with this == src1_dt.
 *
 * \param out  The hierarchical assignment kernel being constructed.
 * \param offset_out  The offset within 'out'.
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
size_t make_comparison_kernel(ckernel_builder *out, size_t offset_out,
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
 * \param out  The hierarchical assignment kernel being constructed.
 * \param offset_out  The offset within 'out'.
 * \param src0_type_id  The first dynd type id.
 * \param src1_type_id  The second dynd type id.
 * \param comptype  The type of comparison to do.
 */
size_t make_builtin_type_comparison_kernel(ckernel_builder *out,
                                           size_t offset_out,
                                           type_id_t src0_type_id,
                                           type_id_t src1_type_id,
                                           comparison_type_t comptype);

} // namespace dynd

#endif // _DYND__COMPARISON_KERNELS_HPP_
