//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__SINGLE_COMPARE_KERNEL_INSTANCE_HPP_
#define _DND__SINGLE_COMPARE_KERNEL_INSTANCE_HPP_

#include <dynd/kernels/kernel_instance.hpp>

namespace dynd {

/**
 * Unary operations support four specializations as a standardized
 * set, in the order defined by unary_specialization_t. Use the
 * function get_unary_specialization to get an index into an
 * instance of this table.
 */
typedef single_compare_operation_t single_compare_operation_table_t[6];

enum comparison_id_t {
    less_id,
    less_equal_id,
    equal_id,
    not_equal_id,
    greater_equal_id,
    greater_id
};


/**
 * This struct holds a pointer to a single_compare_operation_table_t
 * array, as well as auxiliary data for the kernels.
 */
struct single_compare_kernel_instance {
public:
    single_compare_kernel_instance()
        : comparisons(0), auxdata()
    {
    }
    // Copying a kernel_instance clones the auxiliary data
    single_compare_kernel_instance(const single_compare_kernel_instance& rhs)
        : comparisons(rhs.comparisons), auxdata()
    {
        auxdata.clone_from(rhs.auxdata);
    }

    single_compare_kernel_instance& operator=(const single_compare_kernel_instance& rhs)
    {
        comparisons = rhs.comparisons;
        auxdata.clone_from(rhs.auxdata);
        return *this;
    }

    void swap(single_compare_kernel_instance& rhs) {
        std::swap(comparisons, rhs.comparisons);
        auxdata.swap(rhs.auxdata);
    }


    /**
     * This grabs the requested single compare operation, and
     * borrows the auxiliary data into the output kernel.
     *
     * IMPORTANT: Because the auxiliary data is borrowed, the kernel
     *            instance this writes to should have a shorter lifetime
     *            than this unary_specialization_kernel_instance.
     */
    void borrow_comparison(comparison_id_t comparison_id, kernel_instance<single_compare_operation_t>& out_kernel)
    {
        out_kernel.kernel = comparisons[comparison_id];
        out_kernel.auxdata.borrow_from(auxdata);
    }

    /**
     * This grabs the requested single compare operation, and
     * clones the auxiliary data into the output kernel.
     *
     * IMPORTANT: Because the auxiliary data is copied, this may
     *            be an expensive operation. An example where this
     *            is necessary is to duplicate a kernel so that
     *            multiple threads may run the same kernel simultaneously.
     */
    void copy_comparison(comparison_id_t comparison_id, kernel_instance<single_compare_operation_t>& out_kernel)
    {
        out_kernel.kernel = comparisons[comparison_id];
        out_kernel.auxdata.clone_from(auxdata);
    }

    // The comparisons - a pointer to a static array of function pointers
    single_compare_operation_t *comparisons;
    // The auxiliary data which works with all of the specializations
    auxiliary_data auxdata;
};


} // namespace dnd;



#endif // _DND__SINGLE_COMPARE_KERNEL_INSTANCE_HPP_
