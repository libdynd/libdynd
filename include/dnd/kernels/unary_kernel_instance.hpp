//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__UNARY_KERNEL_INSTANCE_HPP_
#define _DND__UNARY_KERNEL_INSTANCE_HPP_

#include <dnd/kernels/kernel_instance.hpp>

namespace dnd {

/**
 * Unary operations support four specializations as a standardized
 * set, in the order defined by unary_specialization_t. Use the
 * function get_unary_specialization to get an index into an
 * instance of this table.
 */
typedef unary_operation_t specialized_unary_operation_table_t[4];

enum unary_specialization_t {
    // Arbitrary strides
    general_unary_specialization,
    // Both src and dst have stride of zero (always only one element)
    scalar_unary_specialization,
    // Both src and dst are contiguous
    contiguous_unary_specialization,
    // The src stride is zero, the dst stride is contiguous
    scalar_to_contiguous_unary_specialization
};

// Given strides and element sizes, returns the appropriate unary_specialization enum value.
inline unary_specialization_t get_unary_specialization(intptr_t dst_stride, intptr_t dst_element_size,
                        intptr_t src_stride, intptr_t src_element_size)
{
    // The idea of this expression is to have no branches, just a deterministic calculation
    return static_cast<unary_specialization_t>(
                (((dst_stride == dst_element_size)&  // dst is contiguous
                  ((src_stride == 0)|                // src is scalar
                   (src_stride == src_element_size)  // src is contiguous
                 )) << 1
                ) |
                ((src_stride == 0)&                  // src is scalar
                  ((dst_stride == 0)|                // dst is scalar
                   (dst_stride == dst_element_size)  // dst is contiguous
                  )
                ));
}

/**
 * This struct holds a pointer to a specialized_unary_operation_table_t
 * array, as well as auxiliary data for the kernels.
 */
struct unary_specialization_kernel_instance {
    unary_specialization_kernel_instance()
        : specializations(0), auxdata()
    {
    }
    // Copying a kernel_instance clones the auxiliary data
    unary_specialization_kernel_instance(const unary_specialization_kernel_instance& rhs)
        : specializations(rhs.specializations), auxdata()
    {
        auxdata.clone_from(rhs.auxdata);
    }
    unary_specialization_kernel_instance& operator=(const unary_specialization_kernel_instance& rhs)
    {
        specializations = rhs.specializations;
        auxdata.clone_from(rhs.auxdata);
        return *this;
    }

    void swap(unary_specialization_kernel_instance& rhs) {
        std::swap(specializations, rhs.specializations);
        auxdata.swap(rhs.auxdata);
    }

    void borrow_from(const unary_specialization_kernel_instance& rhs)
    {
        specializations = rhs.specializations;
        auxdata.borrow_from(rhs.auxdata);
    }


    /**
     * This grabs the requested unary operation specialization, and
     * borrows the auxiliary data into the output kernel.
     *
     * IMPORTANT: Because the auxiliary data is borrowed, the kernel
     *            instance this writes to should have a shorter lifetime
     *            than this unary_specialization_kernel_instance.
     */
    void borrow_specialization(unary_specialization_t specialization_id, kernel_instance<unary_operation_t>& out_kernel) const
    {
        out_kernel.kernel = specializations[specialization_id];
        out_kernel.auxdata.borrow_from(auxdata);
    }

    /**
     * This grabs the requested unary operation specialization, and
     * clones the auxiliary data into the output kernel.
     *
     * IMPORTANT: Because the auxiliary data is copied, this may
     *            be an expensive operation. An example where this
     *            is necessary is to duplicate a kernel so that
     *            multiple threads may run the same kernel simultaneously.
     */
    void copy_specialization(unary_specialization_t specialization_id, kernel_instance<unary_operation_t>& out_kernel) const
    {
        out_kernel.kernel = specializations[specialization_id];
        out_kernel.auxdata.clone_from(auxdata);
    }

    // The specializations - a pointer to a static array of function pointers
    unary_operation_t *specializations;
    // The auxiliary data which works with all of the specializations
    auxiliary_data auxdata;
};


} // namespace dnd;



#endif // _DND__UNARY_KERNEL_INSTANCE_HPP_
