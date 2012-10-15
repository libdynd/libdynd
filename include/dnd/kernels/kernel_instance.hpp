//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__KERNEL_INSTANCE_HPP_
#define _DND__KERNEL_INSTANCE_HPP_

#include <dnd/auxiliary_data.hpp>

namespace dynd {

typedef void (*nullary_operation_t)(char *dst, intptr_t dst_stride,
                        intptr_t count, const AuxDataBase *auxdata);

typedef void (*unary_operation_t)(char *dst, intptr_t dst_stride,
                        const char *src0, intptr_t src0_stride,
                        intptr_t count, const AuxDataBase *auxdata);

typedef void (*binary_operation_t)(char *dst, intptr_t dst_stride,
                        const char *src0, intptr_t src0_stride,
                        const char *src1, intptr_t src1_stride,
                        intptr_t count, const AuxDataBase *auxdata);

typedef bool (*single_compare_operation_t)(const char *src0, const char *src1,
                        const AuxDataBase *auxdata);

/**
 * This class holds an instance of a kernel function, with its
 * associated auxiliary data. The object is non-copyable, just
 * like the auxiliary_data object, to avoid inefficient copies.
 */
template<typename FT>
class kernel_instance {
    kernel_instance& operator=(const kernel_instance&);
public:
    kernel_instance()
        : kernel(0)
    {
    }
    // Copying a kernel_instance clones the auxiliary data
    kernel_instance(const kernel_instance& rhs)
        : kernel(rhs.kernel)
    {
        auxdata.clone_from(rhs.auxdata);
    }

    void swap(kernel_instance& rhs) {
        std::swap(kernel, rhs.kernel);
        auxdata.swap(rhs.auxdata);
    }

    void copy_from(const kernel_instance& rhs) {
        kernel = rhs.kernel;
        auxdata.clone_from(rhs.auxdata);
    }

    void borrow_from(const kernel_instance& rhs) {
        kernel = rhs.kernel;
        auxdata.borrow_from(rhs.auxdata);
    }

    FT kernel;
    auxiliary_data auxdata;
};

} // namespace dynd;

#endif // _DND__KERNEL_INSTANCE_HPP_
