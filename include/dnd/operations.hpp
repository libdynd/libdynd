//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__OPERATIONS_HPP_
#define _DND__OPERATIONS_HPP_

#include <dnd/auxiliary_data.hpp>

namespace dnd {

typedef void (*nullary_operation_t)(char *dst, intptr_t dst_stride,
                        intptr_t count, const AuxDataBase *auxdata);

typedef void (*unary_operation_t)(char *dst, intptr_t dst_stride,
                        const char *src0, intptr_t src0_stride,
                        intptr_t count, const AuxDataBase *auxdata);

typedef void (*binary_operation_t)(char *dst, intptr_t dst_stride,
                        const char *src0, intptr_t src0_stride,
                        const char *src1, intptr_t src1_stride,
                        intptr_t count, const AuxDataBase *auxdata);

/**
 * This class holds an instance of a kernel function, with its
 * associated auxiliary data. The object is non-copyable, just
 * like the auxiliary_data object, to avoid inefficient copies.
 */
template<typename FT>
class kernel_instance {
    kernel_instance& operator=(const kernel_instance&);
public:
    kernel_instance() {
    }
    // Copying a kernel_instance clones the auxiliary data
    kernel_instance(const kernel_instance& rhs)
        : kernel(rhs.kernel)
    {
        rhs.auxdata.clone_into(auxdata);
    }

    void swap(kernel_instance& rhs) {
        using namespace std;
        swap(kernel, rhs.kernel);
        auxdata.swap(rhs.auxdata);
    }

    FT kernel;
    auxiliary_data auxdata;
};

} // namespace dnd;



#endif // _DND__OPERATIONS_HPP_
