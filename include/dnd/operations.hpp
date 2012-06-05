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

typedef void (*nullary_operation_t)(void *dst, intptr_t dst_stride,
                        intptr_t count, const AuxDataBase *auxdata);

typedef void (*unary_operation_t)(void *dst, intptr_t dst_stride,
                        const void *src0, intptr_t src0_stride,
                        intptr_t count, const AuxDataBase *auxdata);

typedef void (*binary_operation_t)(void *dst, intptr_t dst_stride,
                        const void *src0, intptr_t src0_stride,
                        const void *src1, intptr_t src1_stride,
                        intptr_t count, const AuxDataBase *auxdata);

/**
 * This class holds an instance of a kernel function, with its
 * associated auxiliary data. The object is non-copyable, just
 * like the auxiliary_data object, to avoid inefficient copies.
 * Non-copyability is implicit because auxiliary_data is non-copyable.
 */
template<typename FT>
struct kernel_instance {
    FT kernel;
    auxiliary_data auxdata;
};

} // namespace dnd;



#endif // _DND__OPERATIONS_HPP_
