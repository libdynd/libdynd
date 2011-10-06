//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__OPERATIONS_HPP_
#define _DND__OPERATIONS_HPP_

namespace dnd {

typedef void (*nullary_operation_t)(void *dst, intptr_t dst_stride,
                        intptr_t count, const auxiliary_data *auxdata);

typedef void (*unary_operation_t)(void *dst, intptr_t dst_stride,
                        const void *src0, intptr_t src0_stride,
                        intptr_t count, const auxiliary_data *auxdata);

typedef void (*binary_operation_t)(void *dst, intptr_t dst_stride,
                        const void *src0, intptr_t src0_stride,
                        const void *src1, intptr_t src1_stride,
                        intptr_t count, const auxiliary_data *auxdata);

} // namespace dnd;



#endif // _DND__OPERATIONS_HPP_
