//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__OP_ADD_HPP_
#define _DND__OP_ADD_HPP_

namespace dnd {

typedef void (*binary_operation_t)(void *dst, intptr_t dst_stride,
                        const void *src0, intptr_t src0_stride,
                        const void *src1, intptr_t src1_stride,
                        intptr_t count, const auxiliary_data *auxdata);

class ndarray;

ndarray add(const ndarray& op0, const ndarray& op1);
ndarray subtract(const ndarray& op0, const ndarray& op1);
ndarray multiply(const ndarray& op0, const ndarray& op1);
ndarray divide(const ndarray& op0, const ndarray& op1);

} // namespace dnd;

#endif // _DTYPE_ASSIGN_HPP_
