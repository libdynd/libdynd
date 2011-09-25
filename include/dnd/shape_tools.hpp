//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__SHAPE_TOOLS_HPP_
#define _DND__SHAPE_TOOLS_HPP_

#include <stdint.h>
#include <type_traits>

#include <dnd/ndarray.hpp>

namespace dnd {

/**
 * This function broadcasts the dimensions and strides of 'src' to a given
 * shape, raising an error if it cannot be broadcast.
 */
void broadcast_to_shape(int ndim, const intptr_t *shape,
                int src_ndim, const intptr_t *src_shape, const intptr_t *src_strides,
                intptr_t *out_strides);

/**
 * This function broadcasts the dimensions and strides of 'src' to a given
 * shape, raising an error if it cannot be broadcast.
 */
inline void broadcast_to_shape(int ndim, const intptr_t *shape, const ndarray& op, intptr_t *out_strides) {
    broadcast_to_shape(ndim, shape, op.ndim(), op.shape(), op.strides(), out_strides);
}


/**
 * This function broadcasts the input operands together, populating
 * the output ndim and shape.
 *
 * @param noperands   The number of operands.
 * @param operands    The array of operands.
 * @param out_ndim    The number of broadcast dimensions is placed here.
 * @param out_shape   The broadcast shape is populated here.
 */
void broadcast_input_shapes(int noperands, const ndarray **operands,
                        int* out_ndim, dimvector* out_shape);

/**
 * Convenience function for broadcasting two operands.
 */
inline void broadcast_input_shapes(const ndarray& op0, const ndarray& op1,
                        int* out_ndim, dimvector* out_shape) {
    const ndarray *operands[2] = {&op0, &op1};
    broadcast_input_shapes(2, operands, out_ndim, out_shape);
}

/**
 * This function creates a permutation based on the strides of 'op'.
 * The value strides(out_axisperm[0]) is the smallest stride,
 * and strides(out_axisperm[ndim-1]) is the largest stride.
 */
void strides_to_axisperm(int ndim, const intptr_t *strides, int *out_axisperm);

} // namespace dnd

#endif // _DND__SHAPE_TOOLS_HPP_
