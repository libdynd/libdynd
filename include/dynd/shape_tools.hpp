//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__SHAPE_TOOLS_HPP_
#define _DYND__SHAPE_TOOLS_HPP_

#include <iostream>

#include <dynd/ndarray.hpp>

namespace dynd {

/**
 * This function returns true if the src_shape can broadcast to the dst_shape
 * It's following the same rules as numpy. The
 * destination ndim must be greator or equal, and each
 * dimension size must be broadcastable with everything
 * shoved to the right.
 */
bool shape_can_broadcast(int dst_ndim, const intptr_t *dst_shape,
                        int src_ndim, const intptr_t *src_shape);

/**
 * This function broadcasts the dimensions and strides of 'src' to a given
 * shape, raising an error if it cannot be broadcast.
 *
 * @param ndim        The number of dimensions being broadcast to.
 * @param shape       The shape being broadcast to.
 * @param src_ndim    The number of dimensions of the input which is to be broadcast.
 * @param src_shape   The shape of the input which is to be broadcast.
 * @param src_strides The strides of the input which is to be broadcast.
 * @param out_strides The resulting strides after broadcasting (with length 'ndim').
 */
void broadcast_to_shape(int ndim, const intptr_t *shape,
                int src_ndim, const intptr_t *src_shape, const intptr_t *src_strides,
                intptr_t *out_strides);

/**
 * This function broadcasts the dimensions and strides of 'src' to a given
 * shape, raising an error if it cannot be broadcast.
 */
inline void broadcast_to_shape(int ndim, const intptr_t *shape, const strided_ndarray_node *op,
                                    intptr_t *out_strides)
{
    broadcast_to_shape(ndim, shape, op->get_ndim(), op->get_shape(), op->get_strides(), out_strides);
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
void broadcast_input_shapes(int noperands, ndarray_node_ptr *operands,
                        int* out_ndim, dimvector* out_shape);

/**
 * Convenience function for broadcasting two operands.
 */
inline void broadcast_input_shapes(const ndarray_node_ptr& node0, const ndarray_node_ptr& node1,
                        int* out_ndim, dimvector* out_shape) {
    ndarray_node_ptr operands[2] = {node0, node1};
    broadcast_input_shapes(2, operands, out_ndim, out_shape);
}

/**
 * After broadcasting some input operands with broadcast_input_shapes, this function can
 * be used to copy the input strides into stride arrays where each has the same length,
 * for futher processing by strides_to_axis_perm, for instance.
 *
 * It is similar to 'broadcast_to_shape', but does not validate that the operand's shape
 * broadcasts, it merely copies the strides and pads them with zeros appropriately.
 */
void copy_input_strides(const ndarray& op, int ndim, intptr_t *out_strides);

/**
 * This function creates a permutation based on one ndarray's strides.
 * The value strides(out_axis_perm[0]) is the smallest stride,
 * and strides(out_axis_perm[ndim-1]) is the largest stride.
 */
void strides_to_axis_perm(int ndim, const intptr_t *strides, int *out_axis_perm);

/**
 * This function creates a permutation based on the array of operand strides,
 * trying to match the memory ordering of both where possible and defaulting to
 * C-order where not possible.
 */
void multistrides_to_axis_perm(int ndim, int noperands, const intptr_t **operstrides, int *out_axis_perm);

// For some reason casting 'intptr_t **' to 'const intptr_t **' causes
// a warning in g++ 4.6.1, this overload works around that.
inline void multistrides_to_axis_perm(int ndim, int noperands, intptr_t **operstrides, int *out_axis_perm) {
    multistrides_to_axis_perm(ndim, noperands,
                const_cast<const intptr_t **>(operstrides), out_axis_perm);
}

void print_shape(std::ostream& o, int ndim, const intptr_t *shape);

/**
 * Applies the indexing rules for a single linear indexing irange object to
 * a dimension of the specified size.
 *
 * @param idx  The irange indexing object.
 * @param dimension_size  The size of the dimension to which the idx is being applied.
 * @param error_i  The position in the shape where the indexing is being applied.
 * @param error_dt The dtype to which the indexing is being applied, or NULL.
 * @param out_remove_dimension  Is set to true if the dimension should be removed
 * @param out_start_index  The start index of the resolved indexing.
 * @param out_index_stride  The index stride of the resolved indexing.
 * @param out_dimension_size  The size of the resulting dimension from the resolved indexing.
 */
void apply_single_linear_index(const irange& idx, intptr_t dimension_size, int error_i, const dtype* error_dt,
        bool& out_remove_dimension, intptr_t& out_start_index, intptr_t& out_index_stride, intptr_t& out_dimension_size);

enum shape_signal_t {
    /** Shape value that has never been initialized */
    shape_signal_uninitialized = -1,
    /** Shape value that may have more than one size, depending on index */
    shape_signal_varying = -2,
};

} // namespace dynd

#endif // _DYND__SHAPE_TOOLS_HPP_
