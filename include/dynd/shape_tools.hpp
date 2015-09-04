//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>

#include <dynd/type.hpp>
#include <dynd/shortvector.hpp>
#include <dynd/array.hpp>

namespace dynd {

/**
 * This function returns true if the src_shape can broadcast to the dst_shape
 * It's following the same rules as numpy. The
 * destination ndim must be greator or equal, and each
 * dimension size must be broadcastable with everything
 * shoved to the right.
 */
DYND_API bool shape_can_broadcast(intptr_t dst_ndim, const intptr_t *dst_shape,
                                  intptr_t src_ndim, const intptr_t *src_shape);

inline bool shape_can_broadcast(const std::vector<intptr_t> &dst_shape,
                                const std::vector<intptr_t> &src_shape)
{
    return shape_can_broadcast(
        dst_shape.size(), dst_shape.empty() ? NULL : &dst_shape[0],
        src_shape.size(), src_shape.empty() ? NULL : &src_shape[0]);
}

/**
 * This function broadcasts the dimensions and strides of 'src' to a given
 * shape, raising an error if it cannot be broadcast.
 *
 * \param ndim        The number of dimensions being broadcast to.
 * \param shape       The shape being broadcast to.
 * \param src_ndim    The number of dimensions of the input which is to be broadcast.
 * \param src_shape   The shape of the input which is to be broadcast.
 * \param src_strides The strides of the input which is to be broadcast.
 * \param out_strides The resulting strides after broadcasting (with length 'ndim').
 */
DYND_API void broadcast_to_shape(intptr_t ndim, const intptr_t *shape,
                                 intptr_t src_ndim, const intptr_t *src_shape,
                                 const intptr_t *src_strides,
                                 intptr_t *out_strides);

/**
 * This function broadcasts the input array's shapes together,
 * producing a broadcast shape as the result. For any dimension in
 * an input with a variable-sized shape, the output shape is set
 * to a negative value.
 *
 * \param ninputs  The number of inputs whose shapes are to be broadcasted.
 * \param inputs  The inputs whose shapes are to be broadcasted.
 * \param out_undim  The number of dimensions in the output shape.
 * \param out_shape  This is filled with the broadcast shape.
 * \param out_axis_perm  A permutation of the axis for the output to use to
 *                       match the input's memory ordering.
 */
DYND_API void broadcast_input_shapes(intptr_t ninputs, const nd::array *inputs,
                                     intptr_t &out_undim, dimvector &out_shape,
                                     shortvector<int> &out_axis_perm);

/**
 * Adjusts out_shape to broadcast it with the input shape.
 *
 * \param out_undim  The number of dimensions in the output
 *                   broadcast shape. This should be set to
 *                   the maximum of all the input undim values
 *                   that will be incrementally broadcasted.
 * \param out_shape  The shape that gets updated to become the
 *                   final broadcast shape. This should be
 *                   initialized to all ones before incrementally
 *                   broadcasting.
 * \param undim  The number of dimensions in the input shape.
 * \param shape  The input shape.
 */
DYND_API void incremental_broadcast(intptr_t out_undim, intptr_t *out_shape,
                                    intptr_t undim, const intptr_t *shape);

/**
 * This function broadcasts the three operands together to create an output
 * with the broadcast result, swapping in the provided type for the uniform
 * dimension.
 *
 * \param result_inner_tp  The type that the output should have after the
 *                         broadcast uniform dims.
 * \param op0  The first operand to broadcast.
 * \param op1  The second operand to broadcast.
 * \param op2  The third operand to broadcast.
 * \param out  This is populated with the created array.
 * \param out_ndim  This is populated with the broadcast ndim.
 * \param out_shape  This is populated with the broadcast shape.
 */
DYND_API void create_broadcast_result(const ndt::type &result_inner_tp,
                                      const nd::array &op0,
                                      const nd::array &op1,
                                      const nd::array &op2, nd::array &out,
                                      intptr_t &out_ndim, dimvector &out_shape);

/**
 * This function creates a permutation based on one ndarray's strides.
 * The value strides(out_axis_perm[0]) is the smallest stride,
 * and strides(out_axis_perm[ndim-1]) is the largest stride.
 *
 * \param ndim  The number of values in strides and out_axis_perm.
 * \param strides  The strides values used for sorting.
 * \param out_axis_perm  A permutation which corresponds to the input strides.
 */
DYND_API void strides_to_axis_perm(intptr_t ndim, const intptr_t *strides,
                                   int *out_axis_perm);

/**
 * This function creates fresh strides based on the provided axis
 * permutation and element size. This function does not validate
 * that axis_perm is a valid permutation, the caller must ensure
 * this.
 *
 * \param ndim  The number of elements in axis_perm and out_strides.
 * \param axis_perm  A permutation of the axes, must contain the values
 *                   [0, ..., ndim) in some order.
 * \param shape  The shape of the array for the created strides.
 * \param element_size  The size of one array element (this is the smallest
 *                      stride in the created strides array.
 * \param out_strides  The calculated strides are placed here.
 */
DYND_API void axis_perm_to_strides(intptr_t ndim, const int *axis_perm,
                          const intptr_t *shape, intptr_t element_size,
                          intptr_t *out_strides);

/**
 * This function creates a permutation based on the array of operand strides,
 * trying to match the memory ordering of both where possible and defaulting to
 * C-order where not possible.
 */
DYND_API void multistrides_to_axis_perm(intptr_t ndim, int noperands,
                                        const intptr_t **operstrides,
                                        int *out_axis_perm);

// For some reason casting 'intptr_t **' to 'const intptr_t **' causes
// a warning in g++ 4.6.1, this overload works around that.
inline void multistrides_to_axis_perm(intptr_t ndim, int noperands,
                                      intptr_t **operstrides,
                                      int *out_axis_perm)
{
    multistrides_to_axis_perm(ndim, noperands,
                              const_cast<const intptr_t **>(operstrides),
                              out_axis_perm);
}

DYND_API void print_shape(std::ostream& o, intptr_t ndim, const intptr_t *shape);

inline void print_shape(std::ostream &o, const std::vector<intptr_t> &shape)
{
    print_shape(o, (int)shape.size(), shape.empty() ? NULL : &shape[0]);
}

/**
 * Applies the indexing rules for a single linear indexing irange object to
 * a dimension of the specified size.
 *
 * \param idx  The irange indexing object.
 * \param dimension_size  The size of the dimension to which the idx is being applied.
 * \param error_i  The position in the shape where the indexing is being applied.
 * \param error_tp The type to which the indexing is being applied, or NULL.
 * \param out_remove_dimension  Is set to true if the dimension should be removed
 * \param out_start_index  The start index of the resolved indexing.
 * \param out_index_stride  The index stride of the resolved indexing.
 * \param out_dimension_size  The size of the resulting dimension from the resolved indexing.
 */
DYND_API void apply_single_linear_index(const irange &idx,
                                        intptr_t dimension_size,
                                        intptr_t error_i,
                                        const ndt::type *error_tp,
                                        bool &out_remove_dimension,
                                        intptr_t &out_start_index,
                                        intptr_t &out_index_stride,
                                        intptr_t &out_dimension_size);

/**
 * Applies indexing rules for a single integer index, returning an index
 * in the range [0, dimension_size).
 *
 * \param i0  The integer index.
 * \param dimension_size  The size of the dimension being indexed.
 * \param error_tp  If non-NULL, a type used for error messages.
 *
 * \returns  An index value in the range [0, dimension_size).
 */
inline intptr_t apply_single_index(intptr_t i0, intptr_t dimension_size,
                                   const ndt::type *error_tp)
{
    if (i0 >= 0) {
        if (i0 < dimension_size) {
            return i0;
        } else {
            if (error_tp) {
                intptr_t ndim = error_tp->extended()->get_ndim();
                dimvector shape(ndim);
                error_tp->extended()->get_shape(ndim, 0, shape.get(), NULL,
                                                NULL);
                throw index_out_of_bounds(i0, 0, ndim, shape.get());
            } else {
                throw index_out_of_bounds(i0, dimension_size);
            }
        }
    } else if (i0 >= -dimension_size) {
        return i0 + dimension_size;
    } else {
        if (error_tp) {
            intptr_t ndim = error_tp->extended()->get_ndim();
            dimvector shape(ndim);
            error_tp->extended()->get_shape(ndim, 0, shape.get(), NULL, NULL);
            throw index_out_of_bounds(i0, 0, ndim, shape.get());
        } else {
            throw index_out_of_bounds(i0, dimension_size);
        }
    }
}

/**
 * Checks whether an array represents a valid permutation.
 *
 * \param size  The number of entries in the permutation
 * \param perm  The permutation array.
 *
 * \returns  True if it's a valid permutation, false otherwise.
 */
template<typename T0, typename T1>
inline bool is_valid_perm(T0 size, const T1 *perm)
{
  shortvector<char> flags(size);
  memset(flags.get(), 0, size);
  for (T0 i = 0; i != size; ++i) {
    T1 v = *perm++;
    if (v >= 0 && v < size && !flags[v]) {
      flags[v] = 1;
    }
    else {
      return false;
    }
  }
  return true;
}

inline bool strides_are_c_contiguous(intptr_t ndim, intptr_t element_size,
                                     const intptr_t *shape,
                                     const intptr_t *strides)
{
    // The loop counter must be a signed integer for this reverse loop to work
    for (intptr_t i = static_cast<intptr_t>(ndim) - 1; i >= 0; --i) {
        if (shape[i] != 1 && strides[i] != element_size) {
            return false;
        }
        element_size *= shape[i];
    }
    return true;
}

inline bool strides_are_f_contiguous(intptr_t ndim, intptr_t element_size,
                                     const intptr_t *shape,
                                     const intptr_t *strides)
{
    for (intptr_t i = 0; i < ndim; ++i) {
        if (shape[i] != 1 && strides[i] != element_size) {
            return false;
        }
        element_size *= shape[i];
    }
    return true;
}

/**
 * Classifies the axis order of the type, where current_stride is
 * the absolute value of the stride for the current dimension,
 * and element_tp/element_arrmeta are for the element.
 *
 * \param current_stride  The stride of the current dimension It must be nonzero.
 * \param element_tp  The type of the elements. It must have undim > 0.
 * \param element_arrmeta  The arrmeta of the elements.
 */
DYND_API axis_order_classification_t
classify_strided_axis_order(intptr_t current_stride,
                            const ndt::type &element_tp,
                            const char *element_arrmeta);

enum shape_signal_t {
    /** Shape value that has never been initialized */
    shape_signal_uninitialized = -2,
    /** Shape value that may have more than one size, depending on index */
    shape_signal_varying = -1,
};

} // namespace dynd
