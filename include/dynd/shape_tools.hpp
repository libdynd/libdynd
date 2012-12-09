//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__SHAPE_TOOLS_HPP_
#define _DYND__SHAPE_TOOLS_HPP_

#include <iostream>

#include <dynd/dtype.hpp>
#include <dynd/shortvector.hpp>

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

inline bool shape_can_broadcast(const std::vector<intptr_t>& dst_shape,
                        const std::vector<intptr_t>& src_shape)
{
    return shape_can_broadcast((int)dst_shape.size(), dst_shape.empty() ? NULL : &dst_shape[0],
                        (int)src_shape.size(), src_shape.empty() ? NULL : &src_shape[0]);
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
void broadcast_to_shape(int ndim, const intptr_t *shape,
                int src_ndim, const intptr_t *src_shape, const intptr_t *src_strides,
                intptr_t *out_strides);

/**
 * This function creates a permutation based on one ndarray's strides.
 * The value strides(out_axis_perm[0]) is the smallest stride,
 * and strides(out_axis_perm[ndim-1]) is the largest stride.
 *
 * \param ndim  The number of values in strides and out_axis_perm.
 * \param strides  The strides values used for sorting.
 * \param out_axis_perm  A permutation which corresponds to the input strides.
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

inline void print_shape(std::ostream& o, const std::vector<intptr_t>& shape) {
    print_shape(o, (int)shape.size(), shape.empty() ? NULL : &shape[0]);
}

/**
 * Applies the indexing rules for a single linear indexing irange object to
 * a dimension of the specified size.
 *
 * \param idx  The irange indexing object.
 * \param dimension_size  The size of the dimension to which the idx is being applied.
 * \param error_i  The position in the shape where the indexing is being applied.
 * \param error_dt The dtype to which the indexing is being applied, or NULL.
 * \param out_remove_dimension  Is set to true if the dimension should be removed
 * \param out_start_index  The start index of the resolved indexing.
 * \param out_index_stride  The index stride of the resolved indexing.
 * \param out_dimension_size  The size of the resulting dimension from the resolved indexing.
 */
void apply_single_linear_index(const irange& idx, intptr_t dimension_size, int error_i, const dtype* error_dt,
        bool& out_remove_dimension, intptr_t& out_start_index, intptr_t& out_index_stride, intptr_t& out_dimension_size);

/**
 * \brief Applies indexing rules for a single integer index, returning an index in the range [0, dimension_size).
 *
 * \param i0  The integer index.
 * \param dimension_size  The size of the dimension being indexed.
 * \param error_dt  If non-NULL, a dtype used for error messages.
 *
 * \returns  An index value in the range [0, dimension_size).
 */
inline intptr_t apply_single_index(intptr_t i0, intptr_t dimension_size, const dtype* error_dt) {
    if (i0 >= 0) {
        if (i0 < dimension_size) {
            return i0;
        } else {
            if (error_dt) {
                int ndim = error_dt->extended()->get_uniform_ndim();
                dimvector shape(ndim);
                error_dt->extended()->get_shape(0, shape.get());
                throw index_out_of_bounds(i0, 0, ndim, shape.get());
            } else {
                throw index_out_of_bounds(i0, dimension_size);
            }
        }
    } else if (i0 >= -dimension_size) {
        return i0 + dimension_size;
    } else {
        if (error_dt) {
            int ndim = error_dt->extended()->get_uniform_ndim();
            dimvector shape(ndim);
            error_dt->extended()->get_shape(0, shape.get());
            throw index_out_of_bounds(i0, 0, ndim, shape.get());
        } else {
            throw index_out_of_bounds(i0, dimension_size);
        }
    }
}

/**
 * \brief Checks whether an array represents a valid permutation.
 *
 * \param size  The number of entries in the permutation
 * \param perm  The permutation array.
 *
 * \returns  True if it's a valid permutation, false otherwise.
 */
inline bool is_valid_perm(int size, const int *perm) {
    shortvector<char> flags(size);
    memset(flags.get(), 0, size);
    for (int i = 0; i < size; ++i) {
        int v = perm[i];
        if (v < 0 || v > size || flags[v]) {
            return false;
        }
        flags[v] = 1;
    }
    return true;
}

inline bool strides_are_c_contiguous(int ndim, intptr_t element_size, const intptr_t *shape, const intptr_t *strides) {
    for (int i = ndim-1; i >= 0; --i) {
        if (shape[i] != 1 && strides[i] != element_size) {
            return false;
        }
        element_size *= shape[i];
    }
    return true;
}

inline bool strides_are_f_contiguous(int ndim, intptr_t element_size, const intptr_t *shape, const intptr_t *strides) {
    for (int i = 0; i < ndim; ++i) {
        if (shape[i] != 1 && strides[i] != element_size) {
            return false;
        }
        element_size *= shape[i];
    }
    return true;
}

enum shape_signal_t {
    /** Shape value that has never been initialized */
    shape_signal_uninitialized = -2,
    /** Shape value that may have more than one size, depending on index */
    shape_signal_varying = -1,
};

} // namespace dynd

#endif // _DYND__SHAPE_TOOLS_HPP_
