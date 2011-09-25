//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <dnd/shape_tools.hpp>
#include <dnd/exceptions.hpp>

void dnd::broadcast_to_shape(int dst_ndim, const intptr_t *dst_shape,
                int src_ndim, const intptr_t *src_shape, const intptr_t *src_strides,
                intptr_t *out_strides)
{
    if (src_ndim > dst_ndim) {
        throw broadcast_error(dst_ndim, dst_shape, src_ndim, src_shape);
    }

    int dimdelta = dst_ndim - src_ndim;
    for (int i = 0; i < dimdelta; ++i) {
        out_strides[i] = 0;
    }
    for (int i = dimdelta; i < dst_ndim; ++i) {
        int src_i = i - dimdelta;
        if (src_shape[src_i] == 1) {
            out_strides[i] = 0;
        } else if (src_shape[src_i] != dst_shape[i]) {
            throw broadcast_error(dst_ndim, dst_shape, src_ndim, src_shape);
        } else {
            out_strides[i] = src_strides[src_i];
        }
    }
}

void dnd::broadcast_input_shapes(int noperands, const ndarray **operands,
                        int& out_ndim, dimvector& out_shape)
{
    // Get the number of broadcast dimensions
    int ndim = operands[0]->ndim();
    for (int i = 0; i < noperands; ++i) {
        if (operands[i]->ndim() > ndim) {
            ndim = operands[i]->ndim();
        }
    }

    out_shape.init(ndim);

    // Fill in the broadcast shape
    for (int k = 0; k < ndim; ++k) {
        out_shape[k] = 1;
    }
    for (int i = 0; i < noperands; ++i) {
        int dimdelta = ndim - operands[i]->ndim();
        for (int k = dimdelta; k < ndim; ++k) {
            intptr_t size = operands[i]->shape(k - dimdelta);
            intptr_t itershape_size = out_shape[k];
            if (itershape_size == 1) {
                out_shape[k] = size;
            } else if (itershape_size != size) {
                throw broadcast_error(noperands, operands);
            }
        }
    }

    out_ndim = ndim;
}

static inline intptr_t intptr_abs(intptr_t x) {
    return x >= 0 ? x : -x;
}

void dnd::make_sorted_stride_perm(int ndim, const intptr_t *strides, int *out_strideperm)
{
    switch (ndim) {
        case 0: {
            break;
        }
        case 1: {
            out_strideperm[0] = 0;
            break;
        }
        case 2: {
            if (intptr_abs(strides[0]) >= intptr_abs(strides[1])) {
                out_strideperm[0] = 1;
                out_strideperm[1] = 0;
            } else {
                out_strideperm[0] = 0;
                out_strideperm[1] = 1;
            }
            break;
        }
        case 3: {
            intptr_t abs_strides[3] = {intptr_abs(strides[0]),
                                    intptr_abs(strides[1]),
                                    intptr_abs(strides[2])};
            if (abs_strides[0] >= abs_strides[1]) {
                if (abs_strides[1] >= abs_strides[2]) {
                    out_strideperm[0] = 2;
                    out_strideperm[1] = 1;
                    out_strideperm[2] = 0;
                } else { // abs_strides[1] < abs_strides[2]
                    if (abs_strides[0] >= abs_strides[2]) {
                        out_strideperm[0] = 1;
                        out_strideperm[1] = 2;
                        out_strideperm[2] = 0;
                    } else { // abs_strides[0] < abs_strides[2]
                        out_strideperm[0] = 1;
                        out_strideperm[1] = 0;
                        out_strideperm[2] = 2;
                    }
                }
            } else { // abs_strides[0] < abs_strides[1]
                if (abs_strides[1] >= abs_strides[2]) {
                    if (abs_strides[0] >= abs_strides[2]) {
                        out_strideperm[0] = 2;
                        out_strideperm[1] = 0;
                        out_strideperm[2] = 1;
                    } else { // abs_strides[0] < abs_strides[2]
                        out_strideperm[0] = 0;
                        out_strideperm[1] = 2;
                        out_strideperm[2] = 1;
                    }
                } else { // strides[1] < strides[2]
                    out_strideperm[0] = 0;
                    out_strideperm[1] = 1;
                    out_strideperm[2] = 2;
                }
            }
            break;
        }
        default: {
            // Initialize to a reversal perm (i.e. so C-order is a no-op)
            for (int i = 0; i < ndim; ++i) {
                out_strideperm[i] = ndim - i - 1;
            }
            // Sort based on the absolute value of the strides
            std::sort(out_strideperm, out_strideperm + ndim,
                        [&strides](int i, int j) -> bool {
                return intptr_abs(strides[i]) < intptr_abs(strides[j]);
            });
            break;
        }
    }
}
