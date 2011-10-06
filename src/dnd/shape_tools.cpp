//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <iostream>

#include <dnd/shape_tools.hpp>
#include <dnd/exceptions.hpp>

using namespace std;
using namespace dnd;

void dnd::broadcast_to_shape(int dst_ndim, const intptr_t *dst_shape,
                int src_ndim, const intptr_t *src_shape, const intptr_t *src_strides,
                intptr_t *out_strides)
{
    //cout << "broadcast_to_shape(" << dst_ndim << ", (";
    //for (int i = 0; i < dst_ndim; ++i) cout << dst_shape[i] << " ";
    //cout << "), " << src_ndim << ", (";
    //for (int i = 0; i < src_ndim; ++i) cout << src_shape[i] << " ";
    //cout << "), (";
    //for (int i = 0; i < src_ndim; ++i) cout << src_strides[i] << " ";
    //cout << ")\n";

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
        } else if (src_shape[src_i] == dst_shape[i]) {
            out_strides[i] = src_strides[src_i];
        } else {
            throw broadcast_error(dst_ndim, dst_shape, src_ndim, src_shape);
        }
    }

    //cout << "output strides: ";
    //for (int i = 0; i < dst_ndim; ++i) cout << out_strides[i] << " ";
    //cout << "\n";
}

void dnd::broadcast_input_shapes(int noperands, const ndarray **operands,
                        int* out_ndim, dimvector* out_shape)
{
    // Get the number of broadcast dimensions
    int ndim = operands[0]->ndim();
    for (int i = 0; i < noperands; ++i) {
        if (operands[i]->ndim() > ndim) {
            ndim = operands[i]->ndim();
        }
    }

    out_shape->init(ndim);
    intptr_t *shape = out_shape->get();

    // Fill in the broadcast shape
    for (int k = 0; k < ndim; ++k) {
        shape[k] = 1;
    }
    for (int i = 0; i < noperands; ++i) {
        int dimdelta = ndim - operands[i]->ndim();
        for (int k = dimdelta; k < ndim; ++k) {
            intptr_t size = operands[i]->shape(k - dimdelta);
            intptr_t itershape_size = shape[k];
            if (itershape_size == 1) {
                shape[k] = size;
            } else if (size != 1 && itershape_size != size) {
                //cout << "operand " << i << ", comparing size " << itershape_size << " vs " << size << "\n";
                throw broadcast_error(noperands, operands);
            }
        }
    }

    *out_ndim = ndim;
}

void dnd::copy_input_strides(const ndarray& op, int ndim, intptr_t *out_strides)
{
    // Process op
    int dimdelta = ndim - op.ndim();
    const intptr_t *in_strides = op.strides();
    const intptr_t *in_shape = op.shape();

    for (int i = 0; i < dimdelta; ++i) {
        out_strides[i] = 0;
    }
    for (int i = dimdelta; i < ndim; ++i) {
        out_strides[i] = in_shape[i - dimdelta] == 1 ? 0 : in_strides[i - dimdelta];
    }
}

static inline intptr_t intptr_abs(intptr_t x) {
    return x >= 0 ? x : -x;
}

void dnd::strides_to_axis_perm(int ndim, const intptr_t *strides, int *out_axis_perm)
{
    switch (ndim) {
        case 0: {
            break;
        }
        case 1: {
            out_axis_perm[0] = 0;
            break;
        }
        case 2: {
            if (intptr_abs(strides[0]) >= intptr_abs(strides[1])) {
                out_axis_perm[0] = 1;
                out_axis_perm[1] = 0;
            } else {
                out_axis_perm[0] = 0;
                out_axis_perm[1] = 1;
            }
            break;
        }
        case 3: {
            intptr_t abs_strides[3] = {intptr_abs(strides[0]),
                                    intptr_abs(strides[1]),
                                    intptr_abs(strides[2])};
            if (abs_strides[0] >= abs_strides[1]) {
                if (abs_strides[1] >= abs_strides[2]) {
                    out_axis_perm[0] = 2;
                    out_axis_perm[1] = 1;
                    out_axis_perm[2] = 0;
                } else { // abs_strides[1] < abs_strides[2]
                    if (abs_strides[0] >= abs_strides[2]) {
                        out_axis_perm[0] = 1;
                        out_axis_perm[1] = 2;
                        out_axis_perm[2] = 0;
                    } else { // abs_strides[0] < abs_strides[2]
                        out_axis_perm[0] = 1;
                        out_axis_perm[1] = 0;
                        out_axis_perm[2] = 2;
                    }
                }
            } else { // abs_strides[0] < abs_strides[1]
                if (abs_strides[1] >= abs_strides[2]) {
                    if (abs_strides[0] >= abs_strides[2]) {
                        out_axis_perm[0] = 2;
                        out_axis_perm[1] = 0;
                        out_axis_perm[2] = 1;
                    } else { // abs_strides[0] < abs_strides[2]
                        out_axis_perm[0] = 0;
                        out_axis_perm[1] = 2;
                        out_axis_perm[2] = 1;
                    }
                } else { // strides[1] < strides[2]
                    out_axis_perm[0] = 0;
                    out_axis_perm[1] = 1;
                    out_axis_perm[2] = 2;
                }
            }
            break;
        }
        default: {
            // Initialize to a reversal perm (i.e. so C-order is a no-op)
            for (int i = 0; i < ndim; ++i) {
                out_axis_perm[i] = ndim - i - 1;
            }
            // Sort based on the absolute value of the strides
            std::sort(out_axis_perm, out_axis_perm + ndim,
                        [strides](int i, int j) -> bool {
                return intptr_abs(strides[i]) < intptr_abs(strides[j]);
            });
            break;
        }
    }
}

/**
 * Compares the strides of the operands for axes 'i' and 'j', and returns whether
 * the comparison is ambiguous and, when it's not ambiguous, whether 'i' should occur
 * before 'j'.
 */
static inline void compare_strides(int i, int j, int noperands, const intptr_t **operstrides,
                                bool* out_ambiguous, bool* out_lessthan)
{
    *out_ambiguous = true;

    for (int ioperand = 0; ioperand < noperands; ++ioperand) {
        intptr_t stride_i = operstrides[ioperand][i];
        intptr_t stride_j = operstrides[ioperand][j];
        if (stride_i != 0 && stride_j != 0) {
            if (intptr_abs(stride_i) <= intptr_abs(stride_j)) {
                // Set 'lessthan' even if it's already not ambiguous, since
                // less than beats greater than when there's a conflict
                *out_lessthan = true;
                *out_ambiguous = false;
                return;
            } else if (*out_ambiguous) {
                // Only set greater than when the comparison is still ambiguous
                *out_lessthan = false;
                *out_ambiguous = false;
                // Can't return yet, because a 'lessthan' might override this choice
            }
        }
    }
}

void dnd::multistrides_to_axis_perm(int ndim, int noperands, const intptr_t **operstrides, int *out_axis_perm)
{
    switch (ndim) {
        case 0: {
            break;
        }
        case 1: {
            out_axis_perm[0] = 0;
            break;
        }
        case 2: {
            bool ambiguous = true, lessthan = false;

            // TODO: The comparison function is quite complicated, maybe there's a way to
            //       simplify all this while retaining the generality?
            compare_strides(0, 1, noperands, operstrides, &ambiguous, &lessthan);

            if (ambiguous || !lessthan) {
                out_axis_perm[0] = 1;
                out_axis_perm[1] = 0;
            } else {
                out_axis_perm[0] = 0;
                out_axis_perm[1] = 1;
            }
            break;
        }
        default: {
            // Initialize to a reversal perm (i.e. so C-order is a no-op)
            for (int i = 0; i < ndim; ++i) {
                out_axis_perm[i] = ndim - i - 1;
            }
            // Here we do a custom stable insertion sort, which avoids a swap when a comparison
            // is ambiguous
            for (int i0 = 1; i0 < ndim; ++i0) {
                // 'ipos' is the position where axis_perm[i0] will get inserted
                int ipos = i0;
                int perm_i0 = out_axis_perm[i0];

                for (int i1 = i0 - 1; i1 >= 0; --i1) {
                    bool ambiguous = true, lessthan = false;
                    int perm_i1 = out_axis_perm[i1];

                    compare_strides(perm_i1, perm_i0, noperands, operstrides, &ambiguous, &lessthan);

                    // If the comparison was unambiguous, either shift 'ipos' to 'i1', or
                    // stop looking for an insertion point
                    if (!ambiguous) {
                        if (!lessthan) {
                            ipos = i1;
                        } else {
                            break;
                        }
                    }
                }

                // Insert axis_perm[i0] into axis_perm[ipos]
                if (ipos != i0) {
                    for (int i = i0; i > ipos; --i) {
                        out_axis_perm[i] = out_axis_perm[i - 1];
                    }
                    out_axis_perm[ipos] = perm_i0;
                }
            }
            break;
        }
    }
}
