//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>

#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>

using namespace std;
using namespace dynd;

bool dynd::shape_can_broadcast(int dst_ndim, const intptr_t *dst_shape,
                            int src_ndim, const intptr_t *src_shape)
{
    if (dst_ndim >= src_ndim) {
        dst_shape += (dst_ndim - src_ndim);
        for (int i = 0; i < src_ndim; ++i) {
            if (src_shape[i] != 1 && src_shape[i] != dst_shape[i]) {
                return false;
            }
        }

        return true;
    } else {
        return false;
    }
}

void dynd::broadcast_to_shape(int dst_ndim, const intptr_t *dst_shape,
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

void dynd::broadcast_input_shapes(size_t ninputs, const ndobject* inputs,
                        size_t& out_undim, dimvector& out_shape, shortvector<int>& out_axis_perm)
{
    // Get the number of broadcast dimensions
    size_t undim = inputs[0].get_undim();
    for (size_t i = 0; i < ninputs; ++i) {
        size_t candidate_undim = inputs[i].get_undim();
        if (candidate_undim > undim) {
            undim = candidate_undim;
        }
    }

    out_undim = undim;
    out_shape.init(undim);
    out_axis_perm.init(undim);
    intptr_t *shape = out_shape.get();

    // Fill in the broadcast shape
    for (size_t k = 0; k < undim; ++k) {
        shape[k] = 1;
    }
    dimvector tmpshape(undim);
    for (size_t i = 0; i < ninputs; ++i) {
        size_t input_undim = inputs[i].get_undim();
        inputs[i].get_shape(tmpshape.get());
        int dimdelta = undim - input_undim;
        for (size_t k = dimdelta; k < undim; ++k) {
            intptr_t size = tmpshape[k - dimdelta];
            intptr_t itershape_size = shape[k];
            if (itershape_size == 1) {
                shape[k] = size;
            } else if (size < 0) {
                // A negative shape value means variable-sized
                if (itershape_size > 0) {
                    shape[k] = -itershape_size;
                } else {
                    shape[k] = -1;
                }
            } else if (itershape_size >= 0) {
                if (size != 1 && itershape_size != size) {
                    //cout << "operand " << i << ", comparing size " << itershape_size << " vs " << size << "\n";
                    throw broadcast_error(ninputs, inputs);
                }
            } else { // itershape_size < 0
                if (itershape_size == -1 && size > 0) {
                    shape[k] = -size;
                } else if (size > 1 && itershape_size != -size) {
                    throw broadcast_error(ninputs, inputs);
                }
            }
        }
    }
    // Fill in the axis permutation
    if (undim > 1) {
        int *axis_perm = out_axis_perm.get();
        // TODO: keeporder behavior, currently always C order
        for (size_t i = 0; i < undim; ++i) {
            axis_perm[i] = undim - i - 1;
        }
    } else if (undim == 1) {
        out_axis_perm[0] = 0;
    }
}

void dynd::create_broadcast_result(const dtype& result_inner_dt,
                const ndobject& op0, const ndobject& op1, const ndobject& op2,
                ndobject &out, size_t& out_ndim, dimvector& out_shape)
{
    // Get the shape of the result
    shortvector<int> axis_perm;
    ndobject ops[3] = {op0, op1, op2};
    broadcast_input_shapes(3, ops, out_ndim, out_shape, axis_perm);

    out = make_strided_ndobject(result_inner_dt, out_ndim, out_shape.get(),
                    read_access_flag|write_access_flag, axis_perm.get());
}

static inline intptr_t intptr_abs(intptr_t x) {
    return x >= 0 ? x : -x;
}

namespace {

    class abs_intptr_compare {
        const intptr_t *m_strides;
    public:
        abs_intptr_compare(const intptr_t *strides)
            : m_strides(strides) {}

        bool operator()(int i, int j) {
            return intptr_abs(m_strides[i]) < intptr_abs(m_strides[j]);
        }
    };

} // anonymous namespace

void dynd::strides_to_axis_perm(int ndim, const intptr_t *strides, int *out_axis_perm)
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
            std::sort(out_axis_perm, out_axis_perm + ndim, abs_intptr_compare(strides));
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

void dynd::multistrides_to_axis_perm(int ndim, int noperands, const intptr_t **operstrides, int *out_axis_perm)
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

void dynd::print_shape(std::ostream& o, int ndim, const intptr_t *shape)
{
    o << "(";
    for (int i = 0; i < ndim; ++i) {
        intptr_t size = shape[i];
        if (size >= 0) {
            o << size;
        } else {
            o << "Var";
        }
        if (i != ndim - 1) {
            o << ", ";
        }
    }
    o << ")";
}

void dynd::apply_single_linear_index(const irange& irnge, intptr_t dimension_size, int error_i, const dtype* error_dt,
        bool& out_remove_dimension, intptr_t& out_start_index, intptr_t& out_index_stride, intptr_t& out_dimension_size)
{
    intptr_t step = irnge.step();
    if (step == 0) {
        // A single index
        out_remove_dimension = true;
        intptr_t idx = irnge.start();
        if (idx >= 0) {
            if (idx < dimension_size) {
                out_start_index = idx;
                out_index_stride = 1;
                out_dimension_size = 1;
            } else {
                if (error_dt) {
                    size_t ndim = error_dt->extended()->get_undim();
                    dimvector shape(ndim);
                    error_dt->extended()->get_shape(0, shape.get());
                    throw index_out_of_bounds(idx, error_i, ndim, shape.get());
                } else {
                    throw index_out_of_bounds(idx, dimension_size);
                }
            }
        } else if (idx >= -dimension_size) {
            out_start_index = idx + dimension_size;
            out_index_stride = 1;
            out_dimension_size = 1;
        } else {
            if (error_dt) {
                size_t ndim = error_dt->get_undim();
                dimvector shape(ndim);
                error_dt->extended()->get_shape(0, shape.get());
                throw index_out_of_bounds(idx, error_i, ndim, shape.get());
            } else {
                throw index_out_of_bounds(idx, dimension_size);
            }
        }
    } else if (step > 0) {
        // A range with a positive step
        intptr_t start = irnge.start();
        if (start >= 0) {
            if (start < dimension_size) {
                // Starts with a positive index
            } else {
                if (error_dt) {
                    size_t ndim = error_dt->get_undim();
                    dimvector shape(ndim);
                    error_dt->extended()->get_shape(0, shape.get());
                    throw irange_out_of_bounds(irnge, error_i, ndim, shape.get());
                } else {
                    throw irange_out_of_bounds(irnge, dimension_size);
                }
            }
        } else if (start >= -dimension_size) {
            // Starts with Python style negative index
            start += dimension_size;
        } else if (start == std::numeric_limits<intptr_t>::min()) {
            // Signal for "from the beginning"
            start = 0;
        } else {
            if (error_dt) {
                size_t ndim = error_dt->get_undim();
                dimvector shape(ndim);
                error_dt->extended()->get_shape(0, shape.get());
                throw irange_out_of_bounds(irnge, error_i, ndim, shape.get());
            } else {
                throw irange_out_of_bounds(irnge, dimension_size);
            }
        }

        intptr_t end = irnge.finish();
        if (end >= 0) {
            if (end <= dimension_size) {
                // Ends with a positive index, or the end of the array
            } else if (end == std::numeric_limits<intptr_t>::max()) {
                // Signal for "until the end"
                end = dimension_size;
            } else {
                if (error_dt) {
                    size_t ndim = error_dt->get_undim();
                    dimvector shape(ndim);
                    error_dt->extended()->get_shape(0, shape.get());
                    throw irange_out_of_bounds(irnge, error_i, ndim, shape.get());
                } else {
                    throw irange_out_of_bounds(irnge, dimension_size);
                }
            }
        } else if (end >= -dimension_size) {
            // Ends with a Python style negative index
            end += dimension_size;
        } else {
            if (error_dt) {
                size_t ndim = error_dt->get_undim();
                dimvector shape(ndim);
                error_dt->extended()->get_shape(0, shape.get());
                throw irange_out_of_bounds(irnge, error_i, ndim, shape.get());
            } else {
                throw irange_out_of_bounds(irnge, dimension_size);
            }
        }

        intptr_t size = end - start;
        out_remove_dimension = false;
        if (size > 0) {
            if (step == 1) {
                // Simple range
                out_start_index = start;
                out_index_stride = 1;
                out_dimension_size = size;
            } else {
                // Range with a stride
                out_start_index = start;
                out_index_stride = step;
                out_dimension_size = (size + step - 1) / step;
            }
        } else {
            // Empty slice
            out_start_index = 0;
            out_index_stride = 1;
            out_dimension_size = 0;
        }
    } else {
        // A range with a negative step
        intptr_t start = irnge.start();
        if (start >= 0) {
            if (start < dimension_size) {
                // Starts with a positive index
            } else {
                if (error_dt) {
                    size_t ndim = error_dt->get_undim();
                    dimvector shape(ndim);
                    error_dt->extended()->get_shape(0, shape.get());
                    throw irange_out_of_bounds(irnge, error_i, ndim, shape.get());
                } else {
                    throw irange_out_of_bounds(irnge, dimension_size);
                }
            }
        } else if (start >= -dimension_size) {
            // Starts with Python style negative index
            start += dimension_size;
        } else if (start == std::numeric_limits<intptr_t>::min()) {
            // Signal for "from the beginning" (which means the last element)
            start = dimension_size - 1;
        } else {
            if (error_dt) {
                size_t ndim = error_dt->get_undim();
                dimvector shape(ndim);
                error_dt->extended()->get_shape(0, shape.get());
                throw irange_out_of_bounds(irnge, error_i, ndim, shape.get());
            } else {
                throw irange_out_of_bounds(irnge, dimension_size);
            }
        }

        intptr_t end = irnge.finish();
        if (end >= 0) {
            if (end < dimension_size) {
                // Ends with a positive index, or the end of the array
            } else if (end == std::numeric_limits<intptr_t>::max()) {
                // Signal for "until the end" (which means towards index 0 of the data)
                end = -1;
            } else {
                if (error_dt) {
                    size_t ndim = error_dt->get_undim();
                    dimvector shape(ndim);
                    error_dt->extended()->get_shape(0, shape.get());
                    throw irange_out_of_bounds(irnge, error_i, ndim, shape.get());
                } else {
                    throw irange_out_of_bounds(irnge, dimension_size);
                }
            }
        } else if (end >= -dimension_size) {
            // Ends with a Python style negative index
            end += dimension_size;
        } else {
            if (error_dt) {
                size_t ndim = error_dt->get_undim();
                dimvector shape(ndim);
                error_dt->extended()->get_shape(0, shape.get());
                throw irange_out_of_bounds(irnge, error_i, ndim, shape.get());
            } else {
                throw irange_out_of_bounds(irnge, dimension_size);
            }
        }

        intptr_t size = start - end;
        out_remove_dimension = false;
        if (size > 0) {
            if (step == -1) {
                // Simple range
                out_start_index = start;
                out_index_stride = -1;
                out_dimension_size = size;
            } else {
                // Range with a stride
                out_start_index = start;
                out_index_stride = step;
                out_dimension_size = (size + (-step) - 1) / (-step);
            }
        } else {
            // Empty slice
            out_start_index = 0;
            out_index_stride = 1;
            out_dimension_size = 0;
        }
    }
}
