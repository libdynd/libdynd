//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <dnd/ndarray.hpp>
#include <dnd/scalars.hpp>
#include <dnd/raw_iteration.hpp>
#include <dnd/shape_tools.hpp>
#include <dnd/exceptions.hpp>

using namespace std;
using namespace dnd;

dnd::ndarray::ndarray(const dtype& dt, int ndim, intptr_t num_elements, const dimvector& shape,
        const dimvector& strides, char *originptr,
        const std::shared_ptr<membuffer>& buffer)
    : m_dtype(dt), m_ndim(ndim), m_num_elements(num_elements), m_shape(ndim), m_strides(ndim),
      m_originptr(originptr), m_buffer(buffer)
{
    memcpy(m_shape.get(), shape.get(), ndim * sizeof(intptr_t));
    memcpy(m_strides.get(), strides.get(), ndim * sizeof(intptr_t));
}

dnd::ndarray::ndarray()
    : m_dtype(), m_ndim(1), m_num_elements(0), m_shape(1), m_strides(1), m_originptr(NULL), m_buffer()
{
    m_shape[0] = 0;
    m_strides[0] = 0;
}

dnd::ndarray::ndarray(const dtype& dt)
    : m_dtype(dt), m_ndim(0), m_num_elements(1), m_shape(0), m_strides(0),
      m_buffer(new membuffer(dt, 1))
{
    m_originptr = m_buffer->data();
}

dnd::ndarray::ndarray(intptr_t dim0, const dtype& dt)
    : m_dtype(dt), m_ndim(1), m_num_elements(dim0), m_shape(1), m_strides(1),
      m_buffer(new membuffer(dt, dim0))
{
    m_originptr = m_buffer->data();
    m_shape[0] = dim0;
    m_strides[0] = (dim0 == 1) ? 0 : dt.itemsize();
}

dnd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, const dtype& dt)
    : m_dtype(dt), m_ndim(2), m_num_elements(dim0 * dim1), m_shape(2), m_strides(2),
      m_buffer(new membuffer(dt, dim0*dim1))
{
    m_originptr = m_buffer->data();
    m_shape[0] = dim0;
    m_shape[1] = dim1;
    m_strides[0] = (dim0 == 1) ? 0 : (dt.itemsize() * dim1);
    m_strides[1] = (dim1 == 1) ? 0 : dt.itemsize();
}

dnd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, intptr_t dim2, const dtype& dt)
    : m_dtype(dt), m_ndim(3), m_num_elements(dim0 * dim1 * dim2), m_shape(3), m_strides(3),
      m_buffer(new membuffer(dt, dim0*dim1*dim2))
{
    m_originptr = m_buffer->data();
    m_shape[0] = dim0;
    m_shape[1] = dim1;
    m_shape[2] = dim2;
    m_strides[0] = (dim0 == 1) ? 0 : dt.itemsize() * dim1 * dim2;
    m_strides[1] = (dim1 == 1) ? 0 : dt.itemsize() * dim2;
    m_strides[2] = (dim2 == 1) ? 0 : dt.itemsize();
}

ndarray dnd::ndarray::index(int nindex, const irange *indices) const
{
    // Validate the number of indices
    if (nindex > ndim()) {
        throw too_many_indices(nindex, ndim());
    }

    // Determine how many dimensions the new array will have
    int new_ndim = ndim();
    for (int i = 0; i < nindex; ++i) {
        if (indices[i].step() == 0) {
            --new_ndim;
        }
    }

    // For each irange, adjust the originptr, shape, and strides
    char *new_originptr = m_originptr;
    dimvector new_shape(new_ndim), new_strides(new_ndim);
    int new_i = 0;
    for (int i = 0; i < nindex; ++i) {
        intptr_t step = indices[i].step();
        if (step == 0) {
            // A single index
            intptr_t idx = indices[i].start();
            if (idx < 0 || idx >= m_shape[i]) {
                throw index_out_of_bounds(idx, 0, m_shape[i]);
            }
            new_originptr += idx * m_strides[i];
        } else if (step > 0) {
            // A range with a positive step
            intptr_t start = indices[i].start();
            if (start < 0 || start >= m_shape[i]) {
                if (start == INTPTR_MIN) {
                    start = 0;
                } else {
                    throw irange_out_of_bounds(indices[i], 0, m_shape[i]);
                }
            }
            new_originptr += start * m_strides[i];

            intptr_t end = indices[i].finish();
            if (end > m_shape[i]) {
                if (end == INTPTR_MAX) {
                    end = m_shape[i];
                } else {
                    throw irange_out_of_bounds(indices[i], 0, m_shape[i]);
                }
            }
            end -= start;
            if (end > 0) {
                if (step == 1) {
                    new_shape[new_i] = end;
                    new_strides[new_i] = m_strides[i];
                } else {
                    new_shape[new_i] = (end + step - 1) / step;
                    new_strides[new_i] = m_strides[i] * step;
                }
            } else {
                new_shape[new_i] = 0;
                new_strides[new_i] = 0;
            }
            ++new_i;
        } else {
            // A range with a negative step
            intptr_t start = indices[i].start();
            if (start < 0 || start >= m_shape[i]) {
                if (start == INTPTR_MIN) {
                    start = m_shape[i] - 1;
                } else {
                    throw irange_out_of_bounds(indices[i], 0, m_shape[i]);
                }
            }
            new_originptr += start * m_strides[i];

            intptr_t end = indices[i].finish();
            if (end == INTPTR_MAX) {
                end = -1;
            } else if (end < -1) {
                throw irange_out_of_bounds(indices[i], 0, m_shape[i]);
            }
            end -= start;
            if (end < 0) {
                if (step == -1) {
                    new_shape[new_i] = -end;
                    new_strides[new_i] = -m_strides[i];
                } else {
                    new_shape[new_i] = (-end - step - 1) / (-step);
                    new_strides[new_i] = m_strides[i] * step;
                }
            } else {
                new_shape[new_i] = 0;
                new_strides[new_i] = 0;
            }
            ++new_i;
        }
    }
    // Copy the info for the rest of the dimensions which remain as is
    for (int i = nindex; i < ndim(); ++i) {
        new_shape[new_i] = m_shape[i];
        new_strides[new_i] = m_strides[i];
        ++new_i;
    }

    intptr_t new_num_elements = 1;
    for (int i = 0; i < new_ndim; ++i) {
        new_num_elements *= new_shape[i];
    }

    return ndarray(get_dtype(), new_ndim, new_num_elements,
                    std::move(new_shape), std::move(new_strides), new_originptr, m_buffer);
}

ndarray dnd::ndarray::operator()(intptr_t idx) const
{
    if (1 > ndim()) {
        throw too_many_indices(1, ndim());
    }

    if (idx < 0 || idx >= m_shape[0]) {
        throw index_out_of_bounds(idx, 0, m_shape[0]);
    }

    return ndarray(get_dtype(), ndim()-1, num_elements()/m_shape[0],
                    dimvector(ndim() - 1, shape() + 1), dimvector(ndim() - 1, strides() + 1),
                    m_originptr + idx * m_strides[0], m_buffer);
}

ndarray dnd::empty_like(const ndarray& rhs, const dtype& dt)
{
    //DEBUG_COUT << "empty_like " << rhs << " --- but with dtype " << dt << "\n";
    // Sort the strides to get the memory layout ordering
    const intptr_t *shape = rhs.shape(), *strides = rhs.strides();
    shortvector<int> strideperm(rhs.ndim());
    for (int i = 0; i < rhs.ndim(); ++i) {
        strideperm[i] = i;
    }
    std::sort(strideperm.get(), strideperm.get() + rhs.ndim(),
                            [&strides](int i, int j) -> bool {
        intptr_t astride = strides[i], bstride = strides[j];
        // Take the absolute value
        if (astride < 0) astride = -astride;
        if (bstride < 0) bstride = -bstride;

        return astride < bstride;
    });

    // Build the new strides using the ordering
    dimvector res_strides(rhs.ndim());
    intptr_t stride = dt.itemsize();
    for (int i = 0; i < rhs.ndim(); ++i) {
        int p = strideperm[i];
        intptr_t size = shape[p];
        if (size == 1) {
            res_strides[i] = 0;
        } else {
            res_strides[i] = stride;
            stride *= size;
        }
    }

    // Construct the new array
    shared_ptr<membuffer> buf(new membuffer(dt, rhs.num_elements()));
    return ndarray(dt, rhs.ndim(), rhs.num_elements(), dimvector(rhs.ndim(), rhs.shape()),
                    std::move(res_strides), buf->data(), std::move(buf));
}

ndarray& dnd::ndarray::operator=(const ndarray& rhs)
{
    if (this != &rhs) {
        // Create a temporary and swap, for exception safety
        ndarray tmp(rhs.m_dtype, rhs.m_ndim, rhs.m_num_elements, rhs.m_shape, rhs.m_strides,
                    rhs.m_originptr, rhs.m_buffer);
        tmp.swap(*this);
    }
    return *this;
}

void dnd::ndarray::swap(ndarray& rhs)
{
    m_dtype.swap(rhs.m_dtype);
    std::swap(m_ndim, rhs.m_ndim);
    std::swap(m_num_elements, rhs.m_num_elements);
    m_shape.swap(rhs.m_shape);
    m_strides.swap(rhs.m_strides);
    std::swap(m_originptr, rhs.m_originptr);
    m_buffer.swap(rhs.m_buffer);
}

static void vassign_unequal_dtypes(ndarray& lhs, const ndarray& rhs, assign_error_mode errmode)
{
    //cout << "vassign_unequal_dtypes\n";
    // First broadcast the 'rhs' shape to 'this'
    dimvector rhs_strides(lhs.ndim());
    broadcast_to_shape(lhs.ndim(), lhs.shape(), rhs, rhs_strides.get());

    // Create the raw iterator
    raw_ndarray_iter<2> iter(lhs.ndim(), lhs.shape(), lhs.originptr(), lhs.strides(),
                                const_cast<char *>(rhs.originptr()), rhs_strides.get());
    //iter.debug_dump(cout);

    intptr_t innersize = iter.innersize();
    intptr_t dst_innerstride = iter.innerstride<0>(), src_innerstride = iter.innerstride<1>();

    std::pair<unary_operation_t, std::shared_ptr<auxiliary_data> > assign =
                get_dtype_strided_assign_operation(lhs.get_dtype(), dst_innerstride, iter.get_align_test<0>(),
                                            rhs.get_dtype(), src_innerstride, iter.get_align_test<1>(),
                                            errmode);

    if (innersize > 0) {
        do {
            assign.first(iter.data<0>(), dst_innerstride,
                        iter.data<1>(), src_innerstride,
                        innersize, assign.second.get());
        } while (iter.iternext());
    }
}

static void vassign_equal_dtypes(ndarray& lhs, const ndarray& rhs)
{
    //cout << "vassign_equal_dtypes\n";
    // First broadcast the 'rhs' shape to 'this'
    dimvector rhs_strides(lhs.ndim());
    broadcast_to_shape(lhs.ndim(), lhs.shape(), rhs, rhs_strides.get());

    // Create the raw iterator
    raw_ndarray_iter<2> iter(lhs.ndim(), lhs.shape(), lhs.originptr(), lhs.strides(),
                                const_cast<char *>(rhs.originptr()), rhs_strides.get());
    //iter.debug_dump(cout);

    intptr_t innersize = iter.innersize();
    intptr_t dst_innerstride = iter.innerstride<0>(), src_innerstride = iter.innerstride<1>();

    std::pair<unary_operation_t, std::shared_ptr<auxiliary_data> > assign =
                get_dtype_strided_assign_operation(lhs.get_dtype(), dst_innerstride, iter.get_align_test<0>(),
                                            src_innerstride, iter.get_align_test<1>());

    if (innersize > 0) {
        do {
            assign.first(iter.data<0>(), dst_innerstride,
                        iter.data<1>(), src_innerstride,
                        innersize, assign.second.get());
        } while (iter.iternext());
    }
}

ndarray dnd::ndarray::as_dtype(const dtype& dt, assign_error_mode errmode) const
{
    ndarray result = empty_like(*this, dt);
    vassign_unequal_dtypes(result, *this, errmode);
    return std::move(result);
}

void dnd::ndarray::vassign(const ndarray& rhs, assign_error_mode errmode)
{
    if (get_dtype() == rhs.get_dtype()) {
        // The dtypes match, simpler case
        vassign_equal_dtypes(*this, rhs);
    } else if (num_elements() > 5 * rhs.num_elements()) {
        // If the data is being duplicated more than 5 times, make a temporary copy of rhs
        // converted to the dtype of 'this'
        ndarray tmp = rhs.as_dtype(get_dtype(), errmode);
        vassign_equal_dtypes(*this, tmp);
    } else {
        // Assignment with casting
        vassign_unequal_dtypes(*this, rhs, errmode);
    }
}

void dnd::ndarray::vassign(const dtype& dt, const void *data, assign_error_mode errmode)
{
    //DEBUG_COUT << "scalar vassign\n";
    scalar_copied_if_necessary src(m_dtype, dt, data, errmode);
    raw_ndarray_iter<1> iter(*this);
    
    intptr_t innersize = iter.innersize(), innerstride = iter.innerstride<0>();

    std::pair<unary_operation_t, std::shared_ptr<auxiliary_data> > assign =
                get_dtype_strided_assign_operation(m_dtype, innerstride, iter.get_align_test<0>(), 0, 0);

    if (innersize > 0) {
        do {
            //DEBUG_COUT << "scalar vassign inner loop with size " << innersize << "\n";
            assign.first(iter.data<0>(), innerstride, src.data(), 0, innersize, assign.second.get());
        } while (iter.iternext());
    }
}

static void nested_ndarray_print(std::ostream& o, const ndarray& rhs, const char *data, int i)
{
    o << "{";
    if (i + 1 == rhs.ndim()) {
        rhs.get_dtype().print(o, data, rhs.strides(i), rhs.shape(i), ", ");
    } else {
        intptr_t size = rhs.shape(i);
        intptr_t stride = rhs.strides(i);
        for (intptr_t k = 0; k < size; ++k) {
            nested_ndarray_print(o, rhs, data, i+1);
            if (k + 1 != size) {
                o << ", ";
            }
            data += stride;
        }
    }
    o << "}";
}

std::ostream& dnd::operator<<(std::ostream& o, const ndarray& rhs)
{
    o << "ndarray(" << rhs.get_dtype() << ", ";
    if (rhs.ndim() == 0) {
        rhs.get_dtype().print(o, rhs.originptr(), 0, 1, "");
    } else {
        nested_ndarray_print(o, rhs, rhs.originptr(), 0);
    }
    o << ")";

    return o;
}
