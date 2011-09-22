//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <stdexcept>

#include <dnd/ndarray.hpp>
#include <dnd/scalars.hpp>
#include <dnd/raw_iteration.hpp>
#include <dnd/shape_tools.hpp>

using namespace std;
using namespace dnd;

dnd::ndarray::ndarray(const dtype& dt, int ndim, intptr_t size, const dimvector& shape,
        const dimvector& strides, intptr_t baseoffset,
        const std::shared_ptr<membuffer>& buffer)
    : m_dtype(dt), m_ndim(ndim), m_size(size), m_shape(ndim), m_strides(ndim),
      m_baseoffset(baseoffset), m_buffer(buffer)
{
    memcpy(m_shape.get(), shape.get(), ndim * sizeof(intptr_t));
    memcpy(m_strides.get(), strides.get(), ndim * sizeof(intptr_t));
}

dnd::ndarray::ndarray()
    : m_dtype(), m_ndim(1), m_size(0), m_shape(1), m_strides(1), m_baseoffset(0), m_buffer()
{
    m_shape[0] = 0;
    m_strides[0] = 0;
}

dnd::ndarray::ndarray(const dtype& dt)
    : m_dtype(dt), m_ndim(0), m_size(1), m_shape(0), m_strides(0),
      m_baseoffset(0), m_buffer(new membuffer(dt, 1))
{
}

dnd::ndarray::ndarray(intptr_t dim0, const dtype& dt)
    : m_dtype(dt), m_ndim(1), m_size(dim0), m_shape(1), m_strides(1),
      m_baseoffset(0), m_buffer(new membuffer(dt, dim0))
{
    m_shape[0] = dim0;
    m_strides[0] = (dim0 == 1) ? 0 : dt.itemsize();
}

dnd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, const dtype& dt)
    : m_dtype(dt), m_ndim(2), m_size(dim0 * dim1), m_shape(2), m_strides(2),
      m_baseoffset(0), m_buffer(new membuffer(dt, dim0*dim1))
{
    m_shape[0] = dim0;
    m_shape[1] = dim1;
    m_strides[0] = (dim0 == 1) ? 0 : (dt.itemsize() * dim1);
    m_strides[1] = (dim1 == 1) ? 0 : dt.itemsize();
}

dnd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, intptr_t dim2, const dtype& dt)
    : m_dtype(dt), m_ndim(3), m_size(dim0 * dim1 * dim2), m_shape(3), m_strides(3),
      m_baseoffset(0), m_buffer(new membuffer(dt, dim0*dim1*dim2))
{
    m_shape[0] = dim0;
    m_shape[1] = dim1;
    m_shape[2] = dim2;
    m_strides[0] = (dim0 == 1) ? 0 : dt.itemsize() * dim1 * dim2;
    m_strides[1] = (dim1 == 1) ? 0 : dt.itemsize() * dim2;
    m_strides[2] = (dim2 == 1) ? 0 : dt.itemsize();
}

ndarray dnd::empty_like(const ndarray& rhs, const dtype& dt)
{
    // Sort the strides to get the memory layout ordering
    const intptr_t *shape = rhs.shape(), *strides = rhs.strides();
    shortvector<int, 3> strideperm(rhs.ndim());
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
    return ndarray(dt, rhs.ndim(), rhs.size(), dimvector(rhs.ndim(), rhs.shape()),
                    std::move(res_strides), 0,
                    shared_ptr<membuffer>(new membuffer(dt, rhs.size())));
}

ndarray& dnd::ndarray::operator=(const ndarray& rhs)
{
    if (this != &rhs) {
        // Create a temporary and swap, for exception safety
        ndarray tmp(rhs.m_dtype, rhs.m_ndim, rhs.m_size, rhs.m_shape, rhs.m_strides,
                    rhs.m_baseoffset, rhs.m_buffer);
        tmp.swap(*this);
    }
    return *this;
}

void dnd::ndarray::swap(ndarray& rhs)
{
    m_dtype.swap(rhs.m_dtype);
    std::swap(m_ndim, rhs.m_ndim);
    std::swap(m_size, rhs.m_size);
    m_shape.swap(rhs.m_shape);
    m_strides.swap(rhs.m_strides);
    std::swap(m_baseoffset, rhs.m_baseoffset);
    m_buffer.swap(rhs.m_buffer);
}

static void vassign_unequal_dtypes(ndarray& lhs, const ndarray& rhs, assign_error_mode errmode)
{
    // First broadcast the 'rhs' shape to 'this'
    dimvector rhs_strides(lhs.ndim());
    broadcast_to_shape(lhs.ndim(), lhs.shape(), rhs.ndim(), rhs.shape(), rhs.strides(), rhs_strides.get());

    // Create the raw iterator
    raw_ndarray_iter<2> iter(lhs.ndim(), lhs.shape(), lhs.data(), lhs.strides(),
                                const_cast<char *>(rhs.data()), rhs_strides.get());

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
    // First broadcast the 'rhs' shape to 'this'
    dimvector rhs_strides(lhs.ndim());
    broadcast_to_shape(lhs.ndim(), lhs.shape(), rhs.ndim(), rhs.shape(), rhs.strides(), rhs_strides.get());

    // Create the raw iterator
    raw_ndarray_iter<2> iter(lhs.ndim(), lhs.shape(), lhs.data(), lhs.strides(),
                                const_cast<char *>(rhs.data()), rhs_strides.get());

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
    } else if (size() > 5 * rhs.size()) {
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
        rhs.get_dtype().print(o, rhs.data(), 0, 1, "");
    } else {
        nested_ndarray_print(o, rhs, rhs.data(), 0);
    }
    o << ")";

    return o;
}
