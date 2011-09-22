//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#include <dnd/ndarray.hpp>
#include <dnd/scalars.hpp>
#include <dnd/raw_iteration.hpp>

#include <stdexcept>

using namespace std;
using namespace dnd;

dnd::ndarray::ndarray(const dtype& dt, int ndim, const dimvector& shape,
        const dimvector& strides, intptr_t baseoffset,
        const std::shared_ptr<membuffer>& buffer)
    : m_dtype(dt), m_ndim(ndim), m_shape(ndim), m_strides(ndim),
      m_baseoffset(baseoffset), m_buffer(buffer)
{
    memcpy(m_shape.get(), shape.get(), ndim * sizeof(intptr_t));
    memcpy(m_strides.get(), strides.get(), ndim * sizeof(intptr_t));
}

dnd::ndarray::ndarray()
    : m_dtype(), m_ndim(0), m_shape(0), m_strides(0), m_baseoffset(0), m_buffer()
{
}

dnd::ndarray::ndarray(const dtype& dt)
    : m_dtype(dt), m_ndim(0), m_shape(0), m_strides(0),
      m_baseoffset(0), m_buffer(new membuffer(dt, 1))
{
}

dnd::ndarray::ndarray(intptr_t dim0, const dtype& dt)
    : m_dtype(dt), m_ndim(1), m_shape(1), m_strides(1),
      m_baseoffset(0), m_buffer(new membuffer(dt, dim0))
{
    m_shape[0] = dim0;
    m_strides[0] = (dim0 == 1) ? 0 : dt.itemsize();
}

dnd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, const dtype& dt)
    : m_dtype(dt), m_ndim(2), m_shape(2), m_strides(2),
      m_baseoffset(0), m_buffer(new membuffer(dt, dim0*dim1))
{
    m_shape[0] = dim0;
    m_shape[1] = dim1;
    m_strides[0] = (dim0 == 1) ? 0 : (dt.itemsize() * dim1);
    m_strides[1] = (dim1 == 1) ? 0 : dt.itemsize();
}

dnd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, intptr_t dim2, const dtype& dt)
    : m_dtype(dt), m_ndim(3), m_shape(3), m_strides(3),
      m_baseoffset(0), m_buffer(new membuffer(dt, dim0*dim1*dim2))
{
    m_shape[0] = dim0;
    m_shape[1] = dim1;
    m_shape[1] = dim2;
    m_strides[0] = (dim0 == 1) ? 0 : dt.itemsize() * dim1 * dim2;
    m_strides[1] = (dim1 == 1) ? 0 : dt.itemsize() * dim2;
    m_strides[2] = (dim2 == 1) ? 0 : dt.itemsize();
}

ndarray& dnd::ndarray::operator=(const ndarray& rhs)
{
    if (this != &rhs) {
        // Create a temporary and swap, for exception safety
        ndarray tmp(rhs.m_dtype, rhs.m_ndim, rhs.m_shape, rhs.m_strides,
                    rhs.m_baseoffset, rhs.m_buffer);
        tmp.swap(*this);
    }
    return *this;
}

void dnd::ndarray::swap(ndarray& rhs)
{
    m_dtype.swap(rhs.m_dtype);
    std::swap(m_ndim, rhs.m_ndim);
    m_shape.swap(rhs.m_shape);
    m_strides.swap(rhs.m_strides);
    std::swap(m_baseoffset, rhs.m_baseoffset);
    m_buffer.swap(rhs.m_buffer);
}

void dnd::ndarray::vassign(const ndarray& rhs, assign_error_mode errmode)
{
    //raw_ndarray_iter<2> iter(*this, rhs);
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
        } while(iter.iternext());
    }
}
