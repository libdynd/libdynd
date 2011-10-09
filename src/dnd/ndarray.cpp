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

#include "ndarray_expr_node_instances.hpp"

using namespace std;
using namespace dnd;

// Default buffer allocator for ndarray
void *dnd::detail::ndarray_buffer_allocator(intptr_t size)
{
    return new char[size];
}

// Default buffer deleter for ndarray
void dnd::detail::ndarray_buffer_deleter(void *ptr)
{
    delete[] reinterpret_cast<char *>(ptr);
}

dnd::ndarray::ndarray()
{
}

dnd::ndarray::ndarray(int8_t value)
{
    shared_ptr<void> buffer_owner(::dnd::detail::ndarray_buffer_allocator(1),
                                ::dnd::detail::ndarray_buffer_deleter);
    *reinterpret_cast<int8_t *>(buffer_owner.get()) = value;
    m_expr_tree.reset(new strided_array_expr_node(dtype(int8_type_id), 0, NULL, NULL, 
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}
dnd::ndarray::ndarray(int16_t value)
{
    shared_ptr<void> buffer_owner(::dnd::detail::ndarray_buffer_allocator(2),
                                ::dnd::detail::ndarray_buffer_deleter);
    *reinterpret_cast<int16_t *>(buffer_owner.get()) = value;
    m_expr_tree.reset(new strided_array_expr_node(dtype(int16_type_id), 0, NULL, NULL, 
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}
dnd::ndarray::ndarray(int32_t value)
{
    shared_ptr<void> buffer_owner(::dnd::detail::ndarray_buffer_allocator(4),
                                ::dnd::detail::ndarray_buffer_deleter);
    *reinterpret_cast<int32_t *>(buffer_owner.get()) = value;
    m_expr_tree.reset(new strided_array_expr_node(dtype(int32_type_id), 0, NULL, NULL, 
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}
dnd::ndarray::ndarray(int64_t value)
{
    shared_ptr<void> buffer_owner(::dnd::detail::ndarray_buffer_allocator(8),
                                ::dnd::detail::ndarray_buffer_deleter);
    *reinterpret_cast<int64_t *>(buffer_owner.get()) = value;
    m_expr_tree.reset(new strided_array_expr_node(dtype(int64_type_id), 0, NULL, NULL, 
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}
dnd::ndarray::ndarray(uint8_t value)
{
    shared_ptr<void> buffer_owner(::dnd::detail::ndarray_buffer_allocator(1),
                                ::dnd::detail::ndarray_buffer_deleter);
    *reinterpret_cast<uint8_t *>(buffer_owner.get()) = value;
    m_expr_tree.reset(new strided_array_expr_node(dtype(uint8_type_id), 0, NULL, NULL, 
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}
dnd::ndarray::ndarray(uint16_t value)
{
    shared_ptr<void> buffer_owner(::dnd::detail::ndarray_buffer_allocator(2),
                                ::dnd::detail::ndarray_buffer_deleter);
    *reinterpret_cast<uint16_t *>(buffer_owner.get()) = value;
    m_expr_tree.reset(new strided_array_expr_node(dtype(uint16_type_id), 0, NULL, NULL, 
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}
dnd::ndarray::ndarray(uint32_t value)
{
    shared_ptr<void> buffer_owner(::dnd::detail::ndarray_buffer_allocator(4),
                                ::dnd::detail::ndarray_buffer_deleter);
    *reinterpret_cast<uint32_t *>(buffer_owner.get()) = value;
    m_expr_tree.reset(new strided_array_expr_node(dtype(uint32_type_id), 0, NULL, NULL, 
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}
dnd::ndarray::ndarray(uint64_t value)
{
    shared_ptr<void> buffer_owner(::dnd::detail::ndarray_buffer_allocator(8),
                                ::dnd::detail::ndarray_buffer_deleter);
    *reinterpret_cast<uint64_t *>(buffer_owner.get()) = value;
    m_expr_tree.reset(new strided_array_expr_node(dtype(uint64_type_id), 0, NULL, NULL, 
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}
dnd::ndarray::ndarray(float value)
{
    shared_ptr<void> buffer_owner(::dnd::detail::ndarray_buffer_allocator(4),
                                ::dnd::detail::ndarray_buffer_deleter);
    *reinterpret_cast<float *>(buffer_owner.get()) = value;
    m_expr_tree.reset(new strided_array_expr_node(dtype(float32_type_id), 0, NULL, NULL, 
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}
dnd::ndarray::ndarray(double value)
{
    shared_ptr<void> buffer_owner(::dnd::detail::ndarray_buffer_allocator(8),
                                ::dnd::detail::ndarray_buffer_deleter);
    *reinterpret_cast<double *>(buffer_owner.get()) = value;
    m_expr_tree.reset(new strided_array_expr_node(dtype(float64_type_id), 0, NULL, NULL, 
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}


dnd::ndarray::ndarray(const dtype& dt)
{
    shared_ptr<void> buffer_owner(::dnd::detail::ndarray_buffer_allocator(dt.itemsize()),
                                ::dnd::detail::ndarray_buffer_deleter);
    m_expr_tree.reset(new strided_array_expr_node(dt, 0, NULL, NULL, 
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}

dnd::ndarray::ndarray(const ndarray_expr_node_ptr& expr_tree)
    : m_expr_tree(expr_tree)
{
}

dnd::ndarray::ndarray(ndarray_expr_node_ptr&& expr_tree)
    : m_expr_tree(std::move(expr_tree))
{
}

dnd::ndarray::ndarray(intptr_t dim0, const dtype& dt)
{
    intptr_t stride = (dim0 <= 1) ? 0 : dt.itemsize();
    shared_ptr<void> buffer_owner(
                    ::dnd::detail::ndarray_buffer_allocator(dt.itemsize() * dim0),
                    ::dnd::detail::ndarray_buffer_deleter);
    m_expr_tree.reset(new strided_array_expr_node(dt, 1, &dim0, &stride, 
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}

dnd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, const dtype& dt)
{
    intptr_t shape[2] = {dim0, dim1};
    intptr_t strides[2] = {(dim0 <= 1) ? 0 : dt.itemsize() * dim1,
                           (dim1 <= 1) ? 0 : dt.itemsize()};
    shared_ptr<void> buffer_owner(
                    ::dnd::detail::ndarray_buffer_allocator(dt.itemsize() * dim0 * dim1),
                    ::dnd::detail::ndarray_buffer_deleter);
    m_expr_tree.reset(new strided_array_expr_node(dt, 2, shape, strides, 
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}

dnd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, intptr_t dim2, const dtype& dt)
{
    intptr_t shape[3] = {dim0, dim1, dim2};
    intptr_t strides[3] = {(dim0 <= 1) ? 0 : dt.itemsize() * dim1 * dim2,
                           (dim1 <= 1) ? 0 : dt.itemsize() * dim2,
                           (dim2 <= 1) ? 0 : dt.itemsize()};
    shared_ptr<void> buffer_owner(
                    ::dnd::detail::ndarray_buffer_allocator(dt.itemsize() * dim0 * dim1 * dim2),
                    ::dnd::detail::ndarray_buffer_deleter);
    m_expr_tree.reset(new strided_array_expr_node(dt, 3, shape, strides, 
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}

dnd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, intptr_t dim2, intptr_t dim3, const dtype& dt)
{
    intptr_t shape[4] = {dim0, dim1, dim2, dim3};
    intptr_t strides[4] = {(dim0 <= 1) ? 0 : dt.itemsize() * dim1 * dim2 * dim3,
                           (dim1 <= 1) ? 0 : dt.itemsize() * dim2 * dim3,
                           (dim2 <= 1) ? 0 : dt.itemsize() * dim3,
                           (dim3 <= 1) ? 0 : dt.itemsize()};
    shared_ptr<void> buffer_owner(
                    ::dnd::detail::ndarray_buffer_allocator(dt.itemsize() * dim0 * dim1 * dim2 * dim3),
                    ::dnd::detail::ndarray_buffer_deleter);
    m_expr_tree.reset(new strided_array_expr_node(dt, 4, shape, strides, 
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}

ndarray dnd::ndarray::index(int nindex, const irange *indices) const
{
    // Casting away const is ok here, because we pass 'false' to 'allow_in_place'
    return ndarray(make_linear_index_expr_node(
                        const_cast<ndarray_expr_node *>(m_expr_tree.get()),
                        nindex, indices, false));
}

ndarray dnd::ndarray::operator()(intptr_t idx) const
{
    // Casting away const is ok here, because we pass 'false' to 'allow_in_place'
    return ndarray(make_integer_index_expr_node(get_expr_tree(), 0, idx, false));
}

ndarray& dnd::ndarray::operator=(const ndarray& rhs)
{
    m_expr_tree = rhs.m_expr_tree;
    return *this;
}

void dnd::ndarray::swap(ndarray& rhs)
{
    m_expr_tree.swap(rhs.m_expr_tree);
}

ndarray dnd::empty_like(const ndarray& rhs, const dtype& dt)
{
    // Sort the strides to get the memory layout ordering
    shortvector<int> axis_perm(rhs.get_ndim());
    strides_to_axis_perm(rhs.get_ndim(), rhs.get_strides(), axis_perm.get());

    // Construct the new array
    return ndarray(dt, rhs.get_ndim(), rhs.get_shape(), axis_perm.get());
}

static void vassign_unequal_dtypes(ndarray& lhs, const ndarray& rhs, assign_error_mode errmode)
{
    //cout << "vassign_unequal_dtypes\n";
    // First broadcast the 'rhs' shape to 'this'
    dimvector rhs_strides(lhs.get_ndim());
    broadcast_to_shape(lhs.get_ndim(), lhs.get_shape(), rhs, rhs_strides.get());

    // Create the raw iterator
    raw_ndarray_iter<1,1> iter(lhs.get_ndim(), lhs.get_shape(), lhs.get_originptr(), lhs.get_strides(),
                                rhs.get_originptr(), rhs_strides.get());
    //iter.debug_dump(cout);

    intptr_t innersize = iter.innersize();
    intptr_t dst_innerstride = iter.innerstride<0>(), src_innerstride = iter.innerstride<1>();

    std::pair<unary_operation_t, std::shared_ptr<auxiliary_data> > assign =
                get_dtype_strided_assign_operation(
                                            lhs.get_dtype(), dst_innerstride, iter.get_align_test<0>(),
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
    dimvector rhs_strides(lhs.get_ndim());
    broadcast_to_shape(lhs.get_ndim(), lhs.get_shape(), rhs, rhs_strides.get());

    // Create the raw iterator
    raw_ndarray_iter<1,1> iter(lhs.get_ndim(), lhs.get_shape(), lhs.get_originptr(), lhs.get_strides(),
                                rhs.get_originptr(), rhs_strides.get());
    //iter.debug_dump(cout);

    intptr_t innersize = iter.innersize();
    intptr_t dst_innerstride = iter.innerstride<0>(), src_innerstride = iter.innerstride<1>();

    std::pair<unary_operation_t, std::shared_ptr<auxiliary_data> > assign =
                get_dtype_strided_assign_operation(
                                            lhs.get_dtype(), dst_innerstride, iter.get_align_test<0>(),
                                            src_innerstride, iter.get_align_test<1>());

    if (innersize > 0) {
        do {
            assign.first(iter.data<0>(), dst_innerstride,
                        iter.data<1>(), src_innerstride,
                        innersize, assign.second.get());
        } while (iter.iternext());
    }
}

ndarray dnd::ndarray::as_strided() const
{
    if (m_expr_tree->get_node_type() == strided_array_node_type) {
        return *this;
    } else {
        return ndarray(m_expr_tree->evaluate());
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
    if (m_expr_tree->get_node_type() != strided_array_node_type) {
        throw std::runtime_error("cannot vassign to an expression-view ndarray, must "
                                 "first convert it to a strided array with as_strided");
    }

    if (get_dtype() == rhs.get_dtype()) {
        // The dtypes match, simpler case
        vassign_equal_dtypes(*this, rhs);
    } else if (get_num_elements() > 5 * rhs.get_num_elements()) {
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
    scalar_copied_if_necessary src(get_dtype(), dt, data, errmode);
    raw_ndarray_iter<1,0> iter(*this);
    
    intptr_t innersize = iter.innersize(), innerstride = iter.innerstride<0>();

    std::pair<unary_operation_t, std::shared_ptr<auxiliary_data> > assign =
                get_dtype_strided_assign_operation(get_dtype(), innerstride, iter.get_align_test<0>(), 0, 0);

    if (innersize > 0) {
        do {
            //DEBUG_COUT << "scalar vassign inner loop with size " << innersize << "\n";
            assign.first(iter.data<0>(), innerstride, src.data(), 0, innersize, assign.second.get());
        } while (iter.iternext());
    }
}

void dnd::ndarray::debug_dump(std::ostream& o) const
{
    o << "------ ndarray\n";
    m_expr_tree->debug_dump(o, " ");
    o << "------" << endl;
}

static void nested_ndarray_print(std::ostream& o, const ndarray& rhs, const char *data, int i)
{
    o << "{";
    if (i + 1 == rhs.get_ndim()) {
        rhs.get_dtype().print(o, data, rhs.get_strides(i), rhs.get_shape(i), ", ");
    } else {
        intptr_t size = rhs.get_shape(i);
        intptr_t stride = rhs.get_strides(i);
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
    if (rhs.get_expr_tree()->get_node_type() == strided_array_node_type) {
        o << "ndarray(" << rhs.get_dtype() << ", ";
        if (rhs.get_ndim() == 0) {
            rhs.get_dtype().print(o, rhs.get_originptr(), 0, 1, "");
        } else {
            nested_ndarray_print(o, rhs, rhs.get_originptr(), 0);
        }
        o << ")";
    } else {
        o << rhs.as_strided();
    }

    return o;
}
