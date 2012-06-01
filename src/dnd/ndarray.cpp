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
    : m_expr_tree()
{
}

template<class T>
ndarray_expr_node *make_expr_tree_from_scalar(T value)
{
    shared_ptr<void> buffer_owner(::dnd::detail::ndarray_buffer_allocator(sizeof(T)),
                                ::dnd::detail::ndarray_buffer_deleter);
    *reinterpret_cast<T *>(buffer_owner.get()) = value;
    return new strided_array_expr_node(make_dtype<T>(), 0, NULL, NULL,
                        reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner));
}

dnd::ndarray::ndarray(signed char value)
    : m_expr_tree(make_expr_tree_from_scalar(value))
{
}
dnd::ndarray::ndarray(short value)
    : m_expr_tree(make_expr_tree_from_scalar(value))
{
}
dnd::ndarray::ndarray(int value)
    : m_expr_tree(make_expr_tree_from_scalar(value))
{
}
dnd::ndarray::ndarray(long value)
    : m_expr_tree(make_expr_tree_from_scalar(value))
{
}
dnd::ndarray::ndarray(long long value)
    : m_expr_tree(make_expr_tree_from_scalar(value))
{
}
dnd::ndarray::ndarray(unsigned char value)
    : m_expr_tree(make_expr_tree_from_scalar(value))
{
}
dnd::ndarray::ndarray(unsigned short value)
    : m_expr_tree(make_expr_tree_from_scalar(value))
{
}
dnd::ndarray::ndarray(unsigned int value)
    : m_expr_tree(make_expr_tree_from_scalar(value))
{
}
dnd::ndarray::ndarray(unsigned long value)
    : m_expr_tree(make_expr_tree_from_scalar(value))
{
}
dnd::ndarray::ndarray(unsigned long long value)
    : m_expr_tree(make_expr_tree_from_scalar(value))
{
}
dnd::ndarray::ndarray(float value)
    : m_expr_tree(make_expr_tree_from_scalar(value))
{
}
dnd::ndarray::ndarray(double value)
    : m_expr_tree(make_expr_tree_from_scalar(value))
{
}


dnd::ndarray::ndarray(const dtype& dt)
    : m_expr_tree()
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
    : m_expr_tree()
{
    intptr_t stride = (dim0 <= 1) ? 0 : dt.itemsize();
    shared_ptr<void> buffer_owner(
                    ::dnd::detail::ndarray_buffer_allocator(dt.itemsize() * dim0),
                    ::dnd::detail::ndarray_buffer_deleter);
    m_expr_tree.reset(new strided_array_expr_node(dt, 1, &dim0, &stride,
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}

dnd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, const dtype& dt)
    : m_expr_tree()
{
    intptr_t shape[2] = {dim0, dim1};
    intptr_t strides[2];
    strides[0] = (dim0 <= 1) ? 0 : dt.itemsize() * dim1;
    strides[1] = (dim1 <= 1) ? 0 : dt.itemsize();

    shared_ptr<void> buffer_owner(
                    ::dnd::detail::ndarray_buffer_allocator(dt.itemsize() * dim0 * dim1),
                    ::dnd::detail::ndarray_buffer_deleter);
    m_expr_tree.reset(new strided_array_expr_node(dt, 2, shape, strides,
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}

dnd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, intptr_t dim2, const dtype& dt)
    : m_expr_tree()
{
    intptr_t shape[3] = {dim0, dim1, dim2};
    intptr_t strides[3];
    strides[0] = (dim0 <= 1) ? 0 : dt.itemsize() * dim1 * dim2;
    strides[1] = (dim1 <= 1) ? 0 : dt.itemsize() * dim2;
    strides[2] = (dim2 <= 1) ? 0 : dt.itemsize();

    shared_ptr<void> buffer_owner(
                    ::dnd::detail::ndarray_buffer_allocator(dt.itemsize() * dim0 * dim1 * dim2),
                    ::dnd::detail::ndarray_buffer_deleter);
    m_expr_tree.reset(new strided_array_expr_node(dt, 3, shape, strides,
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}

dnd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, intptr_t dim2, intptr_t dim3, const dtype& dt)
    : m_expr_tree()
{
    intptr_t shape[4] = {dim0, dim1, dim2, dim3};
    intptr_t strides[4];
    strides[0] = (dim0 <= 1) ? 0 : dt.itemsize() * dim1 * dim2 * dim3;
    strides[1] = (dim1 <= 1) ? 0 : dt.itemsize() * dim2 * dim3;
    strides[2] = (dim2 <= 1) ? 0 : dt.itemsize() * dim3;
    strides[3] = (dim3 <= 1) ? 0 : dt.itemsize();

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

const ndarray dnd::ndarray::operator()(intptr_t idx) const
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

static void val_assign_unequal_dtypes(const ndarray& lhs, const ndarray& rhs, assign_error_mode errmode)
{
    //cout << "val_assign_unequal_dtypes\n";
    // First broadcast the 'rhs' shape to 'this'
    dimvector rhs_strides(lhs.get_ndim());
    broadcast_to_shape(lhs.get_ndim(), lhs.get_shape(), rhs, rhs_strides.get());

    // Create the raw iterator
    raw_ndarray_iter<1,1> iter(lhs.get_ndim(), lhs.get_shape(), lhs.get_originptr(), lhs.get_strides(),
                                rhs.get_originptr(), rhs_strides.get());
    //iter.debug_dump(cout);

    intptr_t innersize = iter.innersize();
    intptr_t dst_innerstride = iter.innerstride<0>(), src_innerstride = iter.innerstride<1>();

    std::pair<unary_operation_t, dnd::shared_ptr<auxiliary_data> > assign =
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

static void val_assign_equal_dtypes(const ndarray& lhs, const ndarray& rhs)
{
    //cout << "val_assign_equal_dtypes\n";
    // First broadcast the 'rhs' shape to 'this'
    dimvector rhs_strides(lhs.get_ndim());
    broadcast_to_shape(lhs.get_ndim(), lhs.get_shape(), rhs, rhs_strides.get());

    // Create the raw iterator
    raw_ndarray_iter<1,1> iter(lhs.get_ndim(), lhs.get_shape(), lhs.get_originptr(), lhs.get_strides(),
                                rhs.get_originptr(), rhs_strides.get());
    //iter.debug_dump(cout);

    intptr_t innersize = iter.innersize();
    intptr_t dst_innerstride = iter.innerstride<0>(), src_innerstride = iter.innerstride<1>();

    std::pair<unary_operation_t, dnd::shared_ptr<auxiliary_data> > assign =
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

ndarray dnd::ndarray::as_dtype(const dtype& dt, assign_error_mode errmode) const
{
    if (dt == get_dtype()) {
        return *this;
    } else {
        return ndarray(new convert_dtype_expr_node(m_expr_tree, dt, errmode));
    }
}

void dnd::ndarray::val_assign(const ndarray& rhs, assign_error_mode errmode) const
{
    if (m_expr_tree->get_node_type() != strided_array_node_type) {
        throw std::runtime_error("cannot val_assign to an expression-view ndarray, must "
                                 "first convert it to a strided array with as_strided");
    }

    if (get_dtype() == rhs.get_dtype()) {
        // The dtypes match, simpler case
        val_assign_equal_dtypes(*this, rhs);
    } else if (get_num_elements() > 5 * rhs.get_num_elements()) {
        // If the data is being duplicated more than 5 times, make a temporary copy of rhs
        // converted to the dtype of 'this', then do the broadcasting.
        ndarray tmp = empty_like(rhs, get_dtype());
        val_assign_unequal_dtypes(tmp, rhs, errmode);
        val_assign_equal_dtypes(*this, tmp);
    } else {
        // Assignment with casting
        val_assign_unequal_dtypes(*this, rhs, errmode);
    }
}

void dnd::ndarray::val_assign(const dtype& dt, const void *data, assign_error_mode errmode) const
{
    //DEBUG_COUT << "scalar val_assign\n";
    scalar_copied_if_necessary src(get_dtype(), dt, data, errmode);
    raw_ndarray_iter<1,0> iter(*this);

    intptr_t innersize = iter.innersize(), innerstride = iter.innerstride<0>();

    std::pair<unary_operation_t, dnd::shared_ptr<auxiliary_data> > assign =
                get_dtype_strided_assign_operation(
                                get_dtype(), innerstride, iter.get_align_test<0>(), 0, 0);

    if (innersize > 0) {
        do {
            //DEBUG_COUT << "scalar val_assign inner loop with size " << innersize << "\n";
            assign.first(iter.data<0>(), innerstride, src.data(), 0, innersize, assign.second.get());
        } while (iter.iternext());
    }
}

void dnd::ndarray::debug_dump(std::ostream& o = std::cerr) const
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
    if (rhs.get_expr_tree() != NULL) {
        if (rhs.get_expr_tree()->get_node_type() == strided_array_node_type) {
            o << "ndarray(" << rhs.get_dtype() << ", ";
            if (rhs.get_ndim() == 0) {
                rhs.get_dtype().print(o, rhs.get_originptr(), 0, 1, "");
            } else {
                nested_ndarray_print(o, rhs, rhs.get_originptr(), 0);
            }
            o << ")";
        } else {
            o << rhs.vals();
        }
    } else {
        o << "ndarray()";
    }

    return o;
}
