//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/ndarray.hpp>
#include <dnd/scalars.hpp>
#include <dnd/raw_iteration.hpp>
#include <dnd/shape_tools.hpp>
#include <dnd/exceptions.hpp>
#include <dnd/buffer_storage.hpp>
#include <dnd/kernels/assignment_kernels.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>
#include <dnd/dtypes/dtype_alignment.hpp>
#include <dnd/dtypes/view_dtype.hpp>
#include <dnd/dtypes/fixedstring_dtype.hpp>

#include "nodes/ndarray_expr_node_instances.hpp"
#include <dnd/nodes/immutable_scalar_node.hpp>

using namespace std;
using namespace dnd;

dnd::ndarray::ndarray()
    : m_expr_tree()
{
}

template<class T>
typename enable_if<is_dtype_scalar<T>::value, ndarray_expr_node *>::type make_immutable_scalar_node_raw(const T& value)
{
    return new immutable_scalar_node(make_dtype<T>(), reinterpret_cast<const char *>(&value));
}

dnd::ndarray::ndarray(signed char value)
    : m_expr_tree(make_immutable_scalar_node_raw(value))
{
}
dnd::ndarray::ndarray(short value)
    : m_expr_tree(make_immutable_scalar_node_raw(value))
{
}
dnd::ndarray::ndarray(int value)
    : m_expr_tree(make_immutable_scalar_node_raw(value))
{
}
dnd::ndarray::ndarray(long value)
    : m_expr_tree(make_immutable_scalar_node_raw(value))
{
}
dnd::ndarray::ndarray(long long value)
    : m_expr_tree(make_immutable_scalar_node_raw(value))
{
}
dnd::ndarray::ndarray(unsigned char value)
    : m_expr_tree(make_immutable_scalar_node_raw(value))
{
}
dnd::ndarray::ndarray(unsigned short value)
    : m_expr_tree(make_immutable_scalar_node_raw(value))
{
}
dnd::ndarray::ndarray(unsigned int value)
    : m_expr_tree(make_immutable_scalar_node_raw(value))
{
}
dnd::ndarray::ndarray(unsigned long value)
    : m_expr_tree(make_immutable_scalar_node_raw(value))
{
}
dnd::ndarray::ndarray(unsigned long long value)
    : m_expr_tree(make_immutable_scalar_node_raw(value))
{
}
dnd::ndarray::ndarray(float value)
    : m_expr_tree(make_immutable_scalar_node_raw(value))
{
}
dnd::ndarray::ndarray(double value)
    : m_expr_tree(make_immutable_scalar_node_raw(value))
{
}
dnd::ndarray::ndarray(complex<float> value)
    : m_expr_tree(make_immutable_scalar_node_raw(value))
{
}
dnd::ndarray::ndarray(complex<double> value)
    : m_expr_tree(make_immutable_scalar_node_raw(value))
{
}

// Makes a UTF8 fixedstring from a std::string. Maybe we will want
// another choice later?
dnd::ndarray::ndarray(const std::string& value)
    : m_expr_tree(new immutable_scalar_node(
                make_fixedstring_dtype(string_encoding_utf_8, value.size()), value.c_str()))
{
}

dnd::ndarray::ndarray(const dtype& dt)
    : m_expr_tree()
{
    char *originptr = 0;
    memory_block_ref memblock = make_fixed_size_pod_memory_block(dt.alignment(), dt.element_size(), &originptr);
    m_expr_tree.reset(new strided_array_expr_node(dt, 0, NULL, NULL,
                            originptr, DND_MOVE(memblock)));
}

dnd::ndarray::ndarray(const dtype& dt, const char *raw_data)
    : m_expr_tree(new immutable_scalar_node(dt, raw_data))
{
}


dnd::ndarray::ndarray(const ndarray_expr_node_ptr& expr_tree)
    : m_expr_tree(expr_tree)
{
}

dnd::ndarray::ndarray(ndarray_expr_node_ptr&& expr_tree)
    : m_expr_tree(DND_MOVE(expr_tree))
{
}

dnd::ndarray::ndarray(intptr_t dim0, const dtype& dt)
    : m_expr_tree()
{
    intptr_t stride = (dim0 <= 1) ? 0 : dt.element_size();
    char *originptr = 0;
    memory_block_ref memblock = make_fixed_size_pod_memory_block(dt.alignment(), dt.element_size() * dim0, &originptr);
    m_expr_tree.reset(new strided_array_expr_node(dt, 1, &dim0, &stride,
                            originptr, DND_MOVE(memblock)));
}

dnd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, const dtype& dt)
    : m_expr_tree()
{
    intptr_t shape[2] = {dim0, dim1};
    intptr_t strides[2];
    strides[0] = (dim0 <= 1) ? 0 : dt.element_size() * dim1;
    strides[1] = (dim1 <= 1) ? 0 : dt.element_size();

    char *originptr = 0;
    memory_block_ref memblock = make_fixed_size_pod_memory_block(dt.alignment(), dt.element_size() * dim0 * dim1, &originptr);
    m_expr_tree.reset(new strided_array_expr_node(dt, 2, shape, strides,
                            originptr, DND_MOVE(memblock)));
}

dnd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, intptr_t dim2, const dtype& dt)
    : m_expr_tree()
{
    intptr_t shape[3] = {dim0, dim1, dim2};
    intptr_t strides[3];
    strides[0] = (dim0 <= 1) ? 0 : dt.element_size() * dim1 * dim2;
    strides[1] = (dim1 <= 1) ? 0 : dt.element_size() * dim2;
    strides[2] = (dim2 <= 1) ? 0 : dt.element_size();

    char *originptr = 0;
    memory_block_ref memblock = make_fixed_size_pod_memory_block(dt.alignment(), dt.element_size() * dim0 * dim1 * dim2, &originptr);
    m_expr_tree.reset(new strided_array_expr_node(dt, 3, shape, strides,
                            originptr, DND_MOVE(memblock)));
}

dnd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, intptr_t dim2, intptr_t dim3, const dtype& dt)
    : m_expr_tree()
{
    intptr_t shape[4] = {dim0, dim1, dim2, dim3};
    intptr_t strides[4];
    strides[0] = (dim0 <= 1) ? 0 : dt.element_size() * dim1 * dim2 * dim3;
    strides[1] = (dim1 <= 1) ? 0 : dt.element_size() * dim2 * dim3;
    strides[2] = (dim2 <= 1) ? 0 : dt.element_size() * dim3;
    strides[3] = (dim3 <= 1) ? 0 : dt.element_size();

    char *originptr = 0;
    memory_block_ref memblock = make_fixed_size_pod_memory_block(dt.alignment(), dt.element_size() * dim0 * dim1 * dim2 * dim3, &originptr);
    m_expr_tree.reset(new strided_array_expr_node(dt, 4, shape, strides,
                            originptr, DND_MOVE(memblock)));
}

ndarray dnd::ndarray::index(int nindex, const irange *indices) const
{
    // Casting away const is ok here, because we pass 'false' to 'allow_in_place'
    return ndarray(apply_index_to_node(
                        const_cast<ndarray_expr_node *>(m_expr_tree.get()),
                        nindex, indices, false));
}

const ndarray dnd::ndarray::operator()(intptr_t idx) const
{
    // Casting away const is ok here, because we pass 'false' to 'allow_in_place'
    return ndarray(apply_integer_index_to_node(get_expr_tree(), 0, idx, false));
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

char *dnd::ndarray::get_readwrite_originptr() const
{
    if (m_expr_tree->get_node_type() == strided_array_node_type) {
        return static_cast<const strided_array_expr_node *>(m_expr_tree.get())->get_readwrite_originptr();
    } else {
        throw std::runtime_error("cannot get a readwrite origin ptr from this type of node");
    }
}

const char *dnd::ndarray::get_readonly_originptr() const
{
    switch (m_expr_tree->get_node_type()) {
    case strided_array_node_type:
        return static_cast<const strided_array_expr_node *>(m_expr_tree.get())->get_readwrite_originptr();
    case immutable_scalar_node_type:
        return static_cast<const immutable_scalar_node *>(m_expr_tree.get())->get_readonly_originptr();
    default:
        throw std::runtime_error("cannot get a readwrite origin ptr from this type of node");
    }
}

ndarray dnd::empty_like(const ndarray& rhs, const dtype& dt)
{
    // Sort the strides to get the memory layout ordering
    shortvector<int> axis_perm(rhs.get_ndim());
    strides_to_axis_perm(rhs.get_ndim(), rhs.get_strides(), axis_perm.get());

    // Construct the new array
    return ndarray(dt, rhs.get_ndim(), rhs.get_shape(), axis_perm.get());
}

static void val_assign_loop(const ndarray& lhs, const ndarray& rhs, assign_error_mode errmode)
{
    // Get the data pointer and strides of rhs through the standard interface
    dimvector rhs_original_strides(rhs.get_ndim());
    const char *rhs_originptr = NULL;
    rhs.get_expr_tree()->as_readonly_data_and_strides(rhs.get_ndim(), &rhs_originptr, rhs_original_strides.get());

    // Broadcast the 'rhs' shape to 'this'
    dimvector rhs_modified_strides(lhs.get_ndim());
    broadcast_to_shape(lhs.get_ndim(), lhs.get_shape(), rhs.get_ndim(), rhs.get_shape(), rhs_original_strides.get(), rhs_modified_strides.get());

    // Create the raw iterator
    raw_ndarray_iter<1,1> iter(lhs.get_ndim(), lhs.get_shape(), lhs.get_readwrite_originptr(), lhs.get_strides(),
                                        rhs_originptr, rhs_modified_strides.get());
    //iter.debug_dump(cout);

    intptr_t innersize = iter.innersize();
    intptr_t dst_innerstride = iter.innerstride<0>(), src_innerstride = iter.innerstride<1>();

    unary_specialization_kernel_instance assign;
    get_dtype_assignment_kernel(lhs.get_dtype(),
                                    rhs.get_dtype(),
                                    errmode,
                                    assign);
    unary_operation_t assign_fn = assign.specializations[
        get_unary_specialization(dst_innerstride, lhs.get_dtype().element_size(), src_innerstride, rhs.get_dtype().element_size())];

    if (innersize > 0) {
        do {
            assign_fn(iter.data<0>(), dst_innerstride,
                        iter.data<1>(), src_innerstride,
                        innersize, assign.auxdata);
        } while (iter.iternext());
    }
}

ndarray dnd::ndarray::storage() const
{
    const dtype& dt = get_dtype().storage_dtype();
    if (get_expr_tree()->get_node_type() == strided_array_node_type) {
        return ndarray(new strided_array_expr_node(dt, get_ndim(), get_shape(), get_strides(),
                        get_readwrite_originptr(), get_memory_block()));
    } else {
        throw std::runtime_error("Can only get the storage from dnd::ndarrays which are strided arrays");
    }
}


ndarray dnd::ndarray::as_dtype(const dtype& dt, assign_error_mode errmode) const
{
    if (dt == get_dtype().value_dtype()) {
        return *this;
    } else {
        return ndarray(m_expr_tree->as_dtype(dt, errmode, false));
    }
}

ndarray dnd::ndarray::view_as_dtype(const dtype& dt) const
{
    // Don't allow object dtypes
    if (get_dtype().is_object_type()) {
        std::stringstream ss;
        ss << "cannot view a dnd::ndarray with object dtype " << get_dtype() << " as another dtype";
        throw std::runtime_error(ss.str());
    } else if (dt.is_object_type()) {
        std::stringstream ss;
        ss << "cannot view an dnd::ndarray with POD dtype as another dtype " << dt;
        throw std::runtime_error(ss.str());
    }

    // Special case contiguous one dimensional arrays with a non-expression kind
    if (get_ndim() == 1 && get_expr_tree()->get_node_type() == strided_array_node_type &&
                            get_strides()[0] == get_dtype().element_size() &&
                            get_dtype().kind() != expression_kind) {
        intptr_t nbytes = get_shape()[0] * get_dtype().element_size();
        char *originptr = get_readwrite_originptr();

        if (nbytes % dt.element_size() != 0) {
            std::stringstream ss;
            ss << "cannot view dnd::ndarray with " << nbytes << " bytes as dtype " << dt << ", because its element size doesn't divide evenly";
            throw std::runtime_error(ss.str());
        }

        intptr_t shape[1], strides[1];
        shape[0] = nbytes / dt.element_size();
        strides[0] = dt.element_size();
        if ((((uintptr_t)originptr)&(dt.alignment()-1)) == 0) {
            // If the dtype's alignment is satisfied, can view it as is
            return ndarray(new strided_array_expr_node(dt, 1, shape, strides, originptr, get_memory_block()));
        } else {
            // The dtype's alignment was insufficient, so making it unaligned<>
            return ndarray(new strided_array_expr_node(make_unaligned_dtype(dt), 1, shape, strides, originptr, get_memory_block()));
        }
    }

    // For non-one dimensional and non-contiguous one dimensional arrays, the dtype element_size much match
    if (get_dtype().value_dtype().element_size() != dt.element_size()) {
        std::stringstream ss;
        ss << "cannot view dnd::ndarray with value dtype " << get_dtype().value_dtype() << " as dtype " << dt << " because they have different sizes, and the array is not contiguous one-dimensional";
        throw std::runtime_error(ss.str());
    }

    // In the case of a strided array with a non-expression dtype, simply substitute the dtype
    if (get_expr_tree()->get_node_type() == strided_array_node_type && get_dtype().kind() != expression_kind) {
        bool aligned = true;
        // If the alignment of the requested dtype is greater, check
        // the actual strides to only apply unaligned<> when necessary.
        if (dt.alignment() > get_dtype().value_dtype().alignment()) {
            uintptr_t aligncheck = (uintptr_t)get_readwrite_originptr();
            const intptr_t *strides = get_strides();
            for (int idim = 0; idim < get_ndim(); ++idim) {
                aligncheck |= (uintptr_t)strides[idim];
            }
            if ((aligncheck&(dt.alignment()-1)) != 0) {
                aligned = false;
            }
        }

        if (aligned) {
            return ndarray(new strided_array_expr_node(dt, get_ndim(), get_shape(), get_strides(),
                            get_readwrite_originptr(), get_memory_block()));
        } else {
            return ndarray(new strided_array_expr_node(make_unaligned_dtype(dt), get_ndim(), get_shape(), get_strides(),
                            get_readwrite_originptr(), get_memory_block()));
        }
    }

    // Finally, we've got some kind of expression array or expression_kind dtype,
    // so use the view_dtype.
    return ndarray(get_expr_tree()->as_dtype(
                make_view_dtype(dt, get_dtype().value_dtype()), assign_error_none, false));
}

// Implementation of ndarray.as<std::string>()
std::string dnd::detail::ndarray_as_string(const ndarray& lhs, assign_error_mode DND_UNUSED(errmode))
{
    if (lhs.get_ndim() != 0) {
        throw std::runtime_error("can only convert ndarrays with 0 dimensions to scalars");
    }
    ndarray tmp = lhs.vals();
    if (lhs.get_dtype().type_id() == fixedstring_type_id) {
        const fixedstring_dtype *fs = static_cast<const fixedstring_dtype *>(lhs.get_dtype().extended());
        if (fs->encoding() != string_encoding_ascii && fs->encoding() != string_encoding_utf_8) {
            tmp = tmp.as_dtype(make_fixedstring_dtype(string_encoding_utf_8, 4 * tmp.get_dtype().element_size() / tmp.get_dtype().alignment()));
            tmp = tmp.vals();
        }
        const char *data = tmp.get_readonly_originptr();
        intptr_t size = strnlen(data, tmp.get_dtype().element_size());
        return std::string(data, size);
    }

    stringstream ss;
    ss << "ndarray.as<string> isn't supported for dtype " << lhs.get_dtype() << " yet";
    throw runtime_error(ss.str());
}


void dnd::ndarray::val_assign(const ndarray& rhs, assign_error_mode errmode) const
{
    if (m_expr_tree->get_node_type() != strided_array_node_type) {
        throw std::runtime_error("cannot val_assign to an expression-view ndarray, must "
                                 "first convert it to a strided array with as_strided");
    }

    if (get_dtype() == rhs.get_dtype()) {
        // The dtypes match, simpler case
        val_assign_loop(*this, rhs, assign_error_none);
    } else if (get_num_elements() > 5 * rhs.get_num_elements()) {
        // If the data is being duplicated more than 5 times, make a temporary copy of rhs
        // converted to the dtype of 'this', then do the broadcasting.
        ndarray tmp = empty_like(rhs, get_dtype());
        val_assign_loop(tmp, rhs, errmode);
        val_assign_loop(*this, tmp, assign_error_none);
    } else {
        // Assignment with casting
        val_assign_loop(*this, rhs, errmode);
    }
}

void dnd::ndarray::val_assign(const dtype& dt, const char *data, assign_error_mode errmode) const
{
    //DEBUG_COUT << "scalar val_assign\n";
    scalar_copied_if_necessary src(get_dtype(), dt, data, errmode);
    raw_ndarray_iter<1,0> iter(*this);

    intptr_t innersize = iter.innersize(), innerstride = iter.innerstride<0>();

    unary_specialization_kernel_instance assign;
    get_dtype_assignment_kernel(get_dtype(), assign);
    unary_operation_t assign_fn = assign.specializations[
        get_unary_specialization(innerstride, get_dtype().element_size(), 0, dt.element_size())];

    if (innersize > 0) {
        do {
            //DEBUG_COUT << "scalar val_assign inner loop with size " << innersize << "\n";
            assign_fn(iter.data<0>(), innerstride, src.data(), 0, innersize, assign.auxdata);
        } while (iter.iternext());
    }
}

void dnd::ndarray::debug_dump(std::ostream& o = std::cerr) const
{
    o << "------ ndarray\n";
    if (m_expr_tree) {
        m_expr_tree->debug_dump(o, " ");
    } else {
        o << "NULL\n";
    }
    o << "------" << endl;
}

static void nested_ndarray_print(std::ostream& o, const dtype& d, const char *data, int ndim, const intptr_t *shape, const intptr_t *strides)
{
    if (ndim == 0) {
        d.print_element(o, data);
    } else {
        o << "{";
        if (ndim == 1) {
            d.print_element(o, data);
            for (intptr_t i = 1; i < shape[0]; ++i) {
                data += strides[0];
                o << ", ";
                d.print_element(o, data);
            }
        } else {
            intptr_t size = *shape;
            intptr_t stride = *strides;
            for (intptr_t k = 0; k < size; ++k) {
                nested_ndarray_print(o, d, data, ndim - 1, shape + 1, strides + 1);
                if (k + 1 != size) {
                    o << ", ";
                }
                data += stride;
            }
        }
        o << "}";
    }
}

std::ostream& dnd::operator<<(std::ostream& o, const ndarray& rhs)
{
    if (rhs.get_expr_tree() != NULL) {
        if (rhs.get_expr_tree()->get_node_category() == strided_array_node_category &&
                        rhs.get_dtype().kind() != expression_kind) {
            const char *originptr;
            dimvector strides(rhs.get_ndim());
            rhs.get_expr_tree()->as_readonly_data_and_strides(rhs.get_ndim(), &originptr, strides.get());
            o << "ndarray(" << rhs.get_dtype() << ", ";
            nested_ndarray_print(o, rhs.get_dtype(), originptr, rhs.get_ndim(), rhs.get_shape(), strides.get());
            o << ")";
        } else {
            o << rhs.vals();
        }
    } else {
        o << "ndarray()";
    }

    return o;
}
