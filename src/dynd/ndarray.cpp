//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// DEPRECATED (replacement in ndobject.cpp)

#include <dynd/ndarray.hpp>
#include <dynd/scalars.hpp>
#include <dynd/raw_iteration.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/buffer_storage.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>

#include <dynd/nodes/scalar_node.hpp>
#include <dynd/nodes/immutable_scalar_node.hpp>
#include <dynd/nodes/immutable_builtin_scalar_node.hpp>
#include <dynd/nodes/strided_ndarray_node.hpp>
#include <dynd/nodes/elwise_binary_kernel_node.hpp>

using namespace std;
using namespace dynd;

dynd::ndarray::ndarray()
    : m_node()
{
}

dynd::ndarray::ndarray(dynd_bool value)
    : m_node(make_immutable_builtin_scalar_node(value))
{
}
dynd::ndarray::ndarray(signed char value)
    : m_node(make_immutable_builtin_scalar_node(value))
{
}
dynd::ndarray::ndarray(short value)
    : m_node(make_immutable_builtin_scalar_node(value))
{
}
dynd::ndarray::ndarray(int value)
    : m_node(make_immutable_builtin_scalar_node(value))
{
}
dynd::ndarray::ndarray(long value)
    : m_node(make_immutable_builtin_scalar_node(value))
{
}
dynd::ndarray::ndarray(long long value)
    : m_node(make_immutable_builtin_scalar_node(value))
{
}
dynd::ndarray::ndarray(unsigned char value)
    : m_node(make_immutable_builtin_scalar_node(value))
{
}
dynd::ndarray::ndarray(unsigned short value)
    : m_node(make_immutable_builtin_scalar_node(value))
{
}
dynd::ndarray::ndarray(unsigned int value)
    : m_node(make_immutable_builtin_scalar_node(value))
{
}
dynd::ndarray::ndarray(unsigned long value)
    : m_node(make_immutable_builtin_scalar_node(value))
{
}
dynd::ndarray::ndarray(unsigned long long value)
    : m_node(make_immutable_builtin_scalar_node(value))
{
}
dynd::ndarray::ndarray(float value)
    : m_node(make_immutable_builtin_scalar_node(value))
{
}
dynd::ndarray::ndarray(double value)
    : m_node(make_immutable_builtin_scalar_node(value))
{
}
dynd::ndarray::ndarray(complex<float> value)
    : m_node(make_immutable_builtin_scalar_node(value))
{
}
dynd::ndarray::ndarray(complex<double> value)
    : m_node(make_immutable_builtin_scalar_node(value))
{
}

// Makes a UTF8 string from a std::string. This currently creates
// a memory block for the string data and a scalar node for the data.
// TODO: Pack the string data into the same scalar node to reduce this to
//       a single memory allocation.
dynd::ndarray::ndarray(const std::string& value)
    : m_node()
{
    char *blockref_dataptr = NULL;
    memory_block_ptr blockref_memblock(make_fixed_size_pod_memory_block(value.size(), 1, &blockref_dataptr));
    memcpy(blockref_dataptr, value.c_str(), value.size());
    char *refs[2] = {blockref_dataptr, blockref_dataptr + value.size()};
    m_node = make_scalar_node(make_string_dtype(string_encoding_utf_8),
                    reinterpret_cast<const char *>(&refs), read_access_flag | immutable_access_flag,
                    blockref_memblock);
}

dynd::ndarray::ndarray(const dtype& dt)
    : m_node()
{
    char *originptr = 0;
    memory_block_ptr memblock = make_fixed_size_pod_memory_block(dt.element_size(), dt.alignment(), &originptr);
    make_strided_ndarray_node(dt, 0, NULL, NULL,
                            originptr, read_access_flag | write_access_flag,
                            DYND_MOVE(memblock)).swap(m_node);
}

dynd::ndarray::ndarray(const dtype& dt, const char *raw_data)
    : m_node(make_immutable_scalar_node(dt, raw_data))
{
}

dynd::ndarray::ndarray(const dtype& dt, int ndim, const intptr_t *shape, const int *axis_perm)
    : m_node(make_strided_ndarray_node(dt, ndim, shape, axis_perm)) {
}


dynd::ndarray::ndarray(const ndarray_node_ptr& expr_tree)
    : m_node(expr_tree)
{
}

#if defined(DYND_RVALUE_REFS)
dynd::ndarray::ndarray(ndarray_node_ptr&& expr_tree)
    : m_node(DYND_MOVE(expr_tree))
{
}
#endif // defined(DYND_RVALUE_REFS)

dynd::ndarray::ndarray(intptr_t dim0, const dtype& dt)
    : m_node()
{
    intptr_t stride = (dim0 <= 1) ? 0 : dt.element_size();
    char *originptr = 0;
    memory_block_ptr memblock = make_fixed_size_pod_memory_block(dt.element_size() * dim0, dt.alignment(), &originptr);
    make_strided_ndarray_node(dt, 1, &dim0, &stride,
                            originptr, read_access_flag | write_access_flag,
                            DYND_MOVE(memblock)).swap(m_node);
}

dynd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, const dtype& dt)
    : m_node()
{
    intptr_t shape[2] = {dim0, dim1};
    intptr_t strides[2];
    strides[0] = (dim0 <= 1) ? 0 : dt.element_size() * dim1;
    strides[1] = (dim1 <= 1) ? 0 : dt.element_size();

    char *originptr = 0;
    memory_block_ptr memblock = make_fixed_size_pod_memory_block(dt.element_size() * dim0 * dim1, dt.alignment(), &originptr);
    make_strided_ndarray_node(dt, 2, shape, strides,
                            originptr, read_access_flag | write_access_flag,
                            DYND_MOVE(memblock)).swap(m_node);
}

dynd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, intptr_t dim2, const dtype& dt)
    : m_node()
{
    intptr_t shape[3] = {dim0, dim1, dim2};
    intptr_t strides[3];
    strides[0] = (dim0 <= 1) ? 0 : dt.element_size() * dim1 * dim2;
    strides[1] = (dim1 <= 1) ? 0 : dt.element_size() * dim2;
    strides[2] = (dim2 <= 1) ? 0 : dt.element_size();

    char *originptr = 0;
    memory_block_ptr memblock = make_fixed_size_pod_memory_block(dt.element_size() * dim0 * dim1 * dim2, dt.alignment(), &originptr);
    make_strided_ndarray_node(dt, 3, shape, strides,
                            originptr, read_access_flag | write_access_flag,
                            DYND_MOVE(memblock)).swap(m_node);
}

dynd::ndarray::ndarray(intptr_t dim0, intptr_t dim1, intptr_t dim2, intptr_t dim3, const dtype& dt)
    : m_node()
{
    intptr_t shape[4] = {dim0, dim1, dim2, dim3};
    intptr_t strides[4];
    strides[0] = (dim0 <= 1) ? 0 : dt.element_size() * dim1 * dim2 * dim3;
    strides[1] = (dim1 <= 1) ? 0 : dt.element_size() * dim2 * dim3;
    strides[2] = (dim2 <= 1) ? 0 : dt.element_size() * dim3;
    strides[3] = (dim3 <= 1) ? 0 : dt.element_size();

    char *originptr = 0;
    memory_block_ptr memblock = make_fixed_size_pod_memory_block(dt.element_size() * dim0 * dim1 * dim2 * dim3, dt.alignment(), &originptr);
    make_strided_ndarray_node(dt, 4, shape, strides,
                            originptr, read_access_flag | write_access_flag,
                            DYND_MOVE(memblock)).swap(m_node);
}

ndarray dynd::ndarray::index(int nindex, const irange *indices) const
{
    return ndarray(apply_index_to_node(m_node, nindex, indices, false));
}

const ndarray dynd::ndarray::operator()(intptr_t idx) const
{
    return ndarray(apply_integer_index_to_node(m_node, 0, idx, false));
}

ndarray& dynd::ndarray::operator=(const ndarray& rhs)
{
    m_node = rhs.m_node;
    return *this;
}

void dynd::ndarray::swap(ndarray& rhs)
{
    m_node.swap(rhs.m_node);
}

ndarray dynd::empty_like(const ndarray& rhs, const dtype& dt)
{
    // Sort the strides to get the memory layout ordering
    shortvector<int> axis_perm(rhs.get_ndim());
    strides_to_axis_perm(rhs.get_ndim(), rhs.get_strides(), axis_perm.get());

    // Construct the new array
    return ndarray(dt, rhs.get_ndim(), rhs.get_shape(), axis_perm.get());
}

ndarray dynd::ndarray::storage() const
{
    if (get_node()->get_category() == strided_array_node_category) {
        int access_flags = m_node->get_access_flags();
        if (access_flags&write_access_flag) {
            return ndarray(make_strided_ndarray_node(m_node->get_dtype().storage_dtype(),
                        m_node->get_ndim(), m_node->get_shape(), m_node->get_strides(),
                         m_node->get_readwrite_originptr(),
                        access_flags, m_node->get_data_memory_block()));
        } else {
            return ndarray(make_strided_ndarray_node(m_node->get_dtype().storage_dtype(),
                        m_node->get_ndim(), m_node->get_shape(), m_node->get_strides(),
                        m_node->get_readonly_originptr(),
                        access_flags, m_node->get_data_memory_block()));
        }
    } else {
        throw std::runtime_error("Can only get the storage from strided dynd::ndarrays");
    }
}


ndarray dynd::ndarray::as_dtype(const dtype& dt, assign_error_mode errmode) const
{
    if (dt == get_dtype().value_dtype()) {
        return *this;
    } else {
        return ndarray(m_node->as_dtype(dt, errmode, false));
    }
}

ndarray dynd::ndarray::view_as_dtype(const dtype& dt) const
{
    // Don't allow object dtypes
    if (get_dtype().get_memory_management() != pod_memory_management) {
        std::stringstream ss;
        ss << "cannot view a dynd::ndarray with object dtype " << get_dtype() << " as another dtype";
        throw std::runtime_error(ss.str());
    } else if (dt.get_memory_management() != pod_memory_management) {
        std::stringstream ss;
        ss << "cannot view an dynd::ndarray with POD dtype as another dtype " << dt;
        throw std::runtime_error(ss.str());
    }

    // Special case contiguous one dimensional arrays with a non-expression kind
    if (get_ndim() == 1 && get_node()->get_category() == strided_array_node_category &&
                            get_strides()[0] > 0 && //
                            static_cast<size_t>(get_strides()[0]) == get_dtype().element_size() &&
                            get_dtype().get_kind() != expression_kind) {
        intptr_t nbytes = get_shape()[0] * get_dtype().element_size();
        char *originptr = get_readwrite_originptr();

        if (nbytes % dt.element_size() != 0) {
            std::stringstream ss;
            ss << "cannot view dynd::ndarray with " << nbytes << " bytes as dtype " << dt << ", because its element size doesn't divide evenly";
            throw std::runtime_error(ss.str());
        }

        intptr_t shape[1], strides[1];
        shape[0] = nbytes / dt.element_size();
        strides[0] = dt.element_size();
        if ((((uintptr_t)originptr)&(dt.alignment()-1)) == 0) {
            // If the dtype's alignment is satisfied, can view it as is
            return ndarray(make_strided_ndarray_node(dt, 1, shape, strides, originptr,
                                get_node()->get_access_flags(), m_node->get_data_memory_block()));
        } else {
            // The dtype's alignment was insufficient, so making it unaligned<>
            return ndarray(make_strided_ndarray_node(make_unaligned_dtype(dt), 1, shape, strides, originptr,
                            get_node()->get_access_flags(), m_node->get_data_memory_block()));
        }
    }

    // For non-one dimensional and non-contiguous one dimensional arrays, the dtype element_size much match
    if (get_dtype().value_dtype().element_size() != dt.element_size()) {
        std::stringstream ss;
        ss << "cannot view dynd::ndarray with value dtype " << get_dtype().value_dtype() << " as dtype " << dt << " because they have different sizes, and the array is not contiguous one-dimensional";
        throw std::runtime_error(ss.str());
    }

    // In the case of a strided array with a non-expression dtype, simply substitute the dtype
    if (get_node()->get_category() == strided_array_node_category && get_dtype().get_kind() != expression_kind) {
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
            return ndarray(make_strided_ndarray_node(dt, get_ndim(), get_shape(), get_strides(),
                            get_readwrite_originptr(), read_access_flag | write_access_flag, m_node->get_data_memory_block()));
        } else {
            return ndarray(make_strided_ndarray_node(make_unaligned_dtype(dt), get_ndim(), get_shape(), get_strides(),
                            get_readwrite_originptr(), read_access_flag | write_access_flag, m_node->get_data_memory_block()));
        }
    }

    // Finally, we've got some kind of expression array or expression_kind dtype,
    // so use the view_dtype.
    return ndarray(get_node()->as_dtype(
                make_view_dtype(dt, get_dtype().value_dtype()), assign_error_none, false));
}

// Implementation of ndarray.as<std::string>()
std::string dynd::detail::ndarray_as_string(const ndarray& lhs, assign_error_mode DYND_UNUSED(errmode))
{
    if (lhs.get_ndim() != 0) {
        throw std::runtime_error("can only convert ndarrays with 0 dimensions to scalars");
    }
    ndarray tmp = lhs.vals();
    if (tmp.get_dtype().get_type_id() == fixedstring_type_id) {
        const extended_string_dtype *fs = static_cast<const extended_string_dtype *>(tmp.get_dtype().extended());
        if (fs->get_encoding() == string_encoding_ascii || fs->get_encoding() == string_encoding_utf_8) {
            const char *data = tmp.get_readonly_originptr();
            intptr_t size = strnlen(data, tmp.get_dtype().element_size());
            return std::string(data, size);
        } else {
            tmp = tmp.as_dtype(make_string_dtype(string_encoding_utf_8));
            tmp = tmp.vals();
        }
    }

    if (tmp.get_dtype().get_type_id() == string_type_id) {
        const extended_string_dtype *fs = static_cast<const extended_string_dtype *>(tmp.get_dtype().extended());
        // Make sure it's ASCII or UTF8
        if (fs->get_encoding() != string_encoding_ascii && fs->get_encoding() != string_encoding_utf_8) {
            tmp = tmp.as_dtype(make_string_dtype(string_encoding_utf_8));
            tmp = tmp.vals();
            fs = static_cast<const extended_string_dtype *>(tmp.get_dtype().extended());
        }
        const char * const *data = reinterpret_cast<const char * const *>(tmp.get_readonly_originptr());
        return std::string(data[0], data[1]);
    }

    stringstream ss;
    ss << "ndarray.as<string> isn't supported for dtype " << lhs.get_dtype() << " yet";
    throw runtime_error(ss.str());
}


static void val_assign_loop(const ndarray& lhs, const ndarray& rhs, assign_error_mode errmode, const eval::eval_context *ectx)
{
    // Get the data pointer and strides of rhs through the standard interface
    const char *rhs_originptr = rhs.get_readonly_originptr();
    const intptr_t *rhs_original_strides = rhs.get_strides();

    // Broadcast the 'rhs' shape to 'this'
    dimvector rhs_modified_strides(lhs.get_ndim());
    broadcast_to_shape(lhs.get_ndim(), lhs.get_shape(), rhs.get_ndim(), rhs.get_shape(), rhs_original_strides, rhs_modified_strides.get());

    // Create the raw iterator
    raw_ndarray_iter<1,1> iter(lhs.get_ndim(), lhs.get_shape(), lhs.get_readwrite_originptr(), lhs.get_strides(),
                                        rhs_originptr, rhs_modified_strides.get());
    //iter.debug_print(cout);

    intptr_t innersize = iter.innersize();
    intptr_t dst_innerstride = iter.innerstride<0>(), src_innerstride = iter.innerstride<1>();

    unary_specialization_kernel_instance assign;
    get_dtype_assignment_kernel(lhs.get_dtype(), rhs.get_dtype(),
                                    errmode, ectx,
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

void dynd::ndarray::val_assign(const ndarray& rhs, assign_error_mode errmode, const eval::eval_context *ectx) const
{
    if (get_dtype() == rhs.get_dtype()) {
        val_assign_loop(*this, rhs, assign_error_none, ectx);
    } else if (get_num_elements() <= 5 * rhs.get_num_elements() ) {
        val_assign_loop(*this, rhs, errmode, ectx);
    } else {
        // If the data is being duplicated more than 5 times, make a temporary copy of rhs
        // converted to the dtype of 'this', then do the broadcasting.
        ndarray tmp = empty_like(rhs, get_dtype());
        val_assign_loop(tmp, rhs, errmode, ectx);
        val_assign_loop(*this, tmp, assign_error_none, ectx);
    }
}

void dynd::ndarray::val_assign(const dtype& dt, const char *data, assign_error_mode errmode, const eval::eval_context *ectx) const
{
    //cout << "scalar val_assign " << dt << " ptr " << (const void *)data << "\n";
    scalar_copied_if_necessary src(get_dtype(), dt, data, errmode, ectx);
    raw_ndarray_iter<1,0> iter(*this);

    intptr_t innersize = iter.innersize(), innerstride = iter.innerstride<0>();

    unary_specialization_kernel_instance assign;
    get_dtype_assignment_kernel(get_dtype(), assign);
    unary_operation_t assign_fn = assign.specializations[
        get_unary_specialization(innerstride, get_dtype().element_size(), 0, dt.element_size())];

    if (innersize > 0) {
        do {
            //cout << "scalar val_assign inner loop with size " << innersize << "\n";
            assign_fn(iter.data<0>(), innerstride, src.data(), 0, innersize, assign.auxdata);
        } while (iter.iternext());
    }
}

ndarray dynd::ndarray::eval_immutable(const eval::eval_context *ectx) const
{
    return ndarray(evaluate(m_node.get_node(), ectx, false, read_access_flag|immutable_access_flag));
}

ndarray dynd::ndarray::eval_copy(const eval::eval_context *ectx, uint32_t access_flags) const
{
    return ndarray(evaluate(m_node.get_node(), ectx, true, access_flags));
}

bool dynd::ndarray::equals_exact(const ndarray& rhs) const
{
    if (get_node() == rhs.get_node()) {
        return true;
    } else if (get_node() != NULL && rhs.get_node() != NULL) {
        throw runtime_error("ndarray::equals_exact is not yet implemented");
    } else {
        return false;
    }
}


void dynd::ndarray::debug_print(std::ostream& o = std::cerr, const std::string& indent) const
{
    o << indent << "------ ndarray\n";
    if (m_node.get()) {
        m_node->debug_print(o, indent + " ");
    } else {
        o << indent << "NULL\n";
    }
    o << indent << "------" << endl;
}

static void nested_ndarray_print(std::ostream& o, const dtype& d, const char *data, int ndim, const intptr_t *shape, const intptr_t *strides)
{
    if (ndim == 0) {
        d.print_element(o, data, NULL); // TODO: ndobject metadata
    } else {
        o << "[";
        if (ndim == 1) {
            d.print_element(o, data, NULL); // TODO: ndobject metadata
            for (intptr_t i = 1; i < shape[0]; ++i) {
                data += strides[0];
                o << ", ";
                d.print_element(o, data, NULL); // TODO: ndobject metadata
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
        o << "]";
    }
}

std::ostream& dynd::operator<<(std::ostream& o, const ndarray& rhs)
{
    if (rhs.get_node() != NULL) {
        o << "ndarray(";
        if (rhs.get_node()->get_category() == strided_array_node_category &&
                        rhs.get_dtype().get_kind() != expression_kind) {
            const char *originptr = rhs.get_node()->get_readonly_originptr();
            const intptr_t *strides = rhs.get_node()->get_strides();
            nested_ndarray_print(o, rhs.get_dtype(), originptr, rhs.get_ndim(), rhs.get_shape(), strides);
        } else {
            ndarray tmp = rhs.vals();
            const char *originptr = tmp.get_node()->get_readonly_originptr();
            const intptr_t *strides = tmp.get_node()->get_strides();
            nested_ndarray_print(o, tmp.get_dtype(), originptr, tmp.get_ndim(), tmp.get_shape(), strides);
        }
        o << ", " << rhs.get_dtype();
        o << ")";
    } else {
        o << "ndarray()";
    }

    return o;
}
