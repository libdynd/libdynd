//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/ndobject.hpp>
#include <dynd/ndobject_iter.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/exceptions.hpp>

using namespace std;
using namespace dynd;

dynd::ndobject::ndobject()
    : m_memblock()
{
}

void dynd::ndobject::swap(ndobject& rhs)
{
    m_memblock.swap(rhs.m_memblock);
}

template<class T>
inline typename enable_if<is_dtype_scalar<T>::value, memory_block_ptr>::type
make_immutable_builtin_scalar_ndobject(const T& value)
{
    char *data_ptr = NULL;
    memory_block_ptr result = make_ndobject_memory_block(0, sizeof(T), scalar_align_of<T>::value, &data_ptr);
    *reinterpret_cast<T *>(data_ptr) = value;
    ndobject_preamble *ndo = reinterpret_cast<ndobject_preamble *>(result.get());
    ndo->m_dtype = reinterpret_cast<extended_dtype *>(type_id_of<T>::value);
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = read_access_flag | immutable_access_flag;
    return result;
}

ndobject dynd::make_strided_ndobject(const dtype& uniform_dtype, int ndim, const intptr_t *shape, const int *axis_perm)
{
    // Determine the total data size
    intptr_t size = uniform_dtype.element_size();
    for (int i = 0; i < ndim; ++i) {
        size *= shape[i];
    }

    dtype array_dtype = make_strided_array_dtype(uniform_dtype, ndim);

    // Allocate the ndobject metadata and data in one memory block
    char *data_ptr = NULL;
    memory_block_ptr result = make_ndobject_memory_block(array_dtype.extended()->get_metadata_size(),
                    size, uniform_dtype.alignment(), &data_ptr);

    // Fill in the preamble metadata
    ndobject_preamble *ndo = reinterpret_cast<ndobject_preamble *>(result.get());
    ndo->m_dtype = array_dtype.extended();
    extended_dtype_incref(ndo->m_dtype);
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = read_access_flag | write_access_flag;

    // Fill in the ndobject metadata with C-order strides
    strided_array_dtype_metadata *meta = reinterpret_cast<strided_array_dtype_metadata *>(ndo + 1);
    intptr_t stride = uniform_dtype.element_size();
    if (axis_perm == NULL) {
        for (int i = ndim - 1; i >= 0; --i) {
            intptr_t dim_size = shape[i];
            meta[i].stride = dim_size > 1 ? stride : 0;
            meta[i].size = dim_size;
            stride *= dim_size;
        }
    } else {
        for (int i = 0; i < ndim; ++i) {
            int i_perm = axis_perm[i];
            intptr_t dim_size = shape[i_perm];
            meta[i_perm].stride = dim_size > 1 ? stride : 0;
            meta[i_perm].size = dim_size;
            stride *= dim_size;
        }
    }

    return ndobject(result);
}

// Constructors from C++ scalars
dynd::ndobject::ndobject(dynd_bool value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(bool value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(dynd_bool(value)))
{
}
dynd::ndobject::ndobject(signed char value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(short value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(int value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(long value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(long long value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(unsigned char value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(unsigned short value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(unsigned int value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(unsigned long value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(unsigned long long value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(float value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(double value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(std::complex<float> value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(std::complex<double> value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(const dtype& dt)
    : m_memblock(make_ndobject_memory_block(dt, 0, NULL))
{
}

dynd::ndobject::ndobject(const dtype& dt, intptr_t dim0)
    : m_memblock(make_ndobject_memory_block(dt, 1, &dim0))
{
}
dynd::ndobject::ndobject(const dtype& dt, intptr_t dim0, intptr_t dim1)
{
    intptr_t dims[2] = {dim0, dim1};
    m_memblock = make_ndobject_memory_block(dt, 2, dims);
}
dynd::ndobject::ndobject(const dtype& dt, intptr_t dim0, intptr_t dim1, intptr_t dim2)
{
    intptr_t dims[3] = {dim0, dim1, dim2};
    m_memblock = make_ndobject_memory_block(dt, 3, dims);
}

ndobject dynd::ndobject::at_array(int nindices, const irange *indices) const
{
    if (is_scalar()) {
        if (nindices != 0) {
            throw too_many_indices(nindices, 0);
        }
        return *this;
    } else {
        dtype this_dt(get_ndo()->m_dtype, true);
        dtype dt = get_ndo()->m_dtype->apply_linear_index(nindices, indices, 0, this_dt);
        ndobject result;
        if (dt.extended()) {
            result.set(make_ndobject_memory_block(dt.extended()->get_metadata_size()));
            result.get_ndo()->m_dtype = dt.extended();
            extended_dtype_incref(result.get_ndo()->m_dtype);
        } else {
            result.set(make_ndobject_memory_block(0));
            result.get_ndo()->m_dtype = reinterpret_cast<const extended_dtype *>(dt.type_id());
        }
        intptr_t offset = get_ndo()->m_dtype->apply_linear_index(nindices, indices, get_ndo()->m_data_pointer,
                        get_ndo_meta(), dt, result.get_ndo_meta(), 0, this_dt);
        result.get_ndo()->m_data_pointer = get_ndo()->m_data_pointer + offset;
        if (get_ndo()->m_data_reference) {
            result.get_ndo()->m_data_reference = get_ndo()->m_data_reference;
        } else {
            // If the data reference is NULL, the data is embedded in the ndobject itself
            result.get_ndo()->m_data_reference = m_memblock.get();
        }
        memory_block_incref(result.get_ndo()->m_data_reference);
        result.get_ndo()->m_flags = get_ndo()->m_flags;
        return result;
    }
}

void dynd::ndobject::val_assign(const ndobject& rhs, assign_error_mode errmode,
                    const eval::eval_context *ectx) const
{
    // Verify access permissions
    if (!(get_flags()&write_access_flag)) {
        throw runtime_error("tried to write to a dynd array that is not writeable");
    }
    if (!(rhs.get_flags()&read_access_flag)) {
        throw runtime_error("tried to read from a dynd array that is not readable");
    }

    if (rhs.is_scalar()) {
        unary_specialization_kernel_instance assign;
        const char *src_ptr = rhs.get_ndo()->m_data_pointer;

        // TODO: Performance optimization
        ndobject_iter<1, 0> iter(*this);
        get_dtype_assignment_kernel(iter.get_uniform_dtype(), rhs.get_dtype(), errmode, ectx, assign);
        unary_operation_t assign_fn = assign.specializations[scalar_unary_specialization];
        if (!iter.empty()) {
            do {
                assign_fn(iter.data(), 0, src_ptr, 0, 1, assign.auxdata);
            } while (iter.next());
        }
    } else {
        unary_specialization_kernel_instance assign;

        // TODO: Performance optimization
        ndobject_iter<1, 1> iter(*this, rhs);
        get_dtype_assignment_kernel(iter.get_uniform_dtype<0>(), iter.get_uniform_dtype<1>(), errmode, ectx, assign);
        unary_operation_t assign_fn = assign.specializations[scalar_unary_specialization];

        if (!iter.empty()) {
            do {
                assign_fn(iter.data<0>(), 0, iter.data<1>(), 0, 1, assign.auxdata);
            } while (iter.next());
        }
    }
}

void ndobject::val_assign(const dtype& dt, const char *data, assign_error_mode errmode,
                    const eval::eval_context *ectx) const
{
    // Verify access permissions
    if (!(get_flags()&write_access_flag)) {
        throw runtime_error("tried to write to a dynd array that is not writeable");
    }

    unary_specialization_kernel_instance assign;

    // TODO: Performance optimization
    ndobject_iter<1, 0> iter(*this);
    get_dtype_assignment_kernel(iter.get_uniform_dtype(), dt, errmode, ectx, assign);
    unary_operation_t assign_fn = assign.specializations[scalar_unary_specialization];
    if (!iter.empty()) {
        do {
            assign_fn(iter.data(), 0, data, 0, 1, assign.auxdata);
        } while (iter.next());
    }
}

ndobject ndobject::cast_scalars(const dtype& scalar_dtype, assign_error_mode errmode) const
{
    // This creates a dtype which has a convert dtype for every scalar of different dtype.
    // The result has the exact same metadata and data, so we just have to swap in the new
    // dtype in a shallow copy.
    dtype replaced_dtype = get_dtype().with_replaced_scalar_types(scalar_dtype, errmode);

    ndobject result(shallow_copy_ndobject_memory_block(m_memblock));
    ndobject_preamble *preamble = result.get_ndo();
    // Swap in the dtype
    if (!preamble->is_builtin_dtype()) {
        extended_dtype_decref(preamble->m_dtype);
    }
    if(replaced_dtype.extended()) {
        preamble->m_dtype = replaced_dtype.extended();
        extended_dtype_incref(preamble->m_dtype);
    } else {
        preamble->m_dtype = reinterpret_cast<extended_dtype *>(replaced_dtype.type_id());
    }
    return result;
}

namespace {
    static dtype view_scalar_type(const dtype& dt, const void *extra)
    {
        const dtype *e = reinterpret_cast<const dtype *>(extra);
        // If things aren't simple, use a view_dtype
        if (dt.kind() == expression_kind || dt.element_size() != e->element_size() ||
                    dt.get_memory_management() != pod_memory_management ||
                    e->get_memory_management() != pod_memory_management) {
            return make_view_dtype(*e, dt);
        } else {
            return *e;
        }
    }
} // anonymous namespace

ndobject ndobject::view_scalars(const dtype& scalar_dtype) const
{
    const dtype& array_dtype = get_dtype();
    int uniform_ndim = array_dtype.get_uniform_ndim();
    // First check if we're dealing with a simple one dimensional block of memory we can reinterpret
    // at will.
    if (uniform_ndim == 1 && array_dtype.type_id() == strided_array_type_id) {
        const strided_array_dtype *sad = static_cast<const strided_array_dtype *>(array_dtype.extended());
        const strided_array_dtype_metadata *md = reinterpret_cast<const strided_array_dtype_metadata *>(get_ndo_meta());
        size_t element_size = sad->get_element_dtype().element_size();
        if (element_size != 0 && element_size == md->stride &&
                    sad->get_element_dtype().kind() != expression_kind &&
                    sad->get_element_dtype().get_memory_management() == pod_memory_management) {
            intptr_t nbytes = md->size * element_size;
            // Make sure the element size divides into the # of bytes
            if (nbytes % scalar_dtype.element_size() != 0) {
                std::stringstream ss;
                ss << "cannot view dynd::ndobject with " << nbytes << " bytes as dtype ";
                ss << scalar_dtype << ", because its element size " << scalar_dtype.element_size();
                ss << " doesn't divide evenly into the total array size " << nbytes;
                throw std::runtime_error(ss.str());
            }
            // Create the result array, adjusting the dtype if the data isn't aligned correctly
            char *data_ptr = get_ndo()->m_data_pointer;
            dtype result_dtype;
            if ((((uintptr_t)data_ptr)&(scalar_dtype.alignment()-1)) == 0) {
                result_dtype = make_strided_array_dtype(scalar_dtype);
            } else {
                result_dtype = make_strided_array_dtype(make_unaligned_dtype(scalar_dtype));
            }
            ndobject result(make_ndobject_memory_block(result_dtype.extended()->get_metadata_size()));
            // Copy all the ndobject metadata fields
            result.get_ndo()->m_data_pointer = get_ndo()->m_data_pointer;
            if (get_ndo()->m_data_reference) {
                result.get_ndo()->m_data_reference = get_ndo()->m_data_reference;
            } else {
                result.get_ndo()->m_data_reference = m_memblock.get();
            }
            memory_block_incref(result.get_ndo()->m_data_reference);
            result.get_ndo()->m_dtype = result_dtype.extended();
            extended_dtype_incref(result.get_ndo()->m_dtype);
            result.get_ndo()->m_flags = get_ndo()->m_flags;
            // The result has one strided ndarray field
            strided_array_dtype_metadata *result_md = reinterpret_cast<strided_array_dtype_metadata *>(result.get_ndo_meta());
            result_md->size = nbytes / scalar_dtype.element_size();
            result_md->stride = scalar_dtype.element_size();
            return result;
        }
    }

    const dtype& viewed_dtype = array_dtype.with_transformed_scalar_types(view_scalar_type, &scalar_dtype);

    ndobject result(shallow_copy_ndobject_memory_block(m_memblock));
    ndobject_preamble *preamble = result.get_ndo();
    // Swap in the dtype
    if (!preamble->is_builtin_dtype()) {
        extended_dtype_decref(preamble->m_dtype);
    }
    if(viewed_dtype.extended()) {
        preamble->m_dtype = viewed_dtype.extended();
        extended_dtype_incref(preamble->m_dtype);
    } else {
        preamble->m_dtype = reinterpret_cast<extended_dtype *>(viewed_dtype.type_id());
    }
    return result;
}


void ndobject::debug_dump(std::ostream& o, const std::string& indent) const
{
    o << indent << "------ ndobject\n";
    if (m_memblock.get()) {
        const ndobject_preamble *ndo = get_ndo();
        o << " address: " << (void *)m_memblock.get() << "\n";
        o << " refcount: " << ndo->m_memblockdata.m_use_count << "\n";
        o << " data pointer: " << (void *)ndo->m_data_pointer << "\n";
        o << " data reference: " << (void *)ndo->m_data_reference << "\n";
        o << " flags: " << ndo->m_flags << " (";
        if (ndo->m_flags & read_access_flag) o << "read_access ";
        if (ndo->m_flags & write_access_flag) o << "write_access ";
        if (ndo->m_flags & immutable_access_flag) o << "immutable ";
        o << ")\n";
        o << " dtype raw value: " << (void *)ndo->m_dtype << "\n";
        o << " dtype: " << get_dtype() << "\n";
        if (!ndo->is_builtin_dtype()) {
            o << " metadata:\n";
            ndo->m_dtype->metadata_debug_dump(get_ndo_meta(), o, indent + " ");
        }
    } else {
        o << indent << "NULL\n";
    }
    o << indent << "------" << endl;
}

std::ostream& dynd::operator<<(std::ostream& o, const ndobject& rhs)
{
    if (rhs.get_ndo() != NULL) {
        if (rhs.get_ndo()->is_builtin_dtype()) {
            print_builtin_scalar(rhs.get_ndo()->get_builtin_type_id(), o, rhs.get_ndo()->m_data_pointer);
        } else {
            rhs.get_ndo()->m_dtype->print_element(o, rhs.get_ndo()->m_data_pointer, rhs.get_ndo_meta());
        }
    } else {
        o << "<null>";
    }
    return o;
}

ndobject dynd::empty_like(const ndobject& rhs, const dtype& uniform_dtype)
{
    // FIXME: This implementation only works for linearly strided arrays
    if (rhs.is_scalar()) {
        return ndobject(uniform_dtype);
    } else {
        int ndim = rhs.get_ndo()->m_dtype->get_uniform_ndim();
        dimvector shape(ndim), strides(ndim);
        rhs.get_shape(shape.get());
        rhs.get_strides(strides.get());
        shortvector<int> axis_perm(ndim);
        strides_to_axis_perm(ndim, strides.get(), axis_perm.get());
        return make_strided_ndobject(uniform_dtype, ndim, shape.get(), axis_perm.get());
    }
}

ndobject dynd::empty_like(const ndobject& rhs)
{
    // FIXME: This implementation only works for linearly strided arrays
    return empty_like(rhs, rhs.get_dtype().get_dtype_at_dimension(rhs.get_dtype().get_uniform_ndim()));
}

dynd::ndobject_vals::operator ndobject() const
{
    // Create a canonical dtype for the result
    const dtype& current_dtype = m_arr.get_dtype();
    const dtype& dt = current_dtype.get_canonical_dtype();

    if (dt == current_dtype) {
        return m_arr;
    } else {
        // If the canonical dtype is different, make a copy of the array
        int ndim = current_dtype.get_uniform_ndim();
        dimvector shape(ndim);
        m_arr.get_shape(shape.get());
        ndobject result(make_ndobject_memory_block(dt, ndim, shape.get()));
        // TODO: Reorder strides of strided dimensions in a KEEPORDER fashion
        result.val_assign(m_arr);
        return result;
    }
}
