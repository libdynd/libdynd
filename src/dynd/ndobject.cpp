//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/ndobject.hpp>
#include <dynd/ndobject_iter.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/bytes_dtype.hpp>
#include <dynd/dtypes/fixedbytes_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/exceptions.hpp>

using namespace std;
using namespace dynd;

ndobject::ndobject()
    : m_memblock()
{
}

void ndobject::swap(ndobject& rhs)
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

ndobject dynd::make_strided_ndobject(const dtype& uniform_dtype, int ndim, const intptr_t *shape,
                int64_t access_flags, const int *axis_perm)
{
    // Determine the total data size
    intptr_t element_size;
    if (uniform_dtype.extended()) {
        element_size = uniform_dtype.extended()->get_default_element_size(0, NULL);
    } else {
        element_size = uniform_dtype.element_size();
    }
    intptr_t size = element_size;
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
    ndo->m_flags = access_flags;

    // Fill in the ndobject metadata with strides and sizes
    strided_array_dtype_metadata *meta = reinterpret_cast<strided_array_dtype_metadata *>(ndo + 1);
    // Use the default construction to handle the uniform_dtype's metadata
    if (uniform_dtype.extended()) {
        uniform_dtype.extended()->metadata_default_construct(reinterpret_cast<char *>(meta + ndim), 0, NULL);
    }
    intptr_t stride = element_size;
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

ndobject dynd::make_strided_ndobject_from_data(const dtype& uniform_dtype, int ndim, const intptr_t *shape,
                const intptr_t *strides, int64_t access_flags, char *data_ptr,
                const memory_block_ptr& data_reference, char **out_uniform_metadata)
{
    if (out_uniform_metadata == NULL && uniform_dtype.extended() && uniform_dtype.extended()->get_metadata_size() > 0) {
        stringstream ss;
        ss << "Cannot make a strided ndobject with dtype " << uniform_dtype << " from a preexisting data pointer";
        throw runtime_error(ss.str());
    }

    dtype array_dtype = make_strided_array_dtype(uniform_dtype, ndim);

    // Allocate the ndobject metadata and data in one memory block
    memory_block_ptr result = make_ndobject_memory_block(array_dtype.extended()->get_metadata_size());

    // Fill in the preamble metadata
    ndobject_preamble *ndo = reinterpret_cast<ndobject_preamble *>(result.get());
    ndo->m_dtype = array_dtype.extended();
    extended_dtype_incref(ndo->m_dtype);
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = data_reference.get();
    memory_block_incref(ndo->m_data_reference);
    ndo->m_flags = access_flags;

    // Fill in the ndobject metadata with the shape and strides
    strided_array_dtype_metadata *meta = reinterpret_cast<strided_array_dtype_metadata *>(ndo + 1);
    for (int i = 0; i < ndim; ++i) {
        intptr_t dim_size = shape[i];
        meta[i].stride = dim_size > 1 ? strides[i] : 0;
        meta[i].size = dim_size;
    }

    // Return a pointer to the metadata for uniform_dtype.
    if (out_uniform_metadata != NULL) {
        *out_uniform_metadata = reinterpret_cast<char *>(meta + ndim);
    }

    return ndobject(result);
}

ndobject dynd::make_scalar_ndobject(const dtype& scalar_dtype, const void *data)
{
    size_t size = scalar_dtype.element_size();
    if (scalar_dtype.extended() && (size == 0 ||
                scalar_dtype.get_memory_management() != pod_memory_management ||
                scalar_dtype.extended()->is_uniform_dim() ||
                scalar_dtype.extended()->get_metadata_size() != 0)) {
        stringstream ss;
        ss << "Cannot make a dynd scalar from raw data using dtype " << scalar_dtype;
        throw runtime_error(ss.str());
    }

    // Allocate the ndobject metadata and data in one memory block
    char *data_ptr = NULL;
    memory_block_ptr result = make_ndobject_memory_block(0, size, scalar_dtype.alignment(), &data_ptr);

    // Fill in the preamble metadata
    ndobject_preamble *ndo = reinterpret_cast<ndobject_preamble *>(result.get());
    if (scalar_dtype.extended()) {
        ndo->m_dtype = scalar_dtype.extended();
        extended_dtype_incref(ndo->m_dtype);
    } else {
        ndo->m_dtype = reinterpret_cast<const extended_dtype *>(scalar_dtype.get_type_id());
    }
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = immutable_access_flag | read_access_flag;

    memcpy(data_ptr, data, size);

    return ndobject(result);
}

ndobject dynd::make_string_ndobject(const char *str, size_t len, string_encoding_t encoding)
{
    char *data_ptr = NULL, *string_ptr;
    dtype dt = make_string_dtype(encoding);
    ndobject result(make_ndobject_memory_block(dt.extended()->get_metadata_size(),
                        dt.element_size() + len, dt.alignment(), &data_ptr));
    // Set the string extents
    string_ptr = data_ptr + dt.element_size();
    ((char **)data_ptr)[0] = string_ptr;
    ((char **)data_ptr)[1] = string_ptr + len;
    // Copy the string data
    memcpy(string_ptr, str, len);
    // Set the ndobject metadata
    ndobject_preamble *ndo = result.get_ndo();
    ndo->m_dtype = dt.extended();
    extended_dtype_incref(ndo->m_dtype);
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = read_access_flag | immutable_access_flag;
    // Set the string metadata, telling the system that the string data was embedded in the ndobject memory
    string_dtype_metadata *ndo_meta = reinterpret_cast<string_dtype_metadata *>(result.get_ndo_meta());
    ndo_meta->blockref = NULL;
    return result;
}

ndobject dynd::make_utf8_array_ndobject(const char **cstr_array, size_t array_size)
{
    dtype dt = make_string_dtype(string_encoding_utf_8);
    ndobject result = make_strided_ndobject(array_size, dt);
    // Get the allocator for the output string dtype
    const string_dtype_metadata *md = reinterpret_cast<const string_dtype_metadata *>(result.get_ndo_meta() + sizeof(strided_array_dtype_metadata));
    memory_block_data *dst_memblock = md->blockref;
    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(dst_memblock);
    char **out_data = reinterpret_cast<char **>(result.get_ndo()->m_data_pointer);
    for (size_t i = 0; i < array_size; ++i) {
        size_t size = strlen(cstr_array[i]);
        allocator->allocate(dst_memblock, size, 1, &out_data[0], &out_data[1]);
        memcpy(out_data[0], cstr_array[i], size);
        out_data += 2;
    }
    allocator->finalize(dst_memblock);
    return result;
}

/**
 * Clones the metadata and swaps in a new dtype. The dtype must
 * have identical metadata, but this function doesn't check that.
 */
static ndobject make_ndobject_clone_with_new_dtype(const ndobject& n, const dtype& new_dt)
{
    ndobject result(shallow_copy_ndobject_memory_block(n.get_memblock()));
    ndobject_preamble *preamble = result.get_ndo();
    // Swap in the dtype
    if (!preamble->is_builtin_dtype()) {
        extended_dtype_decref(preamble->m_dtype);
    }
    if(new_dt.extended()) {
        preamble->m_dtype = new_dt.extended();
        extended_dtype_incref(preamble->m_dtype);
    } else {
        preamble->m_dtype = reinterpret_cast<extended_dtype *>(new_dt.get_type_id());
    }
    return result;
}


// Constructors from C++ scalars
ndobject::ndobject(dynd_bool value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
ndobject::ndobject(bool value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(dynd_bool(value)))
{
}
ndobject::ndobject(signed char value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
ndobject::ndobject(short value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
ndobject::ndobject(int value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
ndobject::ndobject(long value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
ndobject::ndobject(long long value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
ndobject::ndobject(unsigned char value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
ndobject::ndobject(unsigned short value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
ndobject::ndobject(unsigned int value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
ndobject::ndobject(unsigned long value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
ndobject::ndobject(unsigned long long value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
ndobject::ndobject(float value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
ndobject::ndobject(double value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
ndobject::ndobject(std::complex<float> value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
ndobject::ndobject(std::complex<double> value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
ndobject::ndobject(const std::string& value)
{
    ndobject temp = make_utf8_ndobject(value.c_str(), value.size());
    temp.swap(*this);
}
ndobject::ndobject(const char *cstr)
{
    ndobject temp = make_utf8_ndobject(cstr, strlen(cstr));
    temp.swap(*this);
}
ndobject::ndobject(const char *str, size_t size)
{
    ndobject temp = make_utf8_ndobject(str, size);
    temp.swap(*this);
}

ndobject::ndobject(const dtype& dt)
    : m_memblock(make_ndobject_memory_block(dt, 0, NULL))
{
}

ndobject::ndobject(const dtype& dt, intptr_t dim0)
    : m_memblock(make_ndobject_memory_block(dt, 1, &dim0))
{
}
ndobject::ndobject(const dtype& dt, intptr_t dim0, intptr_t dim1)
{
    intptr_t dims[2] = {dim0, dim1};
    m_memblock = make_ndobject_memory_block(dt, 2, dims);
}
ndobject::ndobject(const dtype& dt, intptr_t dim0, intptr_t dim1, intptr_t dim2)
{
    intptr_t dims[3] = {dim0, dim1, dim2};
    m_memblock = make_ndobject_memory_block(dt, 3, dims);
}

namespace {
    static dtype as_storage_type(const dtype& dt, const void *DYND_UNUSED(extra))
    {
        // If the dtype is a simple POD, switch it to a bytes dtype. Otherwise, keep it
        // the same so that the metadata layout is identical.
        const dtype& storage_dt = dt.storage_dtype();
        if (!storage_dt.extended() || (storage_dt.get_memory_management() == pod_memory_management &&
                                storage_dt.extended()->get_metadata_size() == 0)) {
            return make_fixedbytes_dtype(storage_dt.element_size(), storage_dt.alignment());
        } else if (storage_dt.get_type_id() == string_type_id) {
            return make_bytes_dtype(static_cast<const string_dtype *>(storage_dt.extended())->get_data_alignment());
        } else {
            return storage_dt;
        }
    }
} // anonymous namespace

ndobject ndobject::storage() const
{
    dtype storage_dt = get_dtype().with_transformed_scalar_types(&as_storage_type, NULL);
    return make_ndobject_clone_with_new_dtype(*this, storage_dt);
}

ndobject ndobject::at_array(int nindices, const irange *indices) const
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
            result.get_ndo()->m_dtype = reinterpret_cast<const extended_dtype *>(dt.get_type_id());
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

void ndobject::val_assign(const ndobject& rhs, assign_error_mode errmode,
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
            iter.get_uniform_dtype().prepare_kernel_auxdata(iter.metadata(), assign.auxdata);
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
            iter.get_uniform_dtype<0>().prepare_kernel_auxdata(iter.metadata<0>(), assign.auxdata);
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
        iter.get_uniform_dtype().prepare_kernel_auxdata(iter.metadata(), assign.auxdata);
        do {
            assign_fn(iter.data(), 0, data, 0, 1, assign.auxdata);
        } while (iter.next());
    }
}

ndobject ndobject::eval_immutable(const eval::eval_context *ectx) const
{
    if (get_access_flags()&immutable_access_flag) {
        return *this;
    } else {
        // Create a canonical dtype for the result
        const dtype& current_dtype = get_dtype();
        const dtype& dt = current_dtype.get_canonical_dtype();
        int ndim = current_dtype.get_uniform_ndim();
        dimvector shape(ndim);
        get_shape(shape.get());
        ndobject result(make_ndobject_memory_block(dt, ndim, shape.get()));
        // TODO: Reorder strides of strided dimensions in a KEEPORDER fashion
        result.val_assign(*this, assign_error_default, ectx);
        result.get_ndo()->m_flags = immutable_access_flag|read_access_flag;
        return result;
    }
}

ndobject ndobject::eval_copy(const eval::eval_context *ectx,
                    uint32_t access_flags) const
{
    const dtype& current_dtype = get_dtype();
    const dtype& dt = current_dtype.get_canonical_dtype();
    int ndim = current_dtype.get_uniform_ndim();
    dimvector shape(ndim);
    get_shape(shape.get());
    ndobject result(make_ndobject_memory_block(dt, ndim, shape.get()));
    // TODO: Reorder strides of strided dimensions in a KEEPORDER fashion
    result.val_assign(*this, assign_error_default, ectx);
    result.get_ndo()->m_flags = access_flags;
    return result;
}

ndobject ndobject::cast_scalars(const dtype& scalar_dtype, assign_error_mode errmode) const
{
    // This creates a dtype which has a convert dtype for every scalar of different dtype.
    // The result has the exact same metadata and data, so we just have to swap in the new
    // dtype in a shallow copy.
    dtype replaced_dtype = get_dtype().with_replaced_scalar_types(scalar_dtype, errmode);
    return make_ndobject_clone_with_new_dtype(*this, replaced_dtype);
}

namespace {
    static dtype view_scalar_type(const dtype& dt, const void *extra)
    {
        const dtype *e = reinterpret_cast<const dtype *>(extra);
        // If things aren't simple, use a view_dtype
        if (dt.get_kind() == expression_kind || dt.element_size() != e->element_size() ||
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
    if (uniform_ndim == 1 && array_dtype.get_type_id() == strided_array_type_id) {
        const strided_array_dtype *sad = static_cast<const strided_array_dtype *>(array_dtype.extended());
        const strided_array_dtype_metadata *md = reinterpret_cast<const strided_array_dtype_metadata *>(get_ndo_meta());
        size_t element_size = sad->get_element_dtype().element_size();
        if (element_size != 0 && element_size == md->stride &&
                    sad->get_element_dtype().get_kind() != expression_kind &&
                    sad->get_element_dtype().get_memory_management() == pod_memory_management) {
            intptr_t nbytes = md->size * element_size;
            // Make sure the element size divides into the # of bytes
            if (nbytes % scalar_dtype.element_size() != 0) {
                std::stringstream ss;
                ss << "cannot view ndobject with " << nbytes << " bytes as dtype ";
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
    return make_ndobject_clone_with_new_dtype(*this, viewed_dtype);
}

std::string dynd::detail::ndobject_as_string(const ndobject& lhs, assign_error_mode errmode)
{
    if (!lhs.is_scalar()) {
        throw std::runtime_error("can only convert ndobjects with 0 dimensions to scalars");
    }
    // Try to make a string directly from the data bytes if possible
    const dtype& lhs_dt = lhs.get_dtype();
    if (lhs_dt.get_kind() == string_kind) {
        const extended_string_dtype *esd = static_cast<const extended_string_dtype *>(lhs_dt.extended());
        string_encoding_t encoding = esd->get_encoding();
        if (encoding == string_encoding_utf_8 || encoding == string_encoding_ascii) {
            const char *begin, *end;
            esd->get_string_range(&begin, &end, lhs.get_readonly_originptr(), lhs.get_ndo_meta());
            return std::string(begin, end);
        }
    }

    // Otherwise cast it to a UTF8 string, then get the data bytes.
    ndobject temp = lhs.cast_scalars(make_string_dtype(string_encoding_utf_8));
    temp = temp.vals();
    const extended_string_dtype *esd = static_cast<const extended_string_dtype *>(temp.get_dtype().extended());
    const char *begin, *end;
    esd->get_string_range(&begin, &end, temp.get_readonly_originptr(), temp.get_ndo_meta());
    return std::string(begin, end);
}

void ndobject::debug_print(std::ostream& o, const std::string& indent) const
{
    o << indent << "------ ndobject\n";
    if (m_memblock.get()) {
        const ndobject_preamble *ndo = get_ndo();
        o << " address: " << (void *)m_memblock.get() << "\n";
        o << " refcount: " << ndo->m_memblockdata.m_use_count << "\n";
        o << " data pointer: " << (void *)ndo->m_data_pointer << "\n";
        o << " data reference: " << (void *)ndo->m_data_reference;
        if (ndo->m_data_reference == NULL) {
            o << " (embedded in ndobject memory)\n";
        } else {
            o << "\n";
        }
        o << " flags: " << ndo->m_flags << " (";
        if (ndo->m_flags & read_access_flag) o << "read_access ";
        if (ndo->m_flags & write_access_flag) o << "write_access ";
        if (ndo->m_flags & immutable_access_flag) o << "immutable ";
        o << ")\n";
        o << " dtype raw value: " << (void *)ndo->m_dtype << "\n";
        o << " dtype: " << get_dtype() << "\n";
        if (!ndo->is_builtin_dtype()) {
            o << " metadata:\n";
            ndo->m_dtype->metadata_debug_print(get_ndo_meta(), o, indent + "  ");
        }
    } else {
        o << indent << "NULL\n";
    }
    o << indent << "------" << endl;
}

std::ostream& dynd::operator<<(std::ostream& o, const ndobject& rhs)
{
    if (!rhs.empty()) {
        o << "ndobject(";
        ndobject v = rhs.vals();
        if (v.get_ndo()->is_builtin_dtype()) {
            print_builtin_scalar(v.get_ndo()->get_builtin_type_id(), o, v.get_ndo()->m_data_pointer);
        } else {
            v.get_ndo()->m_dtype->print_element(o, v.get_ndo()->m_data_pointer, v.get_ndo_meta());
        }
        o << ", " << rhs.get_dtype() << ")";
    } else {
        o << "ndobject()";
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
        return make_strided_ndobject(uniform_dtype, ndim, shape.get(), read_access_flag|write_access_flag, axis_perm.get());
    }
}

ndobject dynd::empty_like(const ndobject& rhs)
{
    // FIXME: This implementation only works for linearly strided arrays
    return empty_like(rhs, rhs.get_dtype().get_dtype_at_dimension(NULL, rhs.get_dtype().get_uniform_ndim()));
}

ndobject_vals::operator ndobject() const
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
