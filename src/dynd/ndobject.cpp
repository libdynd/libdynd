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
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/gfunc/callable.hpp>

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
inline typename dynd::enable_if<is_dtype_scalar<T>::value, memory_block_ptr>::type
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
    if (!uniform_dtype.is_builtin()) {
        element_size = uniform_dtype.extended()->get_default_data_size(0, NULL);
    } else {
        element_size = uniform_dtype.get_data_size();
    }
    intptr_t size = element_size;
    for (int i = 0; i < ndim; ++i) {
        size *= shape[i];
    }

    dtype array_dtype = make_strided_array_dtype(uniform_dtype, ndim);

    // Allocate the ndobject metadata and data in one memory block
    char *data_ptr = NULL;
    memory_block_ptr result = make_ndobject_memory_block(array_dtype.extended()->get_metadata_size(),
                    size, uniform_dtype.get_alignment(), &data_ptr);

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
    if (!uniform_dtype.is_builtin()) {
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
    if (out_uniform_metadata == NULL && !uniform_dtype.is_builtin() && uniform_dtype.extended()->get_metadata_size() > 0) {
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
    size_t size = scalar_dtype.get_data_size();
    if (!scalar_dtype.is_builtin() && (size == 0 ||
                scalar_dtype.get_memory_management() != pod_memory_management ||
                scalar_dtype.extended()->get_kind() == uniform_array_kind ||
                scalar_dtype.extended()->get_metadata_size() != 0)) {
        stringstream ss;
        ss << "Cannot make a dynd scalar from raw data using dtype " << scalar_dtype;
        throw runtime_error(ss.str());
    }

    // Allocate the ndobject metadata and data in one memory block
    char *data_ptr = NULL;
    memory_block_ptr result = make_ndobject_memory_block(0, size, scalar_dtype.get_alignment(), &data_ptr);

    // Fill in the preamble metadata
    ndobject_preamble *ndo = reinterpret_cast<ndobject_preamble *>(result.get());
    if (!scalar_dtype.is_builtin()) {
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
                        dt.get_data_size() + len, dt.get_alignment(), &data_ptr));
    // Set the string extents
    string_ptr = data_ptr + dt.get_data_size();
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
    preamble->m_dtype = new_dt.extended();
    if(!new_dt.is_builtin()) {
        extended_dtype_incref(preamble->m_dtype);
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

ndobject dynd::detail::make_from_vec<std::string>::make(const std::vector<std::string>& vec)
{
    // Constructor detail for making an ndobject from a vector of strings
    size_t total_string_size = 0;
    for (size_t i = 0, i_end = vec.size(); i != i_end; ++i) {
        total_string_size += vec[i].size();
    }

    dtype dt = make_strided_array_dtype(make_string_dtype(string_encoding_utf_8));
    char *data_ptr = NULL;
    // Make an ndobject memory block which contains both the string pointers and
    // the string data
    ndobject result(make_ndobject_memory_block(dt.extended()->get_metadata_size(),
                    sizeof(string_dtype_data) * vec.size() + total_string_size,
                    dt.get_alignment(), &data_ptr));
    char *string_ptr = data_ptr + sizeof(string_dtype_data) * vec.size();
    // The main ndobject metadata
    ndobject_preamble *preamble = result.get_ndo();
    preamble->m_data_pointer = data_ptr;
    preamble->m_data_reference = NULL;
    preamble->m_dtype = dt.extended();
    extended_dtype_incref(preamble->m_dtype);
    preamble->m_flags = read_access_flag | immutable_access_flag;
    // The metadata for the strided and string parts of the dtype
    strided_array_dtype_metadata *sa_md = reinterpret_cast<strided_array_dtype_metadata *>(
                                            result.get_ndo_meta());
    sa_md->size = vec.size();
    sa_md->stride = vec.empty() ? 0 : sizeof(string_dtype_data);
    string_dtype_metadata *s_md = reinterpret_cast<string_dtype_metadata *>(sa_md + 1);
    s_md->blockref = NULL;
    // The string pointers and data
    string_dtype_data *data = reinterpret_cast<string_dtype_data *>(data_ptr);
    for (size_t i = 0, i_end = vec.size(); i != i_end; ++i) {
        size_t size = vec[i].size();
        memcpy(string_ptr, vec[i].data(), size);
        data[i].begin = string_ptr;
        string_ptr += size;
        data[i].end = string_ptr;
    }
    return result;
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
    static void as_storage_type(const dtype& dt, const void *DYND_UNUSED(extra),
                dtype& out_transformed_dtype, bool& out_was_transformed)
    {
        // If the dtype is a simple POD, switch it to a bytes dtype. Otherwise, keep it
        // the same so that the metadata layout is identical.
        if (dt.is_scalar() && dt.get_type_id() != pointer_type_id) {
            const dtype& storage_dt = dt.storage_dtype();
            if (storage_dt.is_builtin() || (storage_dt.get_memory_management() == pod_memory_management &&
                                    storage_dt.extended()->get_metadata_size() == 0)) {
                out_transformed_dtype = make_fixedbytes_dtype(storage_dt.get_data_size(), storage_dt.get_alignment());
                out_was_transformed = true;
            } else if (storage_dt.get_type_id() == string_type_id) {
                out_transformed_dtype = make_bytes_dtype(static_cast<const string_dtype *>(storage_dt.extended())->get_data_alignment());
                out_was_transformed = true;
            } else {
                if (dt.get_kind() == expression_kind) {
                    out_transformed_dtype = storage_dt;
                    out_was_transformed = true;
                } else {
                    // No transformation
                    out_transformed_dtype = dt;
                }
            }
        } else {
            dt.extended()->transform_child_dtypes(&as_storage_type, NULL, out_transformed_dtype, out_was_transformed);
        }
    }
} // anonymous namespace

ndobject ndobject::storage() const
{
    dtype storage_dt = get_dtype();
    bool was_transformed;
    as_storage_type(get_dtype(), NULL, storage_dt, was_transformed);
    if (was_transformed) {
        return make_ndobject_clone_with_new_dtype(*this, storage_dt);
    } else {
        return *this;
    }
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
        if (!dt.is_builtin()) {
            result.set(make_ndobject_memory_block(dt.extended()->get_metadata_size()));
            result.get_ndo()->m_dtype = dt.extended();
            extended_dtype_incref(result.get_ndo()->m_dtype);
        } else {
            result.set(make_ndobject_memory_block(0));
            result.get_ndo()->m_dtype = reinterpret_cast<const extended_dtype *>(dt.get_type_id());
        }
        intptr_t offset = get_ndo()->m_dtype->apply_linear_index(nindices, indices, get_ndo()->m_data_pointer,
                        get_ndo_meta(), dt, result.get_ndo_meta(),
                        m_memblock.get(), 0, this_dt);
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
        kernel_instance<unary_operation_pair_t> assign;
        const char *src_ptr = rhs.get_ndo()->m_data_pointer;

        // TODO: Performance optimization
        ndobject_iter<1, 0> iter(*this);
        get_dtype_assignment_kernel(iter.get_uniform_dtype(), rhs.get_dtype(), errmode, ectx, assign);
        unary_kernel_static_data extra(assign.auxdata, iter.metadata(), rhs.get_ndo_meta());
        if (!iter.empty()) {
            do {
                assign.kernel.single(iter.data(), src_ptr, &extra);
            } while (iter.next());
        }
    } else {
        kernel_instance<unary_operation_pair_t> assign;

        // TODO: Performance optimization
        ndobject_iter<1, 1> iter(*this, rhs);
        get_dtype_assignment_kernel(iter.get_uniform_dtype<0>(), iter.get_uniform_dtype<1>(), errmode, ectx, assign);
        unary_kernel_static_data extra(assign.auxdata, iter.metadata<0>(), iter.metadata<1>());
        if (!iter.empty()) {
            do {
                assign.kernel.single(iter.data<0>(), iter.data<1>(), &extra);
            } while (iter.next());
        }
    }
}

void ndobject::val_assign(const dtype& rhs_dt, const char *rhs_metadata, const char *rhs_data,
                    assign_error_mode errmode, const eval::eval_context *ectx) const
{
    // Verify access permissions
    if (!(get_flags()&write_access_flag)) {
        throw runtime_error("tried to write to a dynd array that is not writeable");
    }

    kernel_instance<unary_operation_pair_t> assign;

    // TODO: Performance optimization
    ndobject_iter<1, 0> iter(*this);
    get_dtype_assignment_kernel(iter.get_uniform_dtype(), rhs_dt, errmode, ectx, assign);
    if (!iter.empty()) {
        unary_kernel_static_data extra(assign.auxdata, iter.metadata(), rhs_metadata);
        do {
            assign.kernel.single(iter.data(), rhs_data, &extra);
        } while (iter.next());
    }
}

ndobject ndobject::p(const char *property_name) const
{
    dtype dt = get_dtype();
    if (!dt.is_builtin()) {
        const std::pair<std::string, gfunc::callable> *properties;
        size_t count;
        dt.extended()->get_dynamic_ndobject_properties(&properties, &count);
        // TODO: We probably want to make some kind of acceleration structure for the name lookup
        if (count > 0) {
            for (int i = 0; i < count; ++i) {
                if (properties[i].first == property_name) {
                    return properties[i].second.call(*this);
                }
            }
        }
    }

    stringstream ss;
    ss << "dynd nobject does not have property " << property_name;
    throw runtime_error(ss.str());
}

ndobject ndobject::p(const std::string& property_name) const
{
    dtype dt = get_dtype();
    if (!dt.is_builtin()) {
        const std::pair<std::string, gfunc::callable> *properties;
        size_t count;
        dt.extended()->get_dynamic_ndobject_properties(&properties, &count);
        // TODO: We probably want to make some kind of acceleration structure for the name lookup
        if (count > 0) {
            for (int i = 0; i < count; ++i) {
                if (properties[i].first == property_name) {
                    return properties[i].second.call(*this);
                }
            }
        }
    }

    stringstream ss;
    ss << "dynd nobject does not have property " << property_name;
    throw runtime_error(ss.str());
}

const gfunc::callable& ndobject::f(const char *function_name) const
{
    dtype dt = get_dtype();
    if (!dt.is_builtin()) {
        const std::pair<std::string, gfunc::callable> *properties;
        size_t count;
        dt.extended()->get_dynamic_ndobject_functions(&properties, &count);
        // TODO: We probably want to make some kind of acceleration structure for the name lookup
        if (count > 0) {
            for (int i = 0; i < count; ++i) {
                if (properties[i].first == function_name) {
                    return properties[i].second;
                }
            }
        }
    }

    stringstream ss;
    ss << "dynd nobject does not have function " << function_name;
    throw runtime_error(ss.str());
}

ndobject ndobject::eval_immutable(const eval::eval_context *ectx) const
{
    if (get_access_flags()&immutable_access_flag) {
        return *this;
    } else {
        // Create a canonical dtype for the result
        const dtype& current_dtype = get_dtype();
        const dtype& dt = current_dtype.get_canonical_dtype();
        size_t ndim = current_dtype.get_undim();
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
    size_t ndim = current_dtype.get_undim();
    dimvector shape(ndim);
    get_shape(shape.get());
    ndobject result(make_ndobject_memory_block(dt, ndim, shape.get()));
    // TODO: Reorder strides of strided dimensions in a KEEPORDER fashion
    result.val_assign(*this, assign_error_default, ectx);
    result.get_ndo()->m_flags = access_flags;
    return result;
}

bool dynd::ndobject::equals_exact(const ndobject& rhs) const
{
    if (get_ndo() == rhs.get_ndo()) {
        return true;
    } else {
        throw runtime_error("ndarray::equals_exact is not yet implemented");
    }
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
    struct switch_udtype_extra {
        switch_udtype_extra(const dtype& dt, assign_error_mode em)
            : replacement_dtype(dt), errmode(em)
        {
        }
        const dtype& replacement_dtype;
        assign_error_mode errmode;
    };
    static void switch_udtype(const dtype& dt, const void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed)
    {
        // If things aren't simple, use a view_dtype
        if (!dt.is_builtin() && dt.extended()->is_uniform_dim()) {
            dt.extended()->transform_child_dtypes(&switch_udtype, extra, out_transformed_dtype, out_was_transformed);
        } else {
            const switch_udtype_extra *e = reinterpret_cast<const switch_udtype_extra *>(extra);
            out_transformed_dtype = make_convert_dtype(e->replacement_dtype, dt, e->errmode);
            out_was_transformed= true;
        }
    }
} // anonymous namespace

ndobject ndobject::cast_udtype(const dtype& scalar_dtype, assign_error_mode errmode) const
{
    // This creates a dtype which has a convert dtype for every scalar of different dtype.
    // The result has the exact same metadata and data, so we just have to swap in the new
    // dtype in a shallow copy.
    dtype replaced_dtype;
    bool was_transformed;
    switch_udtype_extra extra(scalar_dtype, errmode);
    switch_udtype(get_dtype(), &extra, replaced_dtype, was_transformed);
    if (was_transformed) {
        return make_ndobject_clone_with_new_dtype(*this, replaced_dtype);
    } else {
        return *this;
    }
}

namespace {
    static void view_scalar_types(const dtype& dt, const void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed)
    {
        if (dt.is_scalar()) {
            const dtype *e = reinterpret_cast<const dtype *>(extra);
            // If things aren't simple, use a view_dtype
            if (dt.get_kind() == expression_kind || dt.get_data_size() != e->get_data_size() ||
                        dt.get_memory_management() != pod_memory_management ||
                        e->get_memory_management() != pod_memory_management) {
                out_transformed_dtype = make_view_dtype(*e, dt);
                out_was_transformed = true;
            } else {
                out_transformed_dtype = *e;
                if (dt != *e) {
                    out_was_transformed = true;
                }
            }
        } else {
            dt.extended()->transform_child_dtypes(&view_scalar_types, extra, out_transformed_dtype, out_was_transformed);
        }
    }
} // anonymous namespace

ndobject ndobject::view_scalars(const dtype& scalar_dtype) const
{
    const dtype& array_dtype = get_dtype();
    size_t uniform_ndim = array_dtype.get_undim();
    // First check if we're dealing with a simple one dimensional block of memory we can reinterpret
    // at will.
    if (uniform_ndim == 1 && array_dtype.get_type_id() == strided_array_type_id) {
        const strided_array_dtype *sad = static_cast<const strided_array_dtype *>(array_dtype.extended());
        const strided_array_dtype_metadata *md = reinterpret_cast<const strided_array_dtype_metadata *>(get_ndo_meta());
        size_t element_size = sad->get_element_dtype().get_data_size();
        if (element_size != 0 && (intptr_t)element_size == md->stride &&
                    sad->get_element_dtype().get_kind() != expression_kind &&
                    sad->get_element_dtype().get_memory_management() == pod_memory_management) {
            intptr_t nbytes = md->size * element_size;
            // Make sure the element size divides into the # of bytes
            if (nbytes % scalar_dtype.get_data_size() != 0) {
                std::stringstream ss;
                ss << "cannot view ndobject with " << nbytes << " bytes as dtype ";
                ss << scalar_dtype << ", because its element size " << scalar_dtype.get_data_size();
                ss << " doesn't divide evenly into the total array size " << nbytes;
                throw std::runtime_error(ss.str());
            }
            // Create the result array, adjusting the dtype if the data isn't aligned correctly
            char *data_ptr = get_ndo()->m_data_pointer;
            dtype result_dtype;
            if ((((uintptr_t)data_ptr)&(scalar_dtype.get_alignment()-1)) == 0) {
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
            result_md->size = nbytes / scalar_dtype.get_data_size();
            result_md->stride = scalar_dtype.get_data_size();
            return result;
        }
    }

    // Transform the scalars into view dtypes
    dtype viewed_dtype;
    bool was_transformed;
    view_scalar_types(get_dtype(), &scalar_dtype, viewed_dtype, was_transformed);
    return make_ndobject_clone_with_new_dtype(*this, viewed_dtype);
}

std::string dynd::detail::ndobject_as_string(const ndobject& lhs, assign_error_mode DYND_UNUSED(errmode))
{
    if (!lhs.is_scalar()) {
        throw std::runtime_error("can only convert ndobjects with 0 dimensions to scalars");
    }

    ndobject temp = lhs;
    if (temp.get_dtype().get_kind() != string_kind) {
        temp = temp.cast_scalars(make_string_dtype(string_encoding_utf_8)).vals();
    }
    const extended_string_dtype *esd = static_cast<const extended_string_dtype *>(temp.get_dtype().extended());
    return esd->get_utf8_string(temp.get_ndo_meta(), temp.get_ndo()->m_data_pointer, assign_error_none);
}

void ndobject::debug_print(std::ostream& o, const std::string& indent) const
{
    o << indent << "------ ndobject\n";
    if (m_memblock.get()) {
        const ndobject_preamble *ndo = get_ndo();
        o << " address: " << (void *)m_memblock.get() << "\n";
        o << " refcount: " << ndo->m_memblockdata.m_use_count << "\n";
        o << " dtype:\n";
        o << "  pointer: " << (void *)ndo->m_dtype << "\n";
        o << "  type: " << get_dtype() << "\n";
        o << " metadata:\n";
        o << "  flags: " << ndo->m_flags << " (";
        if (ndo->m_flags & read_access_flag) o << "read_access ";
        if (ndo->m_flags & write_access_flag) o << "write_access ";
        if (ndo->m_flags & immutable_access_flag) o << "immutable ";
        o << ")\n";
        if (!ndo->is_builtin_dtype()) {
            o << "  dtype-specific metadata:\n";
            ndo->m_dtype->metadata_debug_print(get_ndo_meta(), o, indent + "   ");
        }
        o << " data:\n";
        o << "   pointer: " << (void *)ndo->m_data_pointer << "\n";
        o << "   reference: " << (void *)ndo->m_data_reference;
        if (ndo->m_data_reference == NULL) {
            o << " (embedded in ndobject memory)\n";
        } else {
            o << "\n";
        }
        if (ndo->m_data_reference != NULL) {
            memory_block_debug_print(ndo->m_data_reference, o, "    ");
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
            v.get_ndo()->m_dtype->print_data(o, v.get_ndo_meta(), v.get_ndo()->m_data_pointer);
        }
        o << ", " << rhs.get_dtype() << ")";
    } else {
        o << "ndobject()";
    }
    return o;
}

ndobject dynd::empty_like(const ndobject& rhs, const dtype& uniform_dtype)
{
    if (rhs.is_scalar()) {
        return ndobject(uniform_dtype);
    } else {
        dtype dt = rhs.get_ndo()->m_dtype->get_canonical_dtype();
        size_t ndim = dt.extended()->get_undim();
        dt = make_strided_array_dtype(uniform_dtype, ndim);
        dimvector shape(ndim);
        rhs.get_shape(shape.get());
        ndobject result(make_ndobject_memory_block(dt, ndim, shape.get()));
        // Reorder strides of output strided dimensions in a KEEPORDER fashion
        dt.extended()->reorder_default_constructed_strides(result.get_ndo_meta(),
                        rhs.get_dtype(), rhs.get_ndo_meta());
        return result;
    }
}

ndobject dynd::empty_like(const ndobject& rhs)
{
    dtype dt;
    if (rhs.get_ndo()->is_builtin_dtype()) {
        dt = dtype(rhs.get_ndo()->get_builtin_type_id());
    } else {
        dt = rhs.get_ndo()->m_dtype->get_canonical_dtype();
    }

    if (rhs.is_scalar()) {
        return ndobject(dt);
    } else {
        size_t ndim = dt.extended()->get_undim();
        dimvector shape(ndim);
        rhs.get_shape(shape.get());
        ndobject result(make_ndobject_memory_block(dt, ndim, shape.get()));
        // Reorder strides of output strided dimensions in a KEEPORDER fashion
        dt.extended()->reorder_default_constructed_strides(result.get_ndo_meta(),
                        rhs.get_dtype(), rhs.get_ndo_meta());
        return result;
    }
}

ndobject_vals::operator ndobject() const
{
    const dtype& current_dtype = m_arr.get_dtype();

    if (!current_dtype.is_expression()) {
        return m_arr;
    } else {
        // If there is any expression in the dtype, make a copy using the canonical dtype
        const dtype& dt = current_dtype.get_canonical_dtype();
        size_t ndim = current_dtype.get_undim();
        dimvector shape(ndim);
        m_arr.get_shape(shape.get());
        ndobject result(make_ndobject_memory_block(dt, ndim, shape.get()));
        if (!dt.is_builtin()) {
            // Reorder strides of output strided dimensions in a KEEPORDER fashion
            dt.extended()->reorder_default_constructed_strides(result.get_ndo_meta(),
                            m_arr.get_dtype(), m_arr.get_ndo_meta());
                            
        }
        result.val_assign(m_arr);
        return result;
    }
}
