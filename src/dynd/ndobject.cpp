//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/ndobject.hpp>
#include <dynd/ndobject_iter.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/var_dim_dtype.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/bytes_dtype.hpp>
#include <dynd/dtypes/fixedbytes_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/comparison_kernels.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/gfunc/callable.hpp>
#include <dynd/gfunc/call_callable.hpp>
#include <dynd/dtypes/groupby_dtype.hpp>
#include <dynd/dtypes/categorical_dtype.hpp>
#include <dynd/dtypes/builtin_dtype_properties.hpp>

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
    ndo->m_dtype = reinterpret_cast<base_dtype *>(type_id_of<T>::value);
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = read_access_flag | immutable_access_flag;
    return result;
}

ndobject dynd::make_strided_ndobject(const dtype& uniform_dtype, size_t ndim, const intptr_t *shape,
                int64_t access_flags, const int *axis_perm)
{
    // Create the dtype of the result
    dtype array_dtype = uniform_dtype;
    bool any_variable_dims = false;
    for (intptr_t i = ndim-1; i >= 0; --i) {
        if (shape[i] >= 0) {
            array_dtype = make_strided_dim_dtype(array_dtype);
        } else {
            array_dtype = make_var_dim_dtype(array_dtype);
            any_variable_dims = true;
        }
    }

    // Determine the total data size
    size_t data_size;
    if (array_dtype.is_builtin()) {
        data_size = array_dtype.get_data_size();
    } else {
        data_size = array_dtype.extended()->get_default_data_size(ndim, shape);
    }

    // Allocate the ndobject metadata and data in one memory block
    char *data_ptr = NULL;
    memory_block_ptr result = make_ndobject_memory_block(array_dtype.extended()->get_metadata_size(),
                    data_size, uniform_dtype.get_alignment(), &data_ptr);

    if (array_dtype.get_flags()&dtype_flag_zeroinit) {
        memset(data_ptr, 0, data_size);
    }

    // Fill in the preamble metadata
    ndobject_preamble *ndo = reinterpret_cast<ndobject_preamble *>(result.get());
    ndo->m_dtype = array_dtype.release();
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = access_flags;

    if (!any_variable_dims) {
        // Fill in the ndobject metadata with strides and sizes
        strided_dim_dtype_metadata *meta = reinterpret_cast<strided_dim_dtype_metadata *>(ndo + 1);
        // Use the default construction to handle the uniform_dtype's metadata
        intptr_t stride = uniform_dtype.get_data_size();
        if (stride == 0) {
            stride = uniform_dtype.extended()->get_default_data_size(0, NULL);
        }
        if (!uniform_dtype.is_builtin()) {
            uniform_dtype.extended()->metadata_default_construct(
                            reinterpret_cast<char *>(meta + ndim), 0, NULL);
        }
        if (axis_perm == NULL) {
            for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
                intptr_t dim_size = shape[i];
                meta[i].stride = dim_size > 1 ? stride : 0;
                meta[i].size = dim_size;
                stride *= dim_size;
            }
        } else {
            for (size_t i = 0; i < ndim; ++i) {
                int i_perm = axis_perm[i];
                intptr_t dim_size = shape[i_perm];
                meta[i_perm].stride = dim_size > 1 ? stride : 0;
                meta[i_perm].size = dim_size;
                stride *= dim_size;
            }
        }
    } else {
        if (axis_perm != NULL) {
            // Maybe force C-order in this case?
            throw runtime_error("dynd presently only supports C-order with variable-sized arrays");
        }
        // Fill in the ndobject metadata with strides and sizes
        char *meta = reinterpret_cast<char *>(ndo + 1);
        ndo->m_dtype->metadata_default_construct(meta, ndim, shape);
    }

    return ndobject(result);
}

ndobject dynd::make_strided_ndobject_from_data(const dtype& uniform_dtype, size_t ndim, const intptr_t *shape,
                const intptr_t *strides, int64_t access_flags, char *data_ptr,
                const memory_block_ptr& data_reference, char **out_uniform_metadata)
{
    if (out_uniform_metadata == NULL && !uniform_dtype.is_builtin() && uniform_dtype.extended()->get_metadata_size() > 0) {
        stringstream ss;
        ss << "Cannot make a strided ndobject with dtype " << uniform_dtype << " from a preexisting data pointer";
        throw runtime_error(ss.str());
    }

    dtype array_dtype = make_strided_dim_dtype(uniform_dtype, ndim);

    // Allocate the ndobject metadata and data in one memory block
    memory_block_ptr result = make_ndobject_memory_block(array_dtype.extended()->get_metadata_size());

    // Fill in the preamble metadata
    ndobject_preamble *ndo = reinterpret_cast<ndobject_preamble *>(result.get());
    ndo->m_dtype = array_dtype.release();
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = data_reference.get();
    memory_block_incref(ndo->m_data_reference);
    ndo->m_flags = access_flags;

    // Fill in the ndobject metadata with the shape and strides
    strided_dim_dtype_metadata *meta = reinterpret_cast<strided_dim_dtype_metadata *>(ndo + 1);
    for (size_t i = 0; i < ndim; ++i) {
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

ndobject dynd::make_pod_ndobject(const dtype& pod_dt, const void *data)
{
    size_t size = pod_dt.get_data_size();
    if (!pod_dt.is_pod()) {
        stringstream ss;
        ss << "Cannot make a dynd ndobject from raw data using non-POD dtype " << pod_dt;
        throw runtime_error(ss.str());
    } else if (pod_dt.extended()->get_metadata_size() != 0) {
        stringstream ss;
        ss << "Cannot make a dynd ndobject from raw data using dtype " << pod_dt;
        ss << " because it has non-empty dynd metadata";
        throw runtime_error(ss.str());
    }

    // Allocate the ndobject metadata and data in one memory block
    char *data_ptr = NULL;
    memory_block_ptr result = make_ndobject_memory_block(0, size, pod_dt.get_alignment(), &data_ptr);

    // Fill in the preamble metadata
    ndobject_preamble *ndo = reinterpret_cast<ndobject_preamble *>(result.get());
    if (pod_dt.is_builtin()) {
        ndo->m_dtype = reinterpret_cast<const base_dtype *>(pod_dt.get_type_id());
    } else {
        ndo->m_dtype = pod_dt.extended();
        base_dtype_incref(ndo->m_dtype);
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
    ndo->m_dtype = dt.release();
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
    const string_dtype_metadata *md = reinterpret_cast<const string_dtype_metadata *>(result.get_ndo_meta() + sizeof(strided_dim_dtype_metadata));
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
        base_dtype_decref(preamble->m_dtype);
    }
    preamble->m_dtype = new_dt.extended();
    if(!new_dt.is_builtin()) {
        base_dtype_incref(preamble->m_dtype);
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

    dtype dt = make_strided_dim_dtype(make_string_dtype(string_encoding_utf_8));
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
    preamble->m_dtype = dt.release();
    preamble->m_flags = read_access_flag | immutable_access_flag;
    // The metadata for the strided and string parts of the dtype
    strided_dim_dtype_metadata *sa_md = reinterpret_cast<strided_dim_dtype_metadata *>(
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
            if (storage_dt.is_builtin()) {
                out_transformed_dtype = make_fixedbytes_dtype(storage_dt.get_data_size(),
                                storage_dt.get_alignment());
                out_was_transformed = true;
            } else if (storage_dt.is_pod() && storage_dt.extended()->get_metadata_size() == 0) {
                out_transformed_dtype = make_fixedbytes_dtype(storage_dt.get_data_size(),
                                storage_dt.get_alignment());
                out_was_transformed = true;
            } else if (storage_dt.get_type_id() == string_type_id) {
                out_transformed_dtype = make_bytes_dtype(static_cast<const string_dtype *>(
                                storage_dt.extended())->get_data_alignment());
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
    bool was_transformed = false;
    as_storage_type(get_dtype(), NULL, storage_dt, was_transformed);
    if (was_transformed) {
        return make_ndobject_clone_with_new_dtype(*this, storage_dt);
    } else {
        return *this;
    }
}

ndobject ndobject::at_array(size_t nindices, const irange *indices, bool collapse_leading) const
{
    if (is_scalar()) {
        if (nindices != 0) {
            throw too_many_indices(get_dtype(), nindices, 0);
        }
        return *this;
    } else {
        dtype this_dt(get_ndo()->m_dtype, true);
        dtype dt = get_ndo()->m_dtype->apply_linear_index(nindices, indices,
                        0, this_dt, collapse_leading);
        ndobject result;
        if (!dt.is_builtin()) {
            result.set(make_ndobject_memory_block(dt.extended()->get_metadata_size()));
            result.get_ndo()->m_dtype = dt.extended();
            base_dtype_incref(result.get_ndo()->m_dtype);
        } else {
            result.set(make_ndobject_memory_block(0));
            result.get_ndo()->m_dtype = reinterpret_cast<const base_dtype *>(dt.get_type_id());
        }
        result.get_ndo()->m_data_pointer = get_ndo()->m_data_pointer;
        if (get_ndo()->m_data_reference) {
            result.get_ndo()->m_data_reference = get_ndo()->m_data_reference;
        } else {
            // If the data reference is NULL, the data is embedded in the ndobject itself
            result.get_ndo()->m_data_reference = m_memblock.get();
        }
        memory_block_incref(result.get_ndo()->m_data_reference);
        intptr_t offset = get_ndo()->m_dtype->apply_linear_index(nindices, indices,
                        get_ndo_meta(), dt, result.get_ndo_meta(),
                        m_memblock.get(), 0, this_dt,
                        collapse_leading,
                        &result.get_ndo()->m_data_pointer, &result.get_ndo()->m_data_reference);
        result.get_ndo()->m_data_pointer += offset;
        result.get_ndo()->m_flags = get_ndo()->m_flags;
        return result;
    }
}

void ndobject::val_assign(const ndobject& rhs, assign_error_mode errmode,
                    const eval::eval_context *ectx) const
{
    // Verify read access permission
    if (!(rhs.get_flags()&read_access_flag)) {
        throw runtime_error("tried to read from a dynd array that is not readable");
    }

    dtype_assign(get_dtype(), get_ndo_meta(), get_readwrite_originptr(),
                    rhs.get_dtype(), rhs.get_ndo_meta(), rhs.get_readonly_originptr(),
                    errmode, ectx);
}

void ndobject::val_assign(const dtype& rhs_dt, const char *rhs_metadata, const char *rhs_data,
                    assign_error_mode errmode, const eval::eval_context *ectx) const
{
    dtype_assign(get_dtype(), get_ndo_meta(), get_readwrite_originptr(),
                    rhs_dt, rhs_metadata, rhs_data,
                    errmode, ectx);
}

void ndobject::flag_as_immutable()
{
    // If it's already immutable, everything's ok
    if ((get_flags()&immutable_access_flag) != 0) {
        return;
    }

    // Check that nobody else is peeking into our data
    bool ok = true;
    if (m_memblock.get()->m_use_count != 1) {
        // More than one reference to the ndobject itself
        ok = false;
    } else if (get_ndo()->m_data_reference != NULL &&
            (get_ndo()->m_data_reference->m_use_count != 1 ||
             !(get_ndo()->m_data_reference->m_type == fixed_size_pod_memory_block_type ||
               get_ndo()->m_data_reference->m_type == pod_memory_block_type))) {
        // More than one reference to the ndobject's data, or the reference is to something
        // other than a memblock owning its data, such as an external memblock.
        ok = false;
    } else if (!get_ndo()->is_builtin_dtype() &&
            !get_ndo()->m_dtype->is_unique_data_owner(get_ndo_meta())) {
        ok = false;
    }

    if (ok) {
        // Finalize any allocated data in the metadata
        if (!is_builtin_dtype(get_ndo()->m_dtype)) {
            get_ndo()->m_dtype->metadata_finalize_buffers(get_ndo_meta());
        }
        // Clear the write flag, and set the immutable flag
        get_ndo()->m_flags = (get_ndo()->m_flags&~(uint64_t)write_access_flag)|immutable_access_flag;
    } else {
        stringstream ss;
        ss << "Unable to flag ndobject of dtype " << get_dtype() << " as immutable, because ";
        ss << "it does not uniquely own all of its data";
        throw runtime_error(ss.str());
    }
}

ndobject ndobject::p(const char *property_name) const
{
    dtype dt = get_dtype();
    const std::pair<std::string, gfunc::callable> *properties;
    size_t count;
    if (!dt.is_builtin()) {
        dt.extended()->get_dynamic_ndobject_properties(&properties, &count);
    } else {
        get_builtin_dtype_dynamic_ndobject_properties(dt.get_type_id(), &properties, &count);
    }
    // TODO: We probably want to make some kind of acceleration structure for the name lookup
    if (count > 0) {
        for (size_t i = 0; i < count; ++i) {
            if (properties[i].first == property_name) {
                return properties[i].second.call(*this);
            }
        }
    }

    stringstream ss;
    ss << "dynd ndobject does not have property " << property_name;
    throw runtime_error(ss.str());
}

ndobject ndobject::p(const std::string& property_name) const
{
    dtype dt = get_dtype();
    const std::pair<std::string, gfunc::callable> *properties;
    size_t count;
    if (!dt.is_builtin()) {
        dt.extended()->get_dynamic_ndobject_properties(&properties, &count);
    } else {
        get_builtin_dtype_dynamic_ndobject_properties(dt.get_type_id(), &properties, &count);
    }
    // TODO: We probably want to make some kind of acceleration structure for the name lookup
    if (count > 0) {
        for (size_t i = 0; i < count; ++i) {
            if (properties[i].first == property_name) {
                return properties[i].second.call(*this);
            }
        }
    }

    stringstream ss;
    ss << "dynd ndobject does not have property " << property_name;
    throw runtime_error(ss.str());
}

const gfunc::callable& ndobject::find_dynamic_function(const char *function_name) const
{
    dtype dt = get_dtype();
    if (!dt.is_builtin()) {
        const std::pair<std::string, gfunc::callable> *properties;
        size_t count;
        dt.extended()->get_dynamic_ndobject_functions(&properties, &count);
        // TODO: We probably want to make some kind of acceleration structure for the name lookup
        if (count > 0) {
            for (size_t i = 0; i < count; ++i) {
                if (properties[i].first == function_name) {
                    return properties[i].second;
                }
            }
        }
    }

    stringstream ss;
    ss << "dynd ndobject does not have function " << function_name;
    throw runtime_error(ss.str());
}

ndobject ndobject::eval(const eval::eval_context *ectx) const
{
    const dtype& current_dtype = get_dtype();
    if (!current_dtype.is_expression()) {
        return *this;
    } else {
        // Create a canonical dtype for the result
        const dtype& dt = current_dtype.get_canonical_dtype();
        size_t ndim = current_dtype.get_undim();
        dimvector shape(ndim);
        get_shape(shape.get());
        ndobject result(make_ndobject_memory_block(dt, ndim, shape.get()));
        if (dt.get_undim() > 1) {
            // Reorder strides of output strided dimensions in a KEEPORDER fashion
            dt.extended()->reorder_default_constructed_strides(result.get_ndo_meta(),
                            get_dtype(), get_ndo_meta());
        }
        result.val_assign(*this, assign_error_default, ectx);
        return result;
    }
}

ndobject ndobject::eval_immutable(const eval::eval_context *ectx) const
{
    const dtype& current_dtype = get_dtype();
    if ((get_access_flags()&immutable_access_flag) &&
                    !current_dtype.is_expression()) {
        return *this;
    } else {
        // Create a canonical dtype for the result
        const dtype& dt = current_dtype.get_canonical_dtype();
        size_t ndim = current_dtype.get_undim();
        dimvector shape(ndim);
        get_shape(shape.get());
        ndobject result(make_ndobject_memory_block(dt, ndim, shape.get()));
        if (dt.get_undim() > 1) {
            // Reorder strides of output strided dimensions in a KEEPORDER fashion
            dt.extended()->reorder_default_constructed_strides(result.get_ndo_meta(),
                            get_dtype(), get_ndo_meta());
        }
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
    if (dt.get_undim() > 1) {
        // Reorder strides of output strided dimensions in a KEEPORDER fashion
        dt.extended()->reorder_default_constructed_strides(result.get_ndo_meta(),
                        get_dtype(), get_ndo_meta());
    }
    result.val_assign(*this, assign_error_default, ectx);
    result.get_ndo()->m_flags = access_flags;
    return result;
}

bool ndobject::op_sorting_less(const ndobject& rhs) const
{
    comparison_kernel k;
    make_comparison_kernel(&k, 0, get_dtype(), get_ndo_meta(),
                    rhs.get_dtype(), rhs.get_ndo_meta(),
                    comparison_type_sorting_less,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool ndobject::operator<(const ndobject& rhs) const
{
    comparison_kernel k;
    make_comparison_kernel(&k, 0, get_dtype(), get_ndo_meta(),
                    rhs.get_dtype(), rhs.get_ndo_meta(),
                    comparison_type_less,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool ndobject::operator<=(const ndobject& rhs) const
{
    comparison_kernel k;
    make_comparison_kernel(&k, 0, get_dtype(), get_ndo_meta(),
                    rhs.get_dtype(), rhs.get_ndo_meta(),
                    comparison_type_less_equal,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool ndobject::operator==(const ndobject& rhs) const
{
    comparison_kernel k;
    make_comparison_kernel(&k, 0, get_dtype(), get_ndo_meta(),
                    rhs.get_dtype(), rhs.get_ndo_meta(),
                    comparison_type_equal,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool ndobject::operator!=(const ndobject& rhs) const
{
    comparison_kernel k;
    make_comparison_kernel(&k, 0, get_dtype(), get_ndo_meta(),
                    rhs.get_dtype(), rhs.get_ndo_meta(),
                    comparison_type_not_equal,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool ndobject::operator>=(const ndobject& rhs) const
{
    comparison_kernel k;
    make_comparison_kernel(&k, 0, get_dtype(), get_ndo_meta(),
                    rhs.get_dtype(), rhs.get_ndo_meta(),
                    comparison_type_greater_equal,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool ndobject::operator>(const ndobject& rhs) const
{
    comparison_kernel k;
    make_comparison_kernel(&k, 0, get_dtype(), get_ndo_meta(),
                    rhs.get_dtype(), rhs.get_ndo_meta(),
                    comparison_type_greater,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool ndobject::equals_exact(const ndobject& rhs) const
{
    if (get_ndo() == rhs.get_ndo()) {
        return true;
    } else if (get_dtype() != rhs.get_dtype()) {
        return false;
    } else if (get_undim() == 0) {
        comparison_kernel k;
        make_comparison_kernel(&k, 0,
                        get_dtype(), get_ndo_meta(),
                        rhs.get_dtype(), rhs.get_ndo_meta(),
                        comparison_type_equal, &eval::default_eval_context);
        return k(get_readonly_originptr(), rhs.get_readonly_originptr());
    } else {
        // First compare the shape, to avoid triggering an exception in common cases
        size_t undim = get_undim();
        dimvector shape0(undim), shape1(undim);
        get_shape(shape0.get());
        rhs.get_shape(shape1.get());
        if (memcmp(shape0.get(), shape1.get(), undim * sizeof(intptr_t)) != 0) {
            return false;
        }
        try {
            ndobject_iter<0,2> iter(*this, rhs);
            if (!iter.empty()) {
                comparison_kernel k;
                make_comparison_kernel(&k, 0,
                                iter.get_uniform_dtype<0>(), iter.metadata<0>(),
                                iter.get_uniform_dtype<1>(), iter.metadata<1>(),
                                comparison_type_not_equal, &eval::default_eval_context);
                do {
                    if (k(iter.data<0>(), iter.data<1>())) {
                        return false;
                    }
                } while (iter.next());
            }
            return true;
        } catch(const broadcast_error&) {
            // If there's a broadcast error in a variable-sized dimension, return false for it too
            return false;
        }
    }
}

ndobject ndobject::cast_scalars(const dtype& scalar_dtype, assign_error_mode errmode) const
{
    // This creates a dtype which has a convert dtype for every scalar of different dtype.
    // The result has the exact same metadata and data, so we just have to swap in the new
    // dtype in a shallow copy.
    dtype replaced_dtype = get_dtype().with_replaced_scalar_types(scalar_dtype, errmode);
    if (replaced_dtype == get_dtype()) {
        return *this;
    } else {
        return make_ndobject_clone_with_new_dtype(*this, replaced_dtype);
    }
}

namespace {
    struct convert_udtype_extra {
        convert_udtype_extra(const dtype& dt, assign_error_mode em)
            : replacement_dtype(dt), errmode(em)
        {
        }
        const dtype& replacement_dtype;
        assign_error_mode errmode;
    };
    static void convert_udtype(const dtype& dt, const void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed)
    {
        if (!dt.is_builtin() && dt.extended()->is_uniform_dim()) {
            dt.extended()->transform_child_dtypes(&convert_udtype, extra, out_transformed_dtype, out_was_transformed);
        } else {
            const convert_udtype_extra *e = reinterpret_cast<const convert_udtype_extra *>(extra);
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
    convert_udtype_extra extra(scalar_dtype, errmode);
    convert_udtype(get_dtype(), &extra, replaced_dtype, was_transformed);
    if (was_transformed) {
        return make_ndobject_clone_with_new_dtype(*this, replaced_dtype);
    } else {
        return *this;
    }
}

namespace {
    static void replace_compatible_udtype(const dtype& dt, const void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed)
    {
        if (!dt.is_builtin() && dt.extended()->is_uniform_dim()) {
            dt.extended()->transform_child_dtypes(&replace_compatible_udtype, extra, out_transformed_dtype, out_was_transformed);
        } else {
            const dtype *e = reinterpret_cast<const dtype *>(extra);
            if (dt != *e) {
                if (!dt.data_layout_compatible_with(*e)) {
                    stringstream ss;
                    ss << "The dynd dtype " << dt << " is not ";
                    ss << " data layout compatible with " << *e;
                    ss << ", so a substitution cannot be made.";
                    throw runtime_error(ss.str());
                }
                out_transformed_dtype = *e;
                out_was_transformed= true;
            }
        }
    }
} // anonymous namespace

ndobject ndobject::replace_udtype(const dtype& new_udtype) const
{
    // This creates a dtype which swaps in the new udtype for
    // the existing one. It raises an error if the data layout
    // is incompatible
    dtype replaced_dtype;
    bool was_transformed = false;
    replace_compatible_udtype(get_dtype(), &new_udtype, replaced_dtype, was_transformed);
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
                        !dt.is_pod() || !e->is_pod()) {
                // Some special cases that have the same memory layouts
                switch (dt.get_type_id()) {
                    case string_type_id:
                    case json_type_id:
                    case bytes_type_id:
                        switch (e->get_type_id()) {
                            case string_type_id:
                            case json_type_id:
                            case bytes_type_id:
                                // All these dtypes have the same data/metadata layout,
                                // allow a view whenever the alignment allows it
                                if (e->get_alignment() <= dt.get_alignment()) {
                                    out_transformed_dtype = *e;
                                    out_was_transformed = true;
                                    return;
                                }
                                break;
                            default:
                                break;
                        }
                        break;
                    default:
                        break;
                }
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
    if (uniform_ndim == 1 && array_dtype.get_type_id() == strided_dim_type_id) {
        const strided_dim_dtype *sad = static_cast<const strided_dim_dtype *>(array_dtype.extended());
        const strided_dim_dtype_metadata *md = reinterpret_cast<const strided_dim_dtype_metadata *>(get_ndo_meta());
        const dtype& edt = sad->get_element_dtype();
        if (edt.is_pod() && (intptr_t)edt.get_data_size() == md->stride &&
                    sad->get_element_dtype().get_kind() != expression_kind) {
            intptr_t nbytes = md->size * edt.get_data_size();
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
                result_dtype = make_strided_dim_dtype(scalar_dtype);
            } else {
                result_dtype = make_strided_dim_dtype(make_unaligned_dtype(scalar_dtype));
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
            result.get_ndo()->m_dtype = result_dtype.release();
            result.get_ndo()->m_flags = get_ndo()->m_flags;
            // The result has one strided ndarray field
            strided_dim_dtype_metadata *result_md = reinterpret_cast<strided_dim_dtype_metadata *>(result.get_ndo_meta());
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
        temp = temp.cast_scalars(make_string_dtype(string_encoding_utf_8)).eval();
    }
    const base_string_dtype *esd = static_cast<const base_string_dtype *>(temp.get_dtype().extended());
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
        ndobject v = rhs.eval();
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

ndobject dynd::eval_raw_copy(const dtype& dt, const char *metadata, const char *data)
{
    // Allocate an output array with the canonical version of the dtype
    dtype cdt = dt.get_canonical_dtype();
    size_t undim = dt.get_undim();
    ndobject result;
    if (undim > 0) {
        dimvector shape(undim);
        dt.extended()->get_shape(0, shape.get(), metadata);
        result.set(make_ndobject_memory_block(cdt, undim, shape.get()));
        // Reorder strides of output strided dimensions in a KEEPORDER fashion
        cdt.extended()->reorder_default_constructed_strides(result.get_ndo_meta(),
                        dt, metadata);
    } else {
        result.set(make_ndobject_memory_block(cdt, 0, NULL));
    }

    dtype_assign(cdt, result.get_ndo_meta(), result.get_readwrite_originptr(),
                    dt, metadata, data,
                    assign_error_default, &eval::default_eval_context);

    return result;
}

ndobject dynd::empty_like(const ndobject& rhs, const dtype& uniform_dtype)
{
    if (rhs.is_scalar()) {
        return ndobject(uniform_dtype);
    } else {
        dtype dt = rhs.get_ndo()->m_dtype->get_canonical_dtype();
        size_t ndim = dt.extended()->get_undim();
        dt = make_strided_dim_dtype(uniform_dtype, ndim);
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

intptr_t dynd::binary_search(const ndobject& n, const char *metadata, const char *data)
{
    if (n.get_undim() == 0) {
        stringstream ss;
        ss << "cannot do a dynd binary_search on ndobject with dtype " << n.get_dtype() << " without a leading uniform dimension";
        throw runtime_error(ss.str());
    }
    const char *n_metadata = n.get_ndo_meta();
    dtype element_dtype = n.get_dtype().at_single(0, &n_metadata);
    if (element_dtype.get_metadata_size() == 0 || n_metadata == metadata ||
                    memcmp(n_metadata, metadata, element_dtype.get_metadata_size()) == 0) {
        // First, a version where the metadata is identical, so we can
        // make do with only a single comparison kernel
        comparison_kernel k_n_less_d;
        make_comparison_kernel(&k_n_less_d, 0,
                        element_dtype, n_metadata,
                        element_dtype, n_metadata,
                        comparison_type_sorting_less,
                        &eval::default_eval_context);

        // TODO: support any type of uniform dimension
        if (n.get_dtype().get_type_id() != strided_dim_type_id) {
            stringstream ss;
            ss << "TODO: binary_search on ndobject with dtype " << n.get_dtype() << " is not implemented";
            throw runtime_error(ss.str());
        }

        const char *n_data = n.get_readonly_originptr();
        intptr_t n_stride = reinterpret_cast<const strided_dim_dtype_metadata *>(n.get_ndo_meta())->stride;
        intptr_t first = 0, last = n.get_dim_size();
        while (first < last) {
            intptr_t trial = first + (last - first) / 2;
            const char *trial_data = n_data + trial * n_stride;

            // In order for the data to always match up with the metadata, need to have
            // trial_data first and data second in the comparison operations.
            if (k_n_less_d(data, trial_data)) {
                // value < arr[trial]
                last = trial;
            } else if (k_n_less_d(trial_data, data)) {
                // value > arr[trial]
                first = trial + 1;
            } else {
                return trial;
            }
        }
        return -1;
    } else {
        // Second, a version where the metadata are different, so
        // we need to get a kernel for each comparison direction.
        comparison_kernel k_n_less_d, k_d_less_n;
        make_comparison_kernel(&k_n_less_d, 0,
                        element_dtype, n_metadata,
                        element_dtype, metadata,
                        comparison_type_sorting_less,
                        &eval::default_eval_context);
        make_comparison_kernel(&k_d_less_n, 0,
                        element_dtype, metadata,
                        element_dtype, n_metadata,
                        comparison_type_sorting_less,
                        &eval::default_eval_context);

        // TODO: support any type of uniform dimension
        if (n.get_dtype().get_type_id() != strided_dim_type_id) {
            stringstream ss;
            ss << "TODO: binary_search on ndobject with dtype " << n.get_dtype() << " is not implemented";
            throw runtime_error(ss.str());
        }

        const char *n_data = n.get_readonly_originptr();
        intptr_t n_stride = reinterpret_cast<const strided_dim_dtype_metadata *>(n.get_ndo_meta())->stride;
        intptr_t first = 0, last = n.get_dim_size();
        while (first < last) {
            intptr_t trial = first + (last - first) / 2;
            const char *trial_data = n_data + trial * n_stride;

            // In order for the data to always match up with the metadata, need to have
            // trial_data first and data second in the comparison operations.
            if (k_d_less_n(data, trial_data)) {
                // value < arr[trial]
                last = trial;
            } else if (k_n_less_d(trial_data, data)) {
                // value > arr[trial]
                first = trial + 1;
            } else {
                return trial;
            }
        }
        return -1;
    }
}

ndobject dynd::groupby(const dynd::ndobject& data_values, const dynd::ndobject& by_values, const dynd::dtype& groups)
{
    if (data_values.get_undim() == 0) {
        throw runtime_error("'data' values provided to dynd groupby must have at least one dimension");
    }
    if (by_values.get_undim() == 0) {
        throw runtime_error("'by' values provided to dynd groupby must have at least one dimension");
    }
    if (data_values.get_dim_size() != by_values.get_dim_size()) {
        stringstream ss;
        ss << "'data' and 'by' values provided to dynd groupby have different sizes, ";
        ss << data_values.get_dim_size() << " and " << by_values.get_dim_size();
        throw runtime_error(ss.str());
    }

    // If no groups dtype is specified, determine one from 'by'
    dtype groups_final;
    if (groups.get_type_id() == uninitialized_type_id) {
        dtype by_dt = by_values.get_udtype();
        if (by_dt.value_dtype().get_type_id() == categorical_type_id) {
            // If 'by' already has a categorical dtype, use that
            groups_final = by_dt.value_dtype();
        } else {
            // Otherwise make a categorical type from the values
            groups_final = factor_categorical_dtype(by_values);
        }
    } else {
        groups_final = groups;
    }

    // Make sure the 'by' values have the 'groups' dtype
    ndobject by_values_as_groups = by_values.cast_udtype(groups_final);

    dtype gbdt = make_groupby_dtype(data_values.get_dtype(), by_values_as_groups.get_dtype());
    const groupby_dtype *gbdt_ext = static_cast<const groupby_dtype *>(gbdt.extended());
    char *data_ptr = NULL;

    ndobject result(make_ndobject_memory_block(gbdt.extended()->get_metadata_size(),
                    gbdt.extended()->get_data_size(), gbdt.extended()->get_alignment(), &data_ptr));

    // Set the metadata for the data values
    pointer_dtype_metadata *pmeta;
    pmeta = gbdt_ext->get_data_values_pointer_metadata(result.get_ndo_meta());
    pmeta->offset = 0;
    pmeta->blockref = data_values.get_ndo()->m_data_reference
                    ? data_values.get_ndo()->m_data_reference
                    : &data_values.get_ndo()->m_memblockdata;
    memory_block_incref(pmeta->blockref);
    data_values.get_dtype().extended()->metadata_copy_construct(reinterpret_cast<char *>(pmeta + 1),
                    data_values.get_ndo_meta(), &data_values.get_ndo()->m_memblockdata);

    // Set the metadata for the by values
    pmeta = gbdt_ext->get_by_values_pointer_metadata(result.get_ndo_meta());
    pmeta->offset = 0;
    pmeta->blockref = by_values_as_groups.get_ndo()->m_data_reference
                    ? by_values_as_groups.get_ndo()->m_data_reference
                    : &by_values_as_groups.get_ndo()->m_memblockdata;
    memory_block_incref(pmeta->blockref);
    by_values_as_groups.get_dtype().extended()->metadata_copy_construct(reinterpret_cast<char *>(pmeta + 1),
                    by_values_as_groups.get_ndo_meta(), &by_values_as_groups.get_ndo()->m_memblockdata);

    // Set the pointers to the data and by values data
    groupby_dtype_data *groupby_data_ptr = reinterpret_cast<groupby_dtype_data *>(data_ptr);
    groupby_data_ptr->data_values_pointer = data_values.get_readonly_originptr();
    groupby_data_ptr->by_values_pointer = by_values_as_groups.get_readonly_originptr();

    // Set the ndobject properties
    result.get_ndo()->m_dtype = gbdt.release();
    result.get_ndo()->m_data_pointer = data_ptr;
    result.get_ndo()->m_data_reference = NULL;
    result.get_ndo()->m_flags = read_access_flag;
    // If the inputs are immutable, the result is too
    if ((data_values.get_access_flags()&immutable_access_flag) != 0 && 
                    (by_values.get_access_flags()&immutable_access_flag) != 0) {
        result.get_ndo()->m_flags |= immutable_access_flag;
    }
    return result;
}

ndobject dynd::combine_into_struct(size_t field_count, const std::string *field_names,
                    const ndobject *field_values)
{
    // Make the pointer types
    vector<dtype> field_types(field_count);
    for (size_t i = 0; i != field_count; ++i) {
        field_types[i] = make_pointer_dtype(field_values[i].get_dtype());
    }
    // The flags are the intersection of all the input flags
    uint64_t flags = field_values[0].get_flags();
    for (size_t i = 1; i != field_count; ++i) {
        flags &= field_values[i].get_flags();
    }

    dtype result_type = make_fixedstruct_dtype(field_count, &field_types[0], field_names);
    const fixedstruct_dtype *fsd = static_cast<const fixedstruct_dtype *>(result_type.extended());
    char *data_ptr = NULL;

    ndobject result(make_ndobject_memory_block(fsd->get_metadata_size(),
                    fsd->get_data_size(),
                    fsd->get_alignment(), &data_ptr));
    // Set the ndobject properties
    result.get_ndo()->m_dtype = result_type.release();
    result.get_ndo()->m_data_pointer = data_ptr;
    result.get_ndo()->m_data_reference = NULL;
    result.get_ndo()->m_flags = flags;

    // Copy all the needed metadata
    const size_t *metadata_offsets = fsd->get_metadata_offsets();
    for (size_t i = 0; i != field_count; ++i) {
        pointer_dtype_metadata *pmeta;
        pmeta = reinterpret_cast<pointer_dtype_metadata *>(result.get_ndo_meta() + metadata_offsets[i]);
        pmeta->offset = 0;
        pmeta->blockref = field_values[i].get_ndo()->m_data_reference
                        ? field_values[i].get_ndo()->m_data_reference
                        : &field_values[i].get_ndo()->m_memblockdata;
        memory_block_incref(pmeta->blockref);

        const dtype& field_dt = field_values[i].get_dtype();
        if (field_dt.get_metadata_size() > 0) {
            field_dt.extended()->metadata_copy_construct(
                            reinterpret_cast<char *>(pmeta + 1),
                            field_values[i].get_ndo_meta(),
                            &field_values[i].get_ndo()->m_memblockdata);
        }
    }

    // Set the data pointers
    const char **dp = reinterpret_cast<const char **>(data_ptr);
    for (size_t i = 0; i != field_count; ++i) {
        dp[i] = field_values[i].get_readonly_originptr();
    }
    return result;
}

/*
static ndobject follow_ndobject_pointers(const ndobject& n)
{
    // Follow the pointers to eliminate them
    dtype dt = n.get_dtype();
    const char *metadata = n.get_ndo_meta();
    char *data = n.get_ndo()->m_data_pointer;
    memory_block_data *dataref = NULL;
    uint64_t flags = n.get_ndo()->m_flags;
    while (dt.get_type_id() == pointer_type_id) {
        const pointer_dtype_metadata *md = reinterpret_cast<const pointer_dtype_metadata *>(metadata);
        const pointer_dtype *pd = static_cast<const pointer_dtype *>(dt.extended());
        dt = pd->get_target_dtype();
        metadata += sizeof(pointer_dtype_metadata);
        data = *reinterpret_cast<char **>(data) + md->offset;
        dataref = md->blockref;
    }
    // Create an ndobject without the pointers
    ndobject result(make_ndobject_memory_block(dt.is_builtin() ? 0 : dt.extended()->get_metadata_size()));
    if (!dt.is_builtin()) {
        dt.extended()->metadata_copy_construct(result.get_ndo_meta(), metadata, &n.get_ndo()->m_memblockdata);
    }
    result.get_ndo()->m_dtype = dt.release();
    result.get_ndo()->m_data_pointer = data;
    result.get_ndo()->m_data_reference = dataref ? dataref : &n.get_ndo()->m_memblockdata;
    memory_block_incref(result.get_ndo()->m_data_reference);
    result.get_ndo()->m_flags = flags;
    return result;
}
*/

