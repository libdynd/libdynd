//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/array_iter.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/types/view_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/fixedbytes_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/base_memory_type.hpp>
#include <dynd/types/cuda_host_type.hpp>
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/comparison_kernels.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/gfunc/callable.hpp>
#include <dynd/gfunc/call_callable.hpp>
#include <dynd/types/groupby_type.hpp>
#include <dynd/types/categorical_type.hpp>
#include <dynd/types/builtin_type_properties.hpp>
#include <dynd/memblock/memmap_memory_block.hpp>

using namespace std;
using namespace dynd;

nd::array::array()
    : m_memblock()
{
}

void nd::array::swap(array& rhs)
{
    m_memblock.swap(rhs.m_memblock);
}

template<class T>
inline typename dynd::enable_if<is_dynd_scalar<T>::value, memory_block_ptr>::type
make_builtin_scalar_array(const T& value, uint64_t flags)
{
    char *data_ptr = NULL;
    memory_block_ptr result = make_array_memory_block(0, sizeof(T), scalar_align_of<T>::value, &data_ptr);
    *reinterpret_cast<T *>(data_ptr) = value;
    array_preamble *ndo = reinterpret_cast<array_preamble *>(result.get());
    ndo->m_type = reinterpret_cast<base_type *>(type_id_of<T>::value);
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = flags;
    return result;
}

nd::array nd::make_strided_array(const ndt::type& dtp, intptr_t ndim, const intptr_t *shape,
                int64_t access_flags, const int *axis_perm)
{
    // Create the type of the result
    bool any_variable_dims = false;
    ndt::type array_tp = ndt::make_type(ndim, shape, dtp, any_variable_dims);

    // Determine the total data size
    size_t data_size;
    if (array_tp.is_builtin()) {
        data_size = array_tp.get_data_size();
    } else {
        data_size = array_tp.extended()->get_default_data_size(ndim, shape);
    }

    memory_block_ptr result;
    char *data_ptr = NULL;
    if (dtp.get_kind() == memory_kind) {
        result = make_array_memory_block(array_tp.extended()->get_metadata_size());
        static_cast<const base_memory_type *>(dtp.extended())->data_alloc(&data_ptr, data_size);
    } else {
        // Allocate the array metadata and data in one memory block
        result = make_array_memory_block(array_tp.extended()->get_metadata_size(),
                    data_size, array_tp.get_data_alignment(), &data_ptr);
    }

    if (array_tp.get_flags()&type_flag_zeroinit) {
        if (dtp.get_kind() == memory_kind) {
            static_cast<const base_memory_type *>(dtp.extended())->data_zeroinit(data_ptr, data_size);
        }
        else {
            memset(data_ptr, 0, data_size);
        }
    }

    // Fill in the preamble metadata
    array_preamble *ndo = reinterpret_cast<array_preamble *>(result.get());
    ndo->m_type = array_tp.release();
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = access_flags;

    if (!any_variable_dims) {
        // Fill in the array metadata with strides and sizes
        strided_dim_type_metadata *meta = reinterpret_cast<strided_dim_type_metadata *>(ndo + 1);
        // Use the default construction to handle the uniform_tp's metadata
        intptr_t stride = dtp.get_data_size();
        if (stride == 0) {
            stride = dtp.extended()->get_default_data_size(0, NULL);
        }
        if (!dtp.is_builtin()) {
            dtp.extended()->metadata_default_construct(
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
            for (intptr_t i = 0; i < ndim; ++i) {
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
        // Fill in the array metadata with strides and sizes
        char *meta = reinterpret_cast<char *>(ndo + 1);
        ndo->m_type->metadata_default_construct(meta, ndim, shape);
    }

    return array(result);
}

nd::array nd::make_strided_array_from_data(const ndt::type& uniform_tp, intptr_t ndim, const intptr_t *shape,
                const intptr_t *strides, int64_t access_flags, char *data_ptr,
                const memory_block_ptr& data_reference, char **out_uniform_metadata)
{
    if (out_uniform_metadata == NULL && !uniform_tp.is_builtin() && uniform_tp.extended()->get_metadata_size() > 0) {
        stringstream ss;
        ss << "Cannot make a strided array with type " << uniform_tp << " from a preexisting data pointer";
        throw runtime_error(ss.str());
    }

    ndt::type array_type = ndt::make_strided_dim(uniform_tp, ndim);

    // Allocate the array metadata and data in one memory block
    memory_block_ptr result = make_array_memory_block(
                    array_type.get_metadata_size());

    // Fill in the preamble metadata
    array_preamble *ndo = reinterpret_cast<array_preamble *>(result.get());
    ndo->m_type = array_type.release();
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = data_reference.get();
    memory_block_incref(ndo->m_data_reference);
    ndo->m_flags = access_flags;

    // Fill in the array metadata with the shape and strides
    strided_dim_type_metadata *meta = reinterpret_cast<strided_dim_type_metadata *>(ndo + 1);
    for (intptr_t i = 0; i < ndim; ++i) {
        intptr_t dim_size = shape[i];
        meta[i].stride = dim_size > 1 ? strides[i] : 0;
        meta[i].size = dim_size;
    }

    // Return a pointer to the metadata for uniform_tp.
    if (out_uniform_metadata != NULL) {
        *out_uniform_metadata = reinterpret_cast<char *>(meta + ndim);
    }

    return nd::array(result);
}

nd::array nd::make_pod_array(const ndt::type& pod_dt, const void *data)
{
    size_t size = pod_dt.get_data_size();
    if (!pod_dt.is_pod()) {
        stringstream ss;
        ss << "Cannot make a dynd array from raw data using non-POD type " << pod_dt;
        throw runtime_error(ss.str());
    } else if (pod_dt.get_metadata_size() != 0) {
        stringstream ss;
        ss << "Cannot make a dynd array from raw data using type " << pod_dt;
        ss << " because it has non-empty dynd metadata";
        throw runtime_error(ss.str());
    }

    // Allocate the array metadata and data in one memory block
    char *data_ptr = NULL;
    memory_block_ptr result = make_array_memory_block(0, size, pod_dt.get_data_alignment(), &data_ptr);

    // Fill in the preamble metadata
    array_preamble *ndo = reinterpret_cast<array_preamble *>(result.get());
    if (pod_dt.is_builtin()) {
        ndo->m_type = reinterpret_cast<const base_type *>(pod_dt.get_type_id());
    } else {
        ndo->m_type = pod_dt.extended();
        base_type_incref(ndo->m_type);
    }
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = nd::default_access_flags;

    memcpy(data_ptr, data, size);

    return nd::array(result);
}

nd::array nd::make_bytes_array(const char *data, size_t len, size_t alignment)
{
    char *data_ptr = NULL, *bytes_data_ptr;
    ndt::type dt = ndt::make_bytes(alignment);
    nd::array result(make_array_memory_block(dt.extended()->get_metadata_size(),
                        dt.get_data_size() + len, dt.get_data_alignment(), &data_ptr));
    // Set the string extents
    bytes_data_ptr = data_ptr + dt.get_data_size();
    ((char **)data_ptr)[0] = bytes_data_ptr;
    ((char **)data_ptr)[1] = bytes_data_ptr + len;
    // Copy the string data
    memcpy(bytes_data_ptr, data, len);
    // Set the array metadata
    array_preamble *ndo = result.get_ndo();
    ndo->m_type = dt.release();
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = nd::default_access_flags;
    // Set the bytes metadata, telling the system that the bytes data was embedded in the array memory
    bytes_type_metadata *ndo_meta = reinterpret_cast<bytes_type_metadata *>(result.get_ndo_meta());
    ndo_meta->blockref = NULL;
    return result;
}


nd::array nd::make_string_array(const char *str, size_t len,
                string_encoding_t encoding, uint64_t access_flags)
{
    char *data_ptr = NULL, *string_ptr;
    ndt::type dt = ndt::make_string(encoding);
    nd::array result(make_array_memory_block(dt.extended()->get_metadata_size(),
                        dt.get_data_size() + len, dt.get_data_alignment(), &data_ptr));
    // Set the string extents
    string_ptr = data_ptr + dt.get_data_size();
    ((char **)data_ptr)[0] = string_ptr;
    ((char **)data_ptr)[1] = string_ptr + len;
    // Copy the string data
    memcpy(string_ptr, str, len);
    // Set the array metadata
    array_preamble *ndo = result.get_ndo();
    ndo->m_type = dt.release();
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = access_flags;
    // Set the string metadata, telling the system that the string data was embedded in the array memory
    string_type_metadata *ndo_meta = reinterpret_cast<string_type_metadata *>(result.get_ndo_meta());
    ndo_meta->blockref = NULL;
    return result;
}

nd::array nd::make_utf8_array_array(const char **cstr_array, size_t array_size)
{
    ndt::type dt = ndt::make_string(string_encoding_utf_8);
    nd::array result = nd::make_strided_array(array_size, dt);
    // Get the allocator for the output string type
    const string_type_metadata *md = reinterpret_cast<const string_type_metadata *>(result.get_ndo_meta() + sizeof(strided_dim_type_metadata));
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
 * Clones the metadata and swaps in a new type. The type must
 * have identical metadata, but this function doesn't check that.
 */
static nd::array make_array_clone_with_new_type(const nd::array& n, const ndt::type& new_dt)
{
    nd::array result(shallow_copy_array_memory_block(n.get_memblock()));
    array_preamble *preamble = result.get_ndo();
    // Swap in the type
    if (!preamble->is_builtin_type()) {
        base_type_decref(preamble->m_type);
    }
    preamble->m_type = new_dt.extended();
    if(!new_dt.is_builtin()) {
        base_type_incref(preamble->m_type);
    }
    return result;
}


// Constructors from C++ scalars
nd::array::array(dynd_bool value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(bool value)
    : m_memblock(make_builtin_scalar_array(dynd_bool(value),
                nd::default_access_flags))
{
}
nd::array::array(signed char value)
    : m_memblock(make_builtin_scalar_array(value,
                nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(short value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(int value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(long value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(long long value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(const dynd_int128& value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(unsigned char value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(unsigned short value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(unsigned int value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(unsigned long value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(unsigned long long value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(const dynd_uint128& value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(dynd_float16 value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(float value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(double value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(const dynd_float128& value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(dynd_complex<float> value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(dynd_complex<double> value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(std::complex<float> value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(std::complex<double> value)
    : m_memblock(make_builtin_scalar_array(value, nd::default_access_flags))
{
}
nd::array::array(const std::string& value)
{
    array temp = make_string_array(value.c_str(), value.size(),
                    string_encoding_utf_8, nd::default_access_flags);
    temp.swap(*this);
}
nd::array::array(const char *cstr)
{
    array temp = make_string_array(cstr, strlen(cstr),
                    string_encoding_utf_8, nd::default_access_flags);
    temp.swap(*this);
}
nd::array::array(const char *str, size_t size)
{
    array temp = make_string_array(str, size,
                    string_encoding_utf_8, nd::default_access_flags);
    temp.swap(*this);
}
nd::array::array(const ndt::type& tp)
{
    array temp = array(make_array_memory_block(ndt::make_type(), 0, NULL));
    temp.swap(*this);
    ndt::type(tp).swap(reinterpret_cast<type_type_data *>(get_ndo()->m_data_pointer)->tp);
    get_ndo()->m_flags = nd::default_access_flags;
}

nd::array nd::array_rw(dynd_bool value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(bool value)
{
    return nd::array(make_builtin_scalar_array(dynd_bool(value),
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(signed char value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(short value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(int value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(long value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(long long value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(const dynd_int128& value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(unsigned char value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(unsigned short value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(unsigned int value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(unsigned long value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(unsigned long long value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(const dynd_uint128& value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(dynd_float16 value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(float value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(double value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(const dynd_float128& value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(dynd_complex<float> value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(dynd_complex<double> value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(std::complex<float> value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(std::complex<double> value)
{
    return nd::array(make_builtin_scalar_array(value,
                    nd::readwrite_access_flags));
}
nd::array nd::array_rw(const std::string& value)
{
    return make_string_array(value.c_str(), value.size(),
                    string_encoding_utf_8, nd::readwrite_access_flags);
}
/** Construct a string from a NULL-terminated UTF8 string */
nd::array nd::array_rw(const char *cstr)
{
    return make_string_array(cstr, strlen(cstr),
                    string_encoding_utf_8, nd::readwrite_access_flags);
}
/** Construct a string from a UTF8 buffer and specified buffer size */
nd::array nd::array_rw(const char *str, size_t size)
{
    return make_string_array(str, size,
                    string_encoding_utf_8, nd::readwrite_access_flags);
}
nd::array nd::array_rw(const ndt::type& tp)
{
    array temp = array(make_array_memory_block(ndt::make_type(), 0, NULL));
    ndt::type(tp).swap(reinterpret_cast<type_type_data *>(temp.get_ndo()->m_data_pointer)->tp);
    temp.get_ndo()->m_flags = nd::readwrite_access_flags;
    return temp;
}

nd::array nd::detail::make_from_vec<ndt::type>::make(const std::vector<ndt::type>& vec)
{
    ndt::type dt = ndt::make_strided_dim(ndt::make_type());
    char *data_ptr = NULL;
    array result(make_array_memory_block(dt.extended()->get_metadata_size(),
                    sizeof(type_type_data) * vec.size(),
                    dt.get_data_alignment(), &data_ptr));
    // The main array metadata
    array_preamble *preamble = result.get_ndo();
    preamble->m_data_pointer = data_ptr;
    preamble->m_data_reference = NULL;
    preamble->m_type = dt.release();
    preamble->m_flags = read_access_flag | immutable_access_flag;
    // The metadata for the strided and string parts of the type
    strided_dim_type_metadata *sa_md = reinterpret_cast<strided_dim_type_metadata *>(
                                            result.get_ndo_meta());
    sa_md->size = vec.size();
    sa_md->stride = vec.empty() ? 0 : sizeof(type_type_data);
    // The data
    type_type_data *data = reinterpret_cast<type_type_data *>(data_ptr);
    for (size_t i = 0, i_end = vec.size(); i != i_end; ++i) {
        data[i].tp = ndt::type(vec[i]).release();
    }
    return result;
}

nd::array nd::detail::make_from_vec<std::string>::make(const std::vector<std::string>& vec)
{
    // Constructor detail for making an array from a vector of strings
    size_t total_string_size = 0;
    for (size_t i = 0, i_end = vec.size(); i != i_end; ++i) {
        total_string_size += vec[i].size();
    }

    ndt::type dt = ndt::make_strided_dim(ndt::make_string(string_encoding_utf_8));
    char *data_ptr = NULL;
    // Make an array memory block which contains both the string pointers and
    // the string data
    array result(make_array_memory_block(dt.extended()->get_metadata_size(),
                    sizeof(string_type_data) * vec.size() + total_string_size,
                    dt.get_data_alignment(), &data_ptr));
    char *string_ptr = data_ptr + sizeof(string_type_data) * vec.size();
    // The main array metadata
    array_preamble *preamble = result.get_ndo();
    preamble->m_data_pointer = data_ptr;
    preamble->m_data_reference = NULL;
    preamble->m_type = dt.release();
    preamble->m_flags = read_access_flag | immutable_access_flag;
    // The metadata for the strided and string parts of the type
    strided_dim_type_metadata *sa_md = reinterpret_cast<strided_dim_type_metadata *>(
                                            result.get_ndo_meta());
    sa_md->size = vec.size();
    sa_md->stride = vec.empty() ? 0 : sizeof(string_type_data);
    string_type_metadata *s_md = reinterpret_cast<string_type_metadata *>(sa_md + 1);
    s_md->blockref = NULL;
    // The string pointers and data
    string_type_data *data = reinterpret_cast<string_type_data *>(data_ptr);
    for (size_t i = 0, i_end = vec.size(); i != i_end; ++i) {
        size_t size = vec[i].size();
        memcpy(string_ptr, vec[i].data(), size);
        data[i].begin = string_ptr;
        string_ptr += size;
        data[i].end = string_ptr;
    }
    return result;
}

namespace {
    static void as_storage_type(const ndt::type& dt, void *DYND_UNUSED(extra),
                ndt::type& out_transformed_tp, bool& out_was_transformed)
    {
        // If the type is a simple POD, switch it to a bytes type. Otherwise, keep it
        // the same so that the metadata layout is identical.
        if (dt.is_scalar() && dt.get_type_id() != pointer_type_id) {
            const ndt::type& storage_dt = dt.storage_type();
            if (storage_dt.is_builtin()) {
                out_transformed_tp = ndt::make_fixedbytes(storage_dt.get_data_size(),
                                storage_dt.get_data_alignment());
                out_was_transformed = true;
            } else if (storage_dt.is_pod() && storage_dt.extended()->get_metadata_size() == 0) {
                out_transformed_tp = ndt::make_fixedbytes(storage_dt.get_data_size(),
                                storage_dt.get_data_alignment());
                out_was_transformed = true;
            } else if (storage_dt.get_type_id() == string_type_id) {
                out_transformed_tp = ndt::make_bytes(static_cast<const string_type *>(
                                storage_dt.extended())->get_target_alignment());
                out_was_transformed = true;
            } else {
                if (dt.get_kind() == expression_kind) {
                    out_transformed_tp = storage_dt;
                    out_was_transformed = true;
                } else {
                    // No transformation
                    out_transformed_tp = dt;
                }
            }
        } else {
            dt.extended()->transform_child_types(&as_storage_type, NULL, out_transformed_tp, out_was_transformed);
        }
    }
} // anonymous namespace

nd::array nd::array::storage() const
{
    ndt::type storage_dt = get_type();
    bool was_transformed = false;
    as_storage_type(get_type(), NULL, storage_dt, was_transformed);
    if (was_transformed) {
        return make_array_clone_with_new_type(*this, storage_dt);
    } else {
        return *this;
    }
}

nd::array nd::array::at_array(intptr_t nindices, const irange *indices, bool collapse_leading) const
{
    if (is_scalar()) {
        if (nindices != 0) {
            throw too_many_indices(get_type(), nindices, 0);
        }
        return *this;
    } else {
        ndt::type this_dt(get_ndo()->m_type, true);
        ndt::type dt = get_ndo()->m_type->apply_linear_index(nindices, indices,
                        0, this_dt, collapse_leading);
        array result;
        if (!dt.is_builtin()) {
            result.set(make_array_memory_block(dt.extended()->get_metadata_size()));
            result.get_ndo()->m_type = dt.extended();
            base_type_incref(result.get_ndo()->m_type);
        } else {
            result.set(make_array_memory_block(0));
            result.get_ndo()->m_type = reinterpret_cast<const base_type *>(dt.get_type_id());
        }
        result.get_ndo()->m_data_pointer = get_ndo()->m_data_pointer;
        if (get_ndo()->m_data_reference) {
            result.get_ndo()->m_data_reference = get_ndo()->m_data_reference;
        } else {
            // If the data reference is NULL, the data is embedded in the array itself
            result.get_ndo()->m_data_reference = m_memblock.get();
        }
        memory_block_incref(result.get_ndo()->m_data_reference);
        intptr_t offset = get_ndo()->m_type->apply_linear_index(nindices, indices,
                        get_ndo_meta(), dt, result.get_ndo_meta(),
                        m_memblock.get(), 0, this_dt,
                        collapse_leading,
                        &result.get_ndo()->m_data_pointer, &result.get_ndo()->m_data_reference);
        result.get_ndo()->m_data_pointer += offset;
        result.get_ndo()->m_flags = get_ndo()->m_flags;
        return result;
    }
}

void nd::array::val_assign(const array& rhs, assign_error_mode errmode,
                    const eval::eval_context *ectx) const
{
    // Verify read access permission
    if (!(rhs.get_flags()&read_access_flag)) {
        throw runtime_error("tried to read from a dynd array that is not readable");
    }

    typed_data_assign(get_type(), get_ndo_meta(), get_readwrite_originptr(),
                    rhs.get_type(), rhs.get_ndo_meta(), rhs.get_readonly_originptr(),
                    errmode, ectx);
}

void nd::array::val_assign(const ndt::type& rhs_dt, const char *rhs_metadata, const char *rhs_data,
                    assign_error_mode errmode, const eval::eval_context *ectx) const
{
    typed_data_assign(get_type(), get_ndo_meta(), get_readwrite_originptr(),
                    rhs_dt, rhs_metadata, rhs_data,
                    errmode, ectx);
}

void nd::array::flag_as_immutable()
{
    // If it's already immutable, everything's ok
    if ((get_flags()&immutable_access_flag) != 0) {
        return;
    }

    // Check that nobody else is peeking into our data
    bool ok = true;
    if (m_memblock.get()->m_use_count != 1) {
        // More than one reference to the array itself
        ok = false;
    } else if (get_ndo()->m_data_reference != NULL &&
            (get_ndo()->m_data_reference->m_use_count != 1 ||
             !(get_ndo()->m_data_reference->m_type == fixed_size_pod_memory_block_type ||
               get_ndo()->m_data_reference->m_type == pod_memory_block_type))) {
        // More than one reference to the array's data, or the reference is to something
        // other than a memblock owning its data, such as an external memblock.
        ok = false;
    } else if (!get_ndo()->is_builtin_type() &&
            !get_ndo()->m_type->is_unique_data_owner(get_ndo_meta())) {
        ok = false;
    }

    if (ok) {
        // Finalize any allocated data in the metadata
        if (!is_builtin_type(get_ndo()->m_type)) {
            get_ndo()->m_type->metadata_finalize_buffers(get_ndo_meta());
        }
        // Clear the write flag, and set the immutable flag
        get_ndo()->m_flags = (get_ndo()->m_flags&~(uint64_t)write_access_flag)|immutable_access_flag;
    } else {
        stringstream ss;
        ss << "Unable to flag array of type " << get_type() << " as immutable, because ";
        ss << "it does not uniquely own all of its data";
        throw runtime_error(ss.str());
    }
}

nd::array nd::array::p(const char *property_name) const
{
    ndt::type dt = get_type();
    const std::pair<std::string, gfunc::callable> *properties;
    size_t count;
    if (!dt.is_builtin()) {
        dt.extended()->get_dynamic_array_properties(&properties, &count);
    } else {
        get_builtin_type_dynamic_array_properties(dt.get_type_id(), &properties, &count);
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
    ss << "dynd array does not have property " << property_name;
    throw runtime_error(ss.str());
}

nd::array nd::array::p(const std::string& property_name) const
{
    ndt::type dt = get_type();
    const std::pair<std::string, gfunc::callable> *properties;
    size_t count;
    if (!dt.is_builtin()) {
        dt.extended()->get_dynamic_array_properties(&properties, &count);
    } else {
        get_builtin_type_dynamic_array_properties(dt.get_type_id(), &properties, &count);
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
    ss << "dynd array does not have property " << property_name;
    throw runtime_error(ss.str());
}

const gfunc::callable& nd::array::find_dynamic_function(const char *function_name) const
{
    ndt::type dt = get_type();
    if (!dt.is_builtin()) {
        const std::pair<std::string, gfunc::callable> *properties;
        size_t count;
        dt.extended()->get_dynamic_array_functions(&properties, &count);
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
    ss << "dynd array does not have function " << function_name;
    throw runtime_error(ss.str());
}

nd::array nd::array::eval(const eval::eval_context *ectx) const
{
    const ndt::type& current_tp = get_type();
    if (!current_tp.is_expression()) {
        return *this;
    } else {
        // Create a canonical type for the result
        const ndt::type& dt = current_tp.get_canonical_type();
        size_t ndim = current_tp.get_ndim();
        dimvector shape(ndim);
        get_shape(shape.get());
        array result(make_array_memory_block(dt, ndim, shape.get()));
        if (dt.get_type_id() == strided_dim_type_id) {
            // Reorder strides of output strided dimensions in a KEEPORDER fashion
            static_cast<const strided_dim_type *>(
                            dt.extended())->reorder_default_constructed_strides(
                                            result.get_ndo_meta(), get_type(), get_ndo_meta());
        }
        result.val_assign(*this, assign_error_default, ectx);
        return result;
    }
}

nd::array nd::array::eval_immutable(const eval::eval_context *ectx) const
{
    const ndt::type& current_tp = get_type();
    if ((get_access_flags()&immutable_access_flag) &&
                    !current_tp.is_expression()) {
        return *this;
    } else {
        // Create a canonical type for the result
        const ndt::type& dt = current_tp.get_canonical_type();
        size_t ndim = current_tp.get_ndim();
        dimvector shape(ndim);
        get_shape(shape.get());
        array result(make_array_memory_block(dt, ndim, shape.get()));
        if (dt.get_type_id() == strided_dim_type_id) {
            // Reorder strides of output strided dimensions in a KEEPORDER fashion
            static_cast<const strided_dim_type *>(
                            dt.extended())->reorder_default_constructed_strides(
                                            result.get_ndo_meta(), get_type(), get_ndo_meta());
        }
        result.val_assign(*this, assign_error_default, ectx);
        result.get_ndo()->m_flags = immutable_access_flag|read_access_flag;
        return result;
    }
}

nd::array nd::array::eval_copy(uint32_t access_flags, const eval::eval_context *ectx) const
{
    const ndt::type& current_tp = get_type();
    const ndt::type& dt = current_tp.get_canonical_type();
    size_t ndim = current_tp.get_ndim();
    dimvector shape(ndim);
    get_shape(shape.get());
    array result(make_array_memory_block(dt, ndim, shape.get()));
    if (dt.get_type_id() == strided_dim_type_id) {
        // Reorder strides of output strided dimensions in a KEEPORDER fashion
        static_cast<const strided_dim_type *>(
                        dt.extended())->reorder_default_constructed_strides(
                                        result.get_ndo_meta(),
                                        get_type(), get_ndo_meta());
    }
    result.val_assign(*this, assign_error_default, ectx);
    // If the access_flags are 0, use the defaults
    access_flags = access_flags ? access_flags
                                : (int32_t)nd::default_access_flags;
    // If the access_flags are just readonly, add immutable
    // because we just created a unique instance
    access_flags = (access_flags != nd::read_access_flag)
                    ? access_flags
                    : (nd::read_access_flag|nd::immutable_access_flag);
    result.get_ndo()->m_flags = access_flags;
    return result;
}

nd::array nd::array::to_host() const
{
    ndt::type dt = get_type().get_dtype();
    if (dt.get_kind() == memory_kind) {
        dt = static_cast<const base_memory_type *>(dt.extended())->get_storage_type();
    }

    array result = empty_like(*this, dt);
    result.val_assign(*this);

    return result;
}

#ifdef DYND_CUDA
nd::array nd::array::to_cuda_host(unsigned int cuda_host_flags) const
{
    ndt::type dt = get_type().get_dtype();
    if (dt.get_kind() == memory_kind) {
        dt = static_cast<const base_memory_type *>(dt.extended())->get_storage_type();
    }

    array result = empty_like(*this, make_cuda_host(dt, cuda_host_flags));
    result.val_assign(*this);

    return result;

}

nd::array nd::array::to_cuda_device() const
{
    ndt::type dt = get_type().get_dtype();
    if (dt.get_kind() == memory_kind) {
        dt = static_cast<const base_memory_type *>(dt.extended())->get_storage_type();
    }

    array result = empty_like(*this, make_cuda_device(dt));
    result.val_assign(*this);

    return result;
}
#endif

bool nd::array::op_sorting_less(const array& rhs) const
{
    comparison_ckernel_builder k;
    make_comparison_kernel(&k, 0, get_type(), get_ndo_meta(),
                    rhs.get_type(), rhs.get_ndo_meta(),
                    comparison_type_sorting_less,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool nd::array::operator<(const array& rhs) const
{
    comparison_ckernel_builder k;
    make_comparison_kernel(&k, 0, get_type(), get_ndo_meta(),
                    rhs.get_type(), rhs.get_ndo_meta(),
                    comparison_type_less,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool nd::array::operator<=(const array& rhs) const
{
    comparison_ckernel_builder k;
    make_comparison_kernel(&k, 0, get_type(), get_ndo_meta(),
                    rhs.get_type(), rhs.get_ndo_meta(),
                    comparison_type_less_equal,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool nd::array::operator==(const array& rhs) const
{
    comparison_ckernel_builder k;
    make_comparison_kernel(&k, 0, get_type(), get_ndo_meta(),
                    rhs.get_type(), rhs.get_ndo_meta(),
                    comparison_type_equal,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool nd::array::operator!=(const array& rhs) const
{
    comparison_ckernel_builder k;
    make_comparison_kernel(&k, 0, get_type(), get_ndo_meta(),
                    rhs.get_type(), rhs.get_ndo_meta(),
                    comparison_type_not_equal,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool nd::array::operator>=(const array& rhs) const
{
    comparison_ckernel_builder k;
    make_comparison_kernel(&k, 0, get_type(), get_ndo_meta(),
                    rhs.get_type(), rhs.get_ndo_meta(),
                    comparison_type_greater_equal,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool nd::array::operator>(const array& rhs) const
{
    comparison_ckernel_builder k;
    make_comparison_kernel(&k, 0, get_type(), get_ndo_meta(),
                    rhs.get_type(), rhs.get_ndo_meta(),
                    comparison_type_greater,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool nd::array::equals_exact(const array& rhs) const
{
    if (get_ndo() == rhs.get_ndo()) {
        return true;
    } else if (get_type() != rhs.get_type()) {
        return false;
    } else if (get_ndim() == 0) {
        comparison_ckernel_builder k;
        make_comparison_kernel(&k, 0,
                        get_type(), get_ndo_meta(),
                        rhs.get_type(), rhs.get_ndo_meta(),
                        comparison_type_equal, &eval::default_eval_context);
        return k(get_readonly_originptr(), rhs.get_readonly_originptr());
    } else {
        // First compare the shape, to avoid triggering an exception in common cases
        size_t ndim = get_ndim();
        dimvector shape0(ndim), shape1(ndim);
        get_shape(shape0.get());
        rhs.get_shape(shape1.get());
        if (memcmp(shape0.get(), shape1.get(), ndim * sizeof(intptr_t)) != 0) {
            return false;
        }
        try {
            array_iter<0,2> iter(*this, rhs);
            if (!iter.empty()) {
                comparison_ckernel_builder k;
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

nd::array nd::array::cast(const ndt::type& tp, assign_error_mode errmode) const
{
    // Use the ucast function specifying to replace all dimensions
    return ucast(tp, get_type().get_ndim(), errmode);
}

namespace {
    struct cast_dtype_extra {
        cast_dtype_extra(const ndt::type& tp, size_t ru, assign_error_mode em)
            : replacement_tp(tp), errmode(em), replace_ndim(ru), out_can_view_data(true)
        {
        }
        const ndt::type& replacement_tp;
        assign_error_mode errmode;
        intptr_t replace_ndim;
        bool out_can_view_data;
    };
    static void cast_dtype(const ndt::type& dt, void *extra,
                ndt::type& out_transformed_tp, bool& out_was_transformed)
    {
        cast_dtype_extra *e = reinterpret_cast<cast_dtype_extra *>(extra);
        intptr_t replace_ndim = e->replace_ndim;
        if (dt.get_ndim() > replace_ndim) {
            dt.extended()->transform_child_types(&cast_dtype, extra, out_transformed_tp, out_was_transformed);
        } else {
            if (replace_ndim > 0) {
                // If the dimension we're replacing doesn't change, then
                // avoid creating the convert type at this level
                if (dt.get_type_id() == e->replacement_tp.get_type_id()) {
                    bool can_keep_dim = false;
                    ndt::type child_dt, child_replacement_tp;
                    switch (dt.get_type_id()) {
                        case fixed_dim_type_id: {
                            const fixed_dim_type *dt_fdd = static_cast<const fixed_dim_type *>(dt.extended());
                            const fixed_dim_type *r_fdd = static_cast<const fixed_dim_type *>(e->replacement_tp.extended());
                            if (dt_fdd->get_fixed_dim_size() == r_fdd->get_fixed_dim_size() &&
                                    dt_fdd->get_fixed_stride() == r_fdd->get_fixed_stride()) {
                                can_keep_dim = true;
                                child_dt = dt_fdd->get_element_type();
                                child_replacement_tp = r_fdd->get_element_type();
                            }
                            break;
                        }
                        case strided_dim_type_id:
                        case var_dim_type_id: {
                            const base_uniform_dim_type *dt_budd =
                                            static_cast<const base_uniform_dim_type *>(dt.extended());
                            const base_uniform_dim_type *r_budd =
                                            static_cast<const base_uniform_dim_type *>(e->replacement_tp.extended());
                            can_keep_dim = true;
                            child_dt = dt_budd->get_element_type();
                            child_replacement_tp = r_budd->get_element_type();
                            break;
                        }
                        default:
                            break;
                    }
                    if (can_keep_dim) {
                        cast_dtype_extra extra_child(child_replacement_tp,
                                    replace_ndim - 1, e->errmode);
                        dt.extended()->transform_child_types(&cast_dtype,
                                        &extra_child, out_transformed_tp, out_was_transformed);
                        return;
                    }
                }
            }
            out_transformed_tp = ndt::make_convert(e->replacement_tp, dt, e->errmode);
            // Only flag the transformation if this actually created a convert type
            if (out_transformed_tp.extended() != e->replacement_tp.extended()) {
                out_was_transformed= true;
                e->out_can_view_data = false;
            }
        }
    }
} // anonymous namespace

nd::array nd::array::ucast(const ndt::type& scalar_tp,
                intptr_t replace_ndim,
                assign_error_mode errmode) const
{
    // This creates a type which has a convert type for every scalar of different type.
    // The result has the exact same metadata and data, so we just have to swap in the new
    // type in a shallow copy.
    ndt::type replaced_tp;
    bool was_transformed = false;
    cast_dtype_extra extra(scalar_tp, replace_ndim, errmode);
    cast_dtype(get_type(), &extra, replaced_tp, was_transformed);
    if (was_transformed) {
        return make_array_clone_with_new_type(*this, replaced_tp);
    } else {
        return *this;
    }
}

nd::array nd::array::view(const ndt::type& DYND_UNUSED(dt)) const
{
    // It appears that the way to fully support this operation is
    // similar to the two-pass indexing operation. This would
    // do one recursive pass through the type to build the output
    // type (e.g. determine whether the data can be viewed directly,
    // or expression types need a view expression added on the end),
    // then a second pass to construct the metadata
    throw runtime_error("TODO: Implement nd::array::view");
}

nd::array nd::array::uview(const ndt::type& uniform_dt, intptr_t replace_ndim) const
{
    // Use the view function specifying to replace all dimensions
    return view(get_type().with_replaced_dtype(uniform_dt, replace_ndim));
}

namespace {
    struct replace_compatible_dtype_extra {
        replace_compatible_dtype_extra(const ndt::type& tp,
                        intptr_t replace_ndim_)
            : replacement_tp(tp), replace_ndim(replace_ndim_)
        {
        }
        const ndt::type& replacement_tp;
        intptr_t replace_ndim;
    };
    static void replace_compatible_dtype(const ndt::type& tp, void *extra,
                ndt::type& out_transformed_tp, bool& out_was_transformed)
    {
        const replace_compatible_dtype_extra *e =
                        reinterpret_cast<const replace_compatible_dtype_extra *>(extra);
        const ndt::type& replacement_tp = e->replacement_tp;
        if (tp.get_ndim() == e->replace_ndim) {
            if (tp != replacement_tp) {
                if (!tp.data_layout_compatible_with(replacement_tp)) {
                    stringstream ss;
                    ss << "The dynd type " << tp << " is not ";
                    ss << " data layout compatible with " << replacement_tp;
                    ss << ", so a substitution cannot be made.";
                    throw runtime_error(ss.str());
                }
                out_transformed_tp = replacement_tp;
                out_was_transformed= true;
            }
        } else {
            tp.extended()->transform_child_types(&replace_compatible_dtype,
                            extra, out_transformed_tp, out_was_transformed);
        }
    }
} // anonymous namespace

nd::array nd::array::replace_dtype(const ndt::type& replacement_tp, intptr_t replace_ndim) const
{
    // This creates a type which swaps in the new dtype for
    // the existing one. It raises an error if the data layout
    // is incompatible
    ndt::type replaced_tp;
    bool was_transformed = false;
    replace_compatible_dtype_extra extra(replacement_tp, replace_ndim);
    replace_compatible_dtype(get_type(), &extra,
                    replaced_tp, was_transformed);
    if (was_transformed) {
        return make_array_clone_with_new_type(*this, replaced_tp);
    } else {
        return *this;
    }
}

namespace {
    static void view_scalar_types(const ndt::type& dt, void *extra,
                ndt::type& out_transformed_tp, bool& out_was_transformed)
    {
        if (dt.is_scalar()) {
            const ndt::type *e = reinterpret_cast<const ndt::type *>(extra);
            // If things aren't simple, use a view_type
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
                                // All these types have the same data/metadata layout,
                                // allow a view whenever the alignment allows it
                                if (e->get_data_alignment() <= dt.get_data_alignment()) {
                                    out_transformed_tp = *e;
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
                out_transformed_tp = ndt::make_view(*e, dt);
                out_was_transformed = true;
            } else {
                out_transformed_tp = *e;
                if (dt != *e) {
                    out_was_transformed = true;
                }
            }
        } else {
            dt.extended()->transform_child_types(&view_scalar_types, extra, out_transformed_tp, out_was_transformed);
        }
    }
} // anonymous namespace

nd::array nd::array::view_scalars(const ndt::type& scalar_tp) const
{
    const ndt::type& array_type = get_type();
    size_t uniform_ndim = array_type.get_ndim();
    // First check if we're dealing with a simple one dimensional block of memory we can reinterpret
    // at will.
    if (uniform_ndim == 1 && array_type.get_type_id() == strided_dim_type_id) {
        const strided_dim_type *sad = static_cast<const strided_dim_type *>(array_type.extended());
        const strided_dim_type_metadata *md = reinterpret_cast<const strided_dim_type_metadata *>(get_ndo_meta());
        const ndt::type& edt = sad->get_element_type();
        if (edt.is_pod() && (intptr_t)edt.get_data_size() == md->stride &&
                    sad->get_element_type().get_kind() != expression_kind) {
            intptr_t nbytes = md->size * edt.get_data_size();
            // Make sure the element size divides into the # of bytes
            if (nbytes % scalar_tp.get_data_size() != 0) {
                std::stringstream ss;
                ss << "cannot view array with " << nbytes << " bytes as type ";
                ss << scalar_tp << ", because its element size " << scalar_tp.get_data_size();
                ss << " doesn't divide evenly into the total array size " << nbytes;
                throw std::runtime_error(ss.str());
            }
            // Create the result array, adjusting the type if the data isn't aligned correctly
            char *data_ptr = get_ndo()->m_data_pointer;
            ndt::type result_tp;
            if ((((uintptr_t)data_ptr)&(scalar_tp.get_data_alignment()-1)) == 0) {
                result_tp = ndt::make_strided_dim(scalar_tp);
            } else {
                result_tp = ndt::make_strided_dim(make_unaligned(scalar_tp));
            }
            array result(make_array_memory_block(result_tp.extended()->get_metadata_size()));
            // Copy all the array metadata fields
            result.get_ndo()->m_data_pointer = get_ndo()->m_data_pointer;
            if (get_ndo()->m_data_reference) {
                result.get_ndo()->m_data_reference = get_ndo()->m_data_reference;
            } else {
                result.get_ndo()->m_data_reference = m_memblock.get();
            }
            memory_block_incref(result.get_ndo()->m_data_reference);
            result.get_ndo()->m_type = result_tp.release();
            result.get_ndo()->m_flags = get_ndo()->m_flags;
            // The result has one strided ndarray field
            strided_dim_type_metadata *result_md = reinterpret_cast<strided_dim_type_metadata *>(result.get_ndo_meta());
            result_md->size = nbytes / scalar_tp.get_data_size();
            result_md->stride = scalar_tp.get_data_size();
            return result;
        }
    }

    // Transform the scalars into view types
    ndt::type viewed_tp;
    bool was_transformed;
    view_scalar_types(get_type(), const_cast<void *>(reinterpret_cast<const void *>(&scalar_tp)), viewed_tp, was_transformed);
    return make_array_clone_with_new_type(*this, viewed_tp);
}

std::string nd::detail::array_as_string(const nd::array& lhs, assign_error_mode errmode)
{
    if (!lhs.is_scalar()) {
        throw std::runtime_error("can only convert arrays with 0 dimensions to scalars");
    }

    nd::array temp = lhs;
    if (temp.get_type().get_kind() != string_kind) {
        temp = temp.ucast(ndt::make_string(string_encoding_utf_8)).eval();
    }
    const base_string_type *esd =
                    static_cast<const base_string_type *>(temp.get_type().extended());
    return esd->get_utf8_string(temp.get_ndo_meta(), temp.get_ndo()->m_data_pointer, errmode);
}

ndt::type nd::detail::array_as_type(const nd::array& lhs, assign_error_mode errmode)
{
    if (!lhs.is_scalar()) {
        throw std::runtime_error("can only convert arrays with 0 dimensions to scalars");
    }

    nd::array temp = lhs;
    if (temp.get_type().get_type_id() != type_type_id) {
        temp = temp.ucast(ndt::make_type(), 0, errmode).eval();
    }
    return ndt::type(reinterpret_cast<const type_type_data *>(temp.get_readonly_originptr())->tp, true);
}

void nd::array::debug_print(std::ostream& o, const std::string& indent) const
{
    o << indent << "------ array\n";
    if (m_memblock.get()) {
        const array_preamble *ndo = get_ndo();
        o << " address: " << (void *)m_memblock.get() << "\n";
        o << " refcount: " << ndo->m_memblockdata.m_use_count << "\n";
        o << " type:\n";
        o << "  pointer: " << (void *)ndo->m_type << "\n";
        o << "  type: " << get_type() << "\n";
        o << " metadata:\n";
        o << "  flags: " << ndo->m_flags << " (";
        if (ndo->m_flags & read_access_flag) o << "read_access ";
        if (ndo->m_flags & write_access_flag) o << "write_access ";
        if (ndo->m_flags & immutable_access_flag) o << "immutable ";
        o << ")\n";
        if (!ndo->is_builtin_type()) {
            o << "  type-specific metadata:\n";
            ndo->m_type->metadata_debug_print(get_ndo_meta(), o, indent + "   ");
        }
        o << " data:\n";
        o << "   pointer: " << (void *)ndo->m_data_pointer << "\n";
        o << "   reference: " << (void *)ndo->m_data_reference;
        if (ndo->m_data_reference == NULL) {
            o << " (embedded in array memory)\n";
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

std::ostream& nd::operator<<(std::ostream& o, const array& rhs)
{
    if (!rhs.is_empty()) {
        o << "array(";
        array v = rhs.eval();
        if (v.get_ndo()->is_builtin_type()) {
            print_builtin_scalar(v.get_ndo()->get_builtin_type_id(), o, v.get_ndo()->m_data_pointer);
        } else {
            if (v.get_ndo()->m_type->get_flags() & type_flag_not_host_readable) {
                v = v.to_host();
            }
            v.get_ndo()->m_type->print_data(o, v.get_ndo_meta(), v.get_ndo()->m_data_pointer);
        }
        o << ", type=\"" << rhs.get_type() << "\")";
    } else {
        o << "array()";
    }
    return o;
}

nd::array nd::eval_raw_copy(const ndt::type& dt, const char *metadata, const char *data)
{
    // Allocate an output array with the canonical version of the type
    ndt::type cdt = dt.get_canonical_type();
    size_t ndim = dt.get_ndim();
    array result;
    if (ndim > 0) {
        dimvector shape(ndim);
        dt.extended()->get_shape(ndim, 0, shape.get(), metadata, data);
        result.set(make_array_memory_block(cdt, ndim, shape.get()));
        // Reorder strides of output strided dimensions in a KEEPORDER fashion
        if (dt.get_type_id() == strided_dim_type_id) {
            static_cast<const strided_dim_type *>(
                        cdt.extended())->reorder_default_constructed_strides(result.get_ndo_meta(),
                                    dt, metadata);
        }
    } else {
        result.set(make_array_memory_block(cdt, 0, NULL));
    }

    typed_data_assign(cdt, result.get_ndo_meta(), result.get_readwrite_originptr(),
                    dt, metadata, data,
                    assign_error_default, &eval::default_eval_context);

    return result;
}

nd::array nd::empty(const ndt::type& tp)
{
    return nd::array(make_array_memory_block(tp, 0, NULL));
}

nd::array nd::empty(intptr_t dim0, const ndt::type& tp)
{
    return nd::array(make_array_memory_block(tp, 1, &dim0));
}

nd::array nd::empty(intptr_t dim0, intptr_t dim1, const ndt::type& dt)
{
    intptr_t dims[2] = {dim0, dim1};
    return nd::array(make_array_memory_block(dt, 2, dims));
}

nd::array nd::empty(intptr_t dim0, intptr_t dim1, intptr_t dim2, const ndt::type& dt)
{
    intptr_t dims[3] = {dim0, dim1, dim2};
    return nd::array(make_array_memory_block(dt, 3, dims));
}

nd::array nd::empty_like(const nd::array& rhs, const ndt::type& uniform_tp)
{
    if (rhs.get_ndim() == 0) {
        return nd::empty(uniform_tp);
    } else {
        size_t ndim = rhs.get_type().extended()->get_ndim();
        dimvector shape(ndim);
        rhs.get_shape(shape.get());
        array result(make_strided_array(uniform_tp, ndim, shape.get()));
        // Reorder strides of output strided dimensions in a KEEPORDER fashion
        if (result.get_type().get_type_id() == strided_dim_type_id) {
            static_cast<const strided_dim_type *>(
                        result.get_type().extended())->reorder_default_constructed_strides(
                                        result.get_ndo_meta(),
                                        rhs.get_type(), rhs.get_ndo_meta());
        }
        return result;
    }
}

nd::array nd::empty_like(const nd::array& rhs)
{
    ndt::type dt;
    if (rhs.get_ndo()->is_builtin_type()) {
        dt = ndt::type(rhs.get_ndo()->get_builtin_type_id());
    } else {
        dt = rhs.get_ndo()->m_type->get_canonical_type();
    }

    if (rhs.is_scalar()) {
        return nd::empty(dt);
    } else {
        size_t ndim = dt.extended()->get_ndim();
        dimvector shape(ndim);
        rhs.get_shape(shape.get());
        nd::array result(make_strided_array(dt.get_dtype(), ndim, shape.get()));
        // Reorder strides of output strided dimensions in a KEEPORDER fashion
        if (result.get_type().get_type_id() == strided_dim_type_id) {
            static_cast<const strided_dim_type *>(
                        result.get_type().extended())->reorder_default_constructed_strides(
                                        result.get_ndo_meta(),
                                        rhs.get_type(), rhs.get_ndo_meta());
        }
        return result;
    }
}

nd::array nd::memmap(const std::string& filename,
    intptr_t begin,
    intptr_t end,
    uint32_t access)
{
    if (access == 0) {
        access = nd::default_access_flags;
    }

    char *mm_ptr = NULL;
    intptr_t mm_size = 0;
    // Create a memory mapped memblock of the file
    memory_block_ptr mm = make_memmap_memory_block(
        filename, access, &mm_ptr, &mm_size, begin, end);
    // Create a bytes array referring to the data.
    ndt::type dt = ndt::make_bytes(1);
    char *data_ptr = 0;
    nd::array result(make_array_memory_block(dt.extended()->get_metadata_size(),
                        dt.get_data_size(), dt.get_data_alignment(), &data_ptr));
    // Set the bytes extents
    ((char **)data_ptr)[0] = mm_ptr;
    ((char **)data_ptr)[1] = mm_ptr + mm_size;
    // Set the array metadata
    array_preamble *ndo = result.get_ndo();
    ndo->m_type = dt.release();
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = access;
    // Set the bytes metadata, telling the system
    // about the memmapped memblock
    bytes_type_metadata *ndo_meta =
        reinterpret_cast<bytes_type_metadata *>(result.get_ndo_meta());
    ndo_meta->blockref = mm.release();
    return result;
}

intptr_t nd::binary_search(const nd::array& n, const char *metadata, const char *data)
{
    if (n.get_ndim() == 0) {
        stringstream ss;
        ss << "cannot do a dynd binary_search on array with type " << n.get_type() << " without a leading array dimension";
        throw runtime_error(ss.str());
    }
    const char *n_metadata = n.get_ndo_meta();
    ndt::type element_tp = n.get_type().at_single(0, &n_metadata);
    if (element_tp.get_metadata_size() == 0 || n_metadata == metadata ||
                    memcmp(n_metadata, metadata, element_tp.get_metadata_size()) == 0) {
        // First, a version where the metadata is identical, so we can
        // make do with only a single comparison kernel
        comparison_ckernel_builder k_n_less_d;
        make_comparison_kernel(&k_n_less_d, 0,
                        element_tp, n_metadata,
                        element_tp, n_metadata,
                        comparison_type_sorting_less,
                        &eval::default_eval_context);

        // TODO: support any type of array dimension
        if (n.get_type().get_type_id() != strided_dim_type_id) {
            stringstream ss;
            ss << "TODO: binary_search on array with type " << n.get_type() << " is not implemented";
            throw runtime_error(ss.str());
        }

        const char *n_data = n.get_readonly_originptr();
        intptr_t n_stride = reinterpret_cast<const strided_dim_type_metadata *>(n.get_ndo_meta())->stride;
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
        comparison_ckernel_builder k_n_less_d, k_d_less_n;
        make_comparison_kernel(&k_n_less_d, 0,
                        element_tp, n_metadata,
                        element_tp, metadata,
                        comparison_type_sorting_less,
                        &eval::default_eval_context);
        make_comparison_kernel(&k_d_less_n, 0,
                        element_tp, metadata,
                        element_tp, n_metadata,
                        comparison_type_sorting_less,
                        &eval::default_eval_context);

        // TODO: support any type of array dimension
        if (n.get_type().get_type_id() != strided_dim_type_id) {
            stringstream ss;
            ss << "TODO: binary_search on array with type " << n.get_type() << " is not implemented";
            throw runtime_error(ss.str());
        }

        const char *n_data = n.get_readonly_originptr();
        intptr_t n_stride = reinterpret_cast<const strided_dim_type_metadata *>(n.get_ndo_meta())->stride;
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

nd::array nd::groupby(const nd::array& data_values, const nd::array& by_values, const dynd::ndt::type& groups)
{
    if (data_values.get_ndim() == 0) {
        throw runtime_error("'data' values provided to dynd groupby must have at least one dimension");
    }
    if (by_values.get_ndim() == 0) {
        throw runtime_error("'by' values provided to dynd groupby must have at least one dimension");
    }
    if (data_values.get_dim_size() != by_values.get_dim_size()) {
        stringstream ss;
        ss << "'data' and 'by' values provided to dynd groupby have different sizes, ";
        ss << data_values.get_dim_size() << " and " << by_values.get_dim_size();
        throw runtime_error(ss.str());
    }

    // If no groups type is specified, determine one from 'by'
    ndt::type groups_final;
    if (groups.get_type_id() == uninitialized_type_id) {
        ndt::type by_dt = by_values.get_dtype();
        if (by_dt.value_type().get_type_id() == categorical_type_id) {
            // If 'by' already has a categorical type, use that
            groups_final = by_dt.value_type();
        } else {
            // Otherwise make a categorical type from the values
            groups_final = ndt::factor_categorical(by_values);
        }
    } else {
        groups_final = groups;
    }

    // Make sure the 'by' values have the 'groups' type
    array by_values_as_groups = by_values.ucast(groups_final);

    ndt::type gbdt = ndt::make_groupby(data_values.get_type(), by_values_as_groups.get_type());
    const groupby_type *gbdt_ext = static_cast<const groupby_type *>(gbdt.extended());
    char *data_ptr = NULL;

    array result(make_array_memory_block(gbdt.extended()->get_metadata_size(),
                    gbdt.extended()->get_data_size(), gbdt.extended()->get_data_alignment(), &data_ptr));

    // Set the metadata for the data values
    pointer_type_metadata *pmeta;
    pmeta = gbdt_ext->get_data_values_pointer_metadata(result.get_ndo_meta());
    pmeta->offset = 0;
    pmeta->blockref = data_values.get_ndo()->m_data_reference
                    ? data_values.get_ndo()->m_data_reference
                    : &data_values.get_ndo()->m_memblockdata;
    memory_block_incref(pmeta->blockref);
    data_values.get_type().extended()->metadata_copy_construct(reinterpret_cast<char *>(pmeta + 1),
                    data_values.get_ndo_meta(), &data_values.get_ndo()->m_memblockdata);

    // Set the metadata for the by values
    pmeta = gbdt_ext->get_by_values_pointer_metadata(result.get_ndo_meta());
    pmeta->offset = 0;
    pmeta->blockref = by_values_as_groups.get_ndo()->m_data_reference
                    ? by_values_as_groups.get_ndo()->m_data_reference
                    : &by_values_as_groups.get_ndo()->m_memblockdata;
    memory_block_incref(pmeta->blockref);
    by_values_as_groups.get_type().extended()->metadata_copy_construct(reinterpret_cast<char *>(pmeta + 1),
                    by_values_as_groups.get_ndo_meta(), &by_values_as_groups.get_ndo()->m_memblockdata);

    // Set the pointers to the data and by values data
    groupby_type_data *groupby_data_ptr = reinterpret_cast<groupby_type_data *>(data_ptr);
    groupby_data_ptr->data_values_pointer = data_values.get_readonly_originptr();
    groupby_data_ptr->by_values_pointer = by_values_as_groups.get_readonly_originptr();

    // Set the array properties
    result.get_ndo()->m_type = gbdt.release();
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

nd::array nd::combine_into_struct(size_t field_count, const std::string *field_names,
                    const array *field_values)
{
    // Make the pointer types
    vector<ndt::type> field_types(field_count);
    for (size_t i = 0; i != field_count; ++i) {
        field_types[i] = ndt::make_pointer(field_values[i].get_type());
    }
    // The flags are the intersection of all the input flags
    uint64_t flags = field_values[0].get_flags();
    for (size_t i = 1; i != field_count; ++i) {
        flags &= field_values[i].get_flags();
    }

    ndt::type result_type = ndt::make_cstruct(field_count, &field_types[0], field_names);
    const cstruct_type *fsd = static_cast<const cstruct_type *>(result_type.extended());
    char *data_ptr = NULL;

    array result(make_array_memory_block(fsd->get_metadata_size(),
                    fsd->get_data_size(),
                    fsd->get_data_alignment(), &data_ptr));
    // Set the array properties
    result.get_ndo()->m_type = result_type.release();
    result.get_ndo()->m_data_pointer = data_ptr;
    result.get_ndo()->m_data_reference = NULL;
    result.get_ndo()->m_flags = flags;

    // Copy all the needed metadata
    const size_t *metadata_offsets = fsd->get_metadata_offsets();
    for (size_t i = 0; i != field_count; ++i) {
        pointer_type_metadata *pmeta;
        pmeta = reinterpret_cast<pointer_type_metadata *>(result.get_ndo_meta() + metadata_offsets[i]);
        pmeta->offset = 0;
        pmeta->blockref = field_values[i].get_ndo()->m_data_reference
                        ? field_values[i].get_ndo()->m_data_reference
                        : &field_values[i].get_ndo()->m_memblockdata;
        memory_block_incref(pmeta->blockref);

        const ndt::type& field_dt = field_values[i].get_type();
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
static array follow_array_pointers(const array& n)
{
    // Follow the pointers to eliminate them
    ndt::type dt = n.get_type();
    const char *metadata = n.get_ndo_meta();
    char *data = n.get_ndo()->m_data_pointer;
    memory_block_data *dataref = NULL;
    uint64_t flags = n.get_ndo()->m_flags;
    while (dt.get_type_id() == pointer_type_id) {
        const pointer_type_metadata *md = reinterpret_cast<const pointer_type_metadata *>(metadata);
        const pointer_type *pd = static_cast<const pointer_type *>(dt.extended());
        dt = pd->get_target_type();
        metadata += sizeof(pointer_type_metadata);
        data = *reinterpret_cast<char **>(data) + md->offset;
        dataref = md->blockref;
    }
    // Create an array without the pointers
    array result(make_array_memory_block(dt.is_builtin() ? 0 : dt.extended()->get_metadata_size()));
    if (!dt.is_builtin()) {
        dt.extended()->metadata_copy_construct(result.get_ndo_meta(), metadata, &n.get_ndo()->m_memblockdata);
    }
    result.get_ndo()->m_type = dt.release();
    result.get_ndo()->m_data_pointer = data;
    result.get_ndo()->m_data_reference = dataref ? dataref : &n.get_ndo()->m_memblockdata;
    memory_block_incref(result.get_ndo()->m_data_reference);
    result.get_ndo()->m_flags = flags;
    return result;
}
*/

