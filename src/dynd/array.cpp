//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/array_iter.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/ctuple_type.hpp>
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
#include <dynd/types/option_type.hpp>
#include <dynd/types/adapt_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/comparison_kernels.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/func/call_callable.hpp>
#include <dynd/types/groupby_type.hpp>
#include <dynd/types/categorical_type.hpp>
#include <dynd/types/builtin_type_properties.hpp>
#include <dynd/memblock/memmap_memory_block.hpp>
#include <dynd/kernels/make_lifted_ckernel.hpp>
#include <dynd/view.hpp>

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

nd::array nd::make_strided_array(const ndt::type &dtp, intptr_t ndim,
                                 const intptr_t *shape, int64_t access_flags,
                                 const int *axis_perm)
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
        result = make_array_memory_block(array_tp.get_arrmeta_size());
        dtp.tcast<base_memory_type>()->data_alloc(&data_ptr, data_size);
    } else {
        // Allocate the array arrmeta and data in one memory block
        result = make_array_memory_block(array_tp.get_arrmeta_size(),
                    data_size, array_tp.get_data_alignment(), &data_ptr);
    }

    if (array_tp.get_flags()&type_flag_zeroinit) {
        if (dtp.get_kind() == memory_kind) {
            dtp.tcast<base_memory_type>()->data_zeroinit(data_ptr, data_size);
        }
        else {
            memset(data_ptr, 0, data_size);
        }
    }

    // Fill in the preamble arrmeta
    array_preamble *ndo = reinterpret_cast<array_preamble *>(result.get());
    ndo->m_type = array_tp.release();
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = access_flags;

    if (!any_variable_dims) {
        // Fill in the array arrmeta with strides and sizes
        strided_dim_type_arrmeta *meta = reinterpret_cast<strided_dim_type_arrmeta *>(ndo + 1);
        // Use the default construction to handle the uniform_tp's arrmeta
        intptr_t stride = dtp.get_data_size();
        if (stride == 0) {
            stride = dtp.extended()->get_default_data_size(0, NULL);
        }
        if (!dtp.is_builtin()) {
            dtp.extended()->arrmeta_default_construct(
                            reinterpret_cast<char *>(meta + ndim), 0, NULL);
        }
        if (axis_perm == NULL) {
            for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
                intptr_t dim_size = shape[i];
                meta[i].stride = dim_size > 1 ? stride : 0;
                meta[i].dim_size = dim_size;
                stride *= dim_size;
            }
        } else {
            for (intptr_t i = 0; i < ndim; ++i) {
                int i_perm = axis_perm[i];
                intptr_t dim_size = shape[i_perm];
                meta[i_perm].stride = dim_size > 1 ? stride : 0;
                meta[i_perm].dim_size = dim_size;
                stride *= dim_size;
            }
        }
    } else {
        if (axis_perm != NULL) {
            // Maybe force C-order in this case?
            throw runtime_error("dynd presently only supports C-order with variable-sized arrays");
        }
        // Fill in the array arrmeta with strides and sizes
        char *meta = reinterpret_cast<char *>(ndo + 1);
        ndo->m_type->arrmeta_default_construct(meta, ndim, shape);
    }

    return array(result);
}

nd::array nd::make_strided_array_from_data(const ndt::type& uniform_tp, intptr_t ndim, const intptr_t *shape,
                const intptr_t *strides, int64_t access_flags, char *data_ptr,
                const memory_block_ptr& data_reference, char **out_uniform_arrmeta)
{
    if (out_uniform_arrmeta == NULL && !uniform_tp.is_builtin() && uniform_tp.extended()->get_arrmeta_size() > 0) {
        stringstream ss;
        ss << "Cannot make a strided array with type " << uniform_tp << " from a preexisting data pointer";
        throw runtime_error(ss.str());
    }

    ndt::type array_type = ndt::make_strided_dim(uniform_tp, ndim);

    // Allocate the array arrmeta and data in one memory block
    memory_block_ptr result =
        make_array_memory_block(array_type.get_arrmeta_size());

    // Fill in the preamble arrmeta
    array_preamble *ndo = reinterpret_cast<array_preamble *>(result.get());
    ndo->m_type = array_type.release();
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = data_reference.get();
    memory_block_incref(ndo->m_data_reference);
    ndo->m_flags = access_flags;

    // Fill in the array arrmeta with the shape and strides
    strided_dim_type_arrmeta *meta = reinterpret_cast<strided_dim_type_arrmeta *>(ndo + 1);
    for (intptr_t i = 0; i < ndim; ++i) {
        intptr_t dim_size = shape[i];
        meta[i].stride = dim_size > 1 ? strides[i] : 0;
        meta[i].dim_size = dim_size;
    }

    // Return a pointer to the arrmeta for uniform_tp.
    if (out_uniform_arrmeta != NULL) {
        *out_uniform_arrmeta = reinterpret_cast<char *>(meta + ndim);
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
    } else if (pod_dt.get_arrmeta_size() != 0) {
        stringstream ss;
        ss << "Cannot make a dynd array from raw data using type " << pod_dt;
        ss << " because it has non-empty dynd arrmeta";
        throw runtime_error(ss.str());
    }

    // Allocate the array arrmeta and data in one memory block
    char *data_ptr = NULL;
    memory_block_ptr result = make_array_memory_block(
        0, size, pod_dt.get_data_alignment(), &data_ptr);

    // Fill in the preamble arrmeta
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
    nd::array result(
        make_array_memory_block(dt.extended()->get_arrmeta_size(),
                                dt.get_data_size() + len + alignment - 1,
                                dt.get_data_alignment(), &data_ptr));
    // Set the string extents
    bytes_data_ptr = inc_to_alignment(data_ptr + dt.get_data_size(), alignment);
    ((char **)data_ptr)[0] = bytes_data_ptr;
    ((char **)data_ptr)[1] = bytes_data_ptr + len;
    // Copy the string data
    memcpy(bytes_data_ptr, data, len);
    // Set the array arrmeta
    array_preamble *ndo = result.get_ndo();
    ndo->m_type = dt.release();
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = nd::default_access_flags;
    // Set the bytes arrmeta, telling the system that the bytes data was embedded in the array memory
    bytes_type_arrmeta *ndo_meta = reinterpret_cast<bytes_type_arrmeta *>(result.get_arrmeta());
    ndo_meta->blockref = NULL;
    return result;
}

nd::array nd::make_string_array(const char *str, size_t len,
                                string_encoding_t encoding,
                                uint64_t access_flags)
{
  char *data_ptr = NULL, *string_ptr;
  ndt::type dt = ndt::make_string(encoding);
  nd::array result(make_array_memory_block(dt.extended()->get_arrmeta_size(),
                                           dt.get_data_size() + len,
                                           dt.get_data_alignment(), &data_ptr));
  // Set the string extents
  string_ptr = data_ptr + dt.get_data_size();
  ((char **)data_ptr)[0] = string_ptr;
  ((char **)data_ptr)[1] = string_ptr + len;
  // Copy the string data
  memcpy(string_ptr, str, len);
  // Set the array arrmeta
  array_preamble *ndo = result.get_ndo();
  ndo->m_type = dt.release();
  ndo->m_data_pointer = data_ptr;
  ndo->m_data_reference = NULL;
  ndo->m_flags = access_flags;
  // Set the string arrmeta, telling the system that the string data was
  // embedded in the array memory
  string_type_arrmeta *ndo_meta =
      reinterpret_cast<string_type_arrmeta *>(result.get_arrmeta());
  ndo_meta->blockref = NULL;
  return result;
}

nd::array nd::make_strided_string_array(const char **cstr_array, size_t array_size)
{
    size_t total_string_length = 0;
    for (size_t i = 0; i != array_size; ++i) {
        total_string_length += strlen(cstr_array[i]);
    }

    char *data_ptr = NULL, *string_ptr;
    string_type_data *string_arr_ptr;
    ndt::type stp = ndt::make_string(string_encoding_utf_8);
    ndt::type tp = ndt::make_strided_dim(stp);
    nd::array result(make_array_memory_block(
        tp.extended()->get_arrmeta_size(),
        array_size * stp.get_data_size() + total_string_length,
        tp.get_data_alignment(), &data_ptr));
    // Set the array arrmeta
    array_preamble *ndo = result.get_ndo();
    ndo->m_type = tp.release();
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = default_access_flags;
    // Get the allocator for the output string type
    strided_dim_type_arrmeta *md =
        reinterpret_cast<strided_dim_type_arrmeta *>(
            result.get_arrmeta());
    md->dim_size = array_size;
    md->stride = stp.get_data_size();
    string_arr_ptr = reinterpret_cast<string_type_data *>(data_ptr);
    string_ptr = data_ptr + array_size * stp.get_data_size();
    for (size_t i = 0; i < array_size; ++i) {
        size_t size = strlen(cstr_array[i]);
        memcpy(string_ptr, cstr_array[i], size);
        string_arr_ptr->begin = string_ptr;
        string_arr_ptr->end = string_ptr + size;
        ++string_arr_ptr;
        string_ptr += size;
    }
    return result;
}

nd::array nd::make_strided_string_array(const std::string **str_array, size_t array_size)
{
    size_t total_string_length = 0;
    for (size_t i = 0; i != array_size; ++i) {
        total_string_length += str_array[i]->size();
    }

    char *data_ptr = NULL, *string_ptr;
    string_type_data *string_arr_ptr;
    ndt::type stp = ndt::make_string(string_encoding_utf_8);
    ndt::type tp = ndt::make_strided_dim(stp);
    nd::array result(make_array_memory_block(
        tp.extended()->get_arrmeta_size(),
        array_size * stp.get_data_size() + total_string_length,
        tp.get_data_alignment(), &data_ptr));
    // Set the array arrmeta
    array_preamble *ndo = result.get_ndo();
    ndo->m_type = tp.release();
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = default_access_flags;
    // Get the allocator for the output string type
    strided_dim_type_arrmeta *md =
        reinterpret_cast<strided_dim_type_arrmeta *>(
            result.get_arrmeta());
    md->dim_size = array_size;
    md->stride = stp.get_data_size();
    string_arr_ptr = reinterpret_cast<string_type_data *>(data_ptr);
    string_ptr = data_ptr + array_size * stp.get_data_size();
    for (size_t i = 0; i < array_size; ++i) {
        size_t size = str_array[i]->size();
        memcpy(string_ptr, str_array[i]->data(), size);
        string_arr_ptr->begin = string_ptr;
        string_arr_ptr->end = string_ptr + size;
        ++string_arr_ptr;
        string_ptr += size;
    }
    result.flag_as_immutable();
    return result;
}


/**
 * Clones the arrmeta and swaps in a new type. The type must
 * have identical arrmeta, but this function doesn't check that.
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
    array temp(nd::typed_empty(0, static_cast<const intptr_t *>(NULL), ndt::make_type()));
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
  array temp = array(nd::typed_empty(0, static_cast<const intptr_t *>(NULL), ndt::make_type()));
    ndt::type(tp).swap(reinterpret_cast<type_type_data *>(temp.get_ndo()->m_data_pointer)->tp);
    return temp;
}

nd::array nd::detail::make_from_vec<ndt::type>::make(const std::vector<ndt::type>& vec)
{
    ndt::type dt = ndt::make_strided_of_type();
    char *data_ptr = NULL;
    array result(make_array_memory_block(dt.extended()->get_arrmeta_size(),
                    sizeof(type_type_data) * vec.size(),
                    dt.get_data_alignment(), &data_ptr));
    // The main array arrmeta
    array_preamble *preamble = result.get_ndo();
    preamble->m_data_pointer = data_ptr;
    preamble->m_data_reference = NULL;
    preamble->m_type = dt.release();
    preamble->m_flags = read_access_flag | immutable_access_flag;
    // The arrmeta for the strided and string parts of the type
    strided_dim_type_arrmeta *sa_md = reinterpret_cast<strided_dim_type_arrmeta *>(
                                            result.get_arrmeta());
    sa_md->dim_size = vec.size();
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

    ndt::type dt = ndt::make_strided_of_string();
    char *data_ptr = NULL;
    // Make an array memory block which contains both the string pointers and
    // the string data
    array result(make_array_memory_block(dt.extended()->get_arrmeta_size(),
                    sizeof(string_type_data) * vec.size() + total_string_size,
                    dt.get_data_alignment(), &data_ptr));
    char *string_ptr = data_ptr + sizeof(string_type_data) * vec.size();
    // The main array arrmeta
    array_preamble *preamble = result.get_ndo();
    preamble->m_data_pointer = data_ptr;
    preamble->m_data_reference = NULL;
    preamble->m_type = dt.release();
    preamble->m_flags = read_access_flag | immutable_access_flag;
    // The arrmeta for the strided and string parts of the type
    strided_dim_type_arrmeta *sa_md = reinterpret_cast<strided_dim_type_arrmeta *>(
                                            result.get_arrmeta());
    sa_md->dim_size = vec.size();
    sa_md->stride = vec.empty() ? 0 : sizeof(string_type_data);
    string_type_arrmeta *s_md = reinterpret_cast<string_type_arrmeta *>(sa_md + 1);
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
    static void as_storage_type(const ndt::type& dt, void *DYND_UNUSED(self),
                ndt::type& out_transformed_tp, bool& out_was_transformed)
    {
        // If the type is a simple POD, switch it to a bytes type. Otherwise, keep it
        // the same so that the arrmeta layout is identical.
        if (dt.is_scalar() && dt.get_type_id() != pointer_type_id) {
            const ndt::type& storage_dt = dt.storage_type();
            if (storage_dt.is_builtin()) {
                out_transformed_tp = ndt::make_fixedbytes(storage_dt.get_data_size(),
                                storage_dt.get_data_alignment());
                out_was_transformed = true;
            } else if (storage_dt.is_pod() && storage_dt.extended()->get_arrmeta_size() == 0) {
                out_transformed_tp = ndt::make_fixedbytes(storage_dt.get_data_size(),
                                storage_dt.get_data_alignment());
                out_was_transformed = true;
            } else if (storage_dt.get_type_id() == string_type_id) {
                out_transformed_tp = ndt::make_bytes(static_cast<const string_type *>(
                                storage_dt.extended())->get_target_alignment());
                out_was_transformed = true;
            } else {
                if (dt.get_kind() == expr_kind) {
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
            result.set(make_array_memory_block(dt.extended()->get_arrmeta_size()));
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
                        get_arrmeta(), dt, result.get_arrmeta(),
                        m_memblock.get(), 0, this_dt,
                        collapse_leading,
                        &result.get_ndo()->m_data_pointer, &result.get_ndo()->m_data_reference);
        result.get_ndo()->m_data_pointer += offset;
        result.get_ndo()->m_flags = get_ndo()->m_flags;
        return result;
    }
}

void nd::array::val_assign(const array &rhs, const eval::eval_context *ectx)
    const
{
  // Verify read access permission
  if (!(rhs.get_flags() & read_access_flag)) {
    throw runtime_error("tried to read from a dynd array that is not readable");
  }

  typed_data_assign(get_type(), get_arrmeta(), get_readwrite_originptr(),
                    rhs.get_type(), rhs.get_arrmeta(),
                    rhs.get_readonly_originptr(), ectx);
}

void nd::array::val_assign(const ndt::type &rhs_dt, const char *rhs_arrmeta,
                           const char *rhs_data, const eval::eval_context *ectx)
    const
{
    typed_data_assign(get_type(), get_arrmeta(), get_readwrite_originptr(),
                      rhs_dt, rhs_arrmeta, rhs_data, ectx);
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
            !get_ndo()->m_type->is_unique_data_owner(get_arrmeta())) {
        ok = false;
    }

    if (ok) {
        // Finalize any allocated data in the arrmeta
        if (!is_builtin_type(get_ndo()->m_type)) {
            get_ndo()->m_type->arrmeta_finalize_buffers(get_arrmeta());
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
        array result(nd::typed_empty(ndim, shape.get(), dt));
        if (dt.get_type_id() == strided_dim_type_id) {
            // Reorder strides of output strided dimensions in a KEEPORDER fashion
            static_cast<const strided_dim_type *>(dt.extended())
                ->reorder_default_constructed_strides(
                    result.get_arrmeta(), get_type(), get_arrmeta());
        }
        result.val_assign(*this, ectx);
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
        array result(nd::typed_empty(ndim, shape.get(), dt));
        if (dt.get_type_id() == strided_dim_type_id) {
            // Reorder strides of output strided dimensions in a KEEPORDER fashion
            static_cast<const strided_dim_type *>(
                            dt.extended())->reorder_default_constructed_strides(
                                            result.get_arrmeta(), get_type(), get_arrmeta());
        }
        result.val_assign(*this, ectx);
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
    array result(nd::typed_empty(ndim, shape.get(), dt));
    if (dt.get_type_id() == strided_dim_type_id) {
        // Reorder strides of output strided dimensions in a KEEPORDER fashion
        static_cast<const strided_dim_type *>(
                        dt.extended())->reorder_default_constructed_strides(
                                        result.get_arrmeta(),
                                        get_type(), get_arrmeta());
    }
    result.val_assign(*this, ectx);
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
        dt = dt.tcast<base_memory_type>()->get_storage_type();
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
        dt = dt.tcast<base_memory_type>()->get_storage_type();
    }

    array result = empty_like(*this, make_cuda_host(dt, cuda_host_flags));
    result.val_assign(*this);

    return result;

}

nd::array nd::array::to_cuda_device() const
{
    ndt::type dt = get_type().get_dtype();
    if (dt.get_kind() == memory_kind) {
        dt = dt.tcast<base_memory_type>()->get_storage_type();
    }

    array result = empty_like(*this, make_cuda_device(dt));
    result.val_assign(*this);

    return result;
}
#endif

bool nd::array::op_sorting_less(const array& rhs) const
{
    comparison_ckernel_builder k;
    make_comparison_kernel(&k, 0, get_type(), get_arrmeta(),
                    rhs.get_type(), rhs.get_arrmeta(),
                    comparison_type_sorting_less,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool nd::array::operator<(const array& rhs) const
{
    comparison_ckernel_builder k;
    make_comparison_kernel(&k, 0, get_type(), get_arrmeta(),
                    rhs.get_type(), rhs.get_arrmeta(),
                    comparison_type_less,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool nd::array::operator<=(const array& rhs) const
{
    comparison_ckernel_builder k;
    make_comparison_kernel(&k, 0, get_type(), get_arrmeta(),
                    rhs.get_type(), rhs.get_arrmeta(),
                    comparison_type_less_equal,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool nd::array::operator==(const array& rhs) const
{
    comparison_ckernel_builder k;
    make_comparison_kernel(&k, 0, get_type(), get_arrmeta(),
                    rhs.get_type(), rhs.get_arrmeta(),
                    comparison_type_equal,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool nd::array::operator!=(const array& rhs) const
{
    comparison_ckernel_builder k;
    make_comparison_kernel(&k, 0, get_type(), get_arrmeta(),
                    rhs.get_type(), rhs.get_arrmeta(),
                    comparison_type_not_equal,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool nd::array::operator>=(const array& rhs) const
{
    comparison_ckernel_builder k;
    make_comparison_kernel(&k, 0, get_type(), get_arrmeta(),
                    rhs.get_type(), rhs.get_arrmeta(),
                    comparison_type_greater_equal,
                    &eval::default_eval_context);
    return k(get_readonly_originptr(), rhs.get_readonly_originptr());
}

bool nd::array::operator>(const array& rhs) const
{
    comparison_ckernel_builder k;
    make_comparison_kernel(&k, 0, get_type(), get_arrmeta(),
                    rhs.get_type(), rhs.get_arrmeta(),
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
                        get_type(), get_arrmeta(),
                        rhs.get_type(), rhs.get_arrmeta(),
                        comparison_type_equal, &eval::default_eval_context);
        return k(get_readonly_originptr(), rhs.get_readonly_originptr());
    } else if (get_type().get_type_id() == var_dim_type_id) {
      // If there's a leading var dimension, convert it to strided and compare
      // (Note: this is a hack)
      return operator()(irange()).equals_exact(rhs(irange()));
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
                                iter.get_uniform_dtype<0>(), iter.arrmeta<0>(),
                                iter.get_uniform_dtype<1>(), iter.arrmeta<1>(),
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

nd::array nd::array::cast(const ndt::type& tp) const
{
    // Use the ucast function specifying to replace all dimensions
    return ucast(tp, get_type().get_ndim());
}

namespace {
    struct cast_dtype_extra {
        cast_dtype_extra(const ndt::type& tp, size_t ru)
            : replacement_tp(tp), replace_ndim(ru), out_can_view_data(true)
        {
        }
        const ndt::type& replacement_tp;
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
                        case cfixed_dim_type_id: {
                            const cfixed_dim_type *dt_fdd = dt.tcast<cfixed_dim_type>();
                            const cfixed_dim_type *r_fdd = static_cast<const cfixed_dim_type *>(e->replacement_tp.extended());
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
                            const base_dim_type *dt_budd =
                                            dt.tcast<base_dim_type>();
                            const base_dim_type *r_budd =
                                            static_cast<const base_dim_type *>(e->replacement_tp.extended());
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
                                                     replace_ndim - 1);
                        dt.extended()->transform_child_types(&cast_dtype,
                                        &extra_child, out_transformed_tp, out_was_transformed);
                        return;
                    }
                }
            }
            out_transformed_tp = ndt::make_convert(e->replacement_tp, dt);
            // Only flag the transformation if this actually created a convert type
            if (out_transformed_tp.extended() != e->replacement_tp.extended()) {
                out_was_transformed= true;
                e->out_can_view_data = false;
            }
        }
    }
} // anonymous namespace

nd::array nd::array::ucast(const ndt::type &scalar_tp, intptr_t replace_ndim)
    const
{
  // This creates a type which has a convert type for every scalar of different
  // type.
  // The result has the exact same arrmeta and data, so we just have to swap in
  // the new
  // type in a shallow copy.
  ndt::type replaced_tp;
  bool was_transformed = false;
  cast_dtype_extra extra(scalar_tp, replace_ndim);
  cast_dtype(get_type(), &extra, replaced_tp, was_transformed);
  if (was_transformed) {
    return make_array_clone_with_new_type(*this, replaced_tp);
  } else {
    return *this;
  }
}

nd::array nd::array::view(const ndt::type& tp) const
{
  return nd::view(*this, tp);
}

nd::array nd::array::uview(const ndt::type& uniform_dt, intptr_t replace_ndim) const
{
  // Use the view function specifying to replace all dimensions
  return view(get_type().with_replaced_dtype(uniform_dt, replace_ndim));
}

nd::array nd::array::adapt(const ndt::type& tp, const nd::string& adapt_op)
{
  return uview(ndt::make_adapt(get_dtype(), tp, adapt_op), 0);
}

namespace {
static void with_strided_dim_type(const ndt::type &tp, void *extra,
                                  ndt::type &out_transformed_tp,
                                  bool &out_was_transformed)
{
  if (tp.get_ndim() > 0) {
    tp.extended()->transform_child_types(
        &with_strided_dim_type, extra, out_transformed_tp, out_was_transformed);
    type_id_t tp_id = tp.get_type_id();
    if (tp_id == fixed_dim_type_id || tp_id == cfixed_dim_type_id) {
      out_transformed_tp = ndt::make_strided_dim(
          out_transformed_tp.tcast<base_dim_type>()
              ->get_element_type());
      out_was_transformed = true;
    }
  } else {
    out_transformed_tp = tp;
  }
}
} // anonymous namespace

nd::array nd::array::permute(intptr_t ndim, const intptr_t *axes) const
{
  if (ndim > get_ndim()) {
    stringstream ss;
    ss << "Too many dimensions provided for axis permutation, got " << ndim
       << " for type " << get_type();
    throw invalid_argument(ss.str());
  }
  ndt::type transformed_tp;
  bool was_transformed = false;
  with_strided_dim_type(get_type(), NULL, transformed_tp, was_transformed);

  nd::array res(shallow_copy_array_memory_block(get_memblock()));
  res = res.view(transformed_tp);

  dimvector shape(get_ndim());
  get_shape(shape.get());

  dimvector strides(get_ndim());
  get_strides(strides.get());

  char *md = res.get_arrmeta();
  shortvector<char> permcheck(ndim);
  memset(permcheck.get(), 0, ndim * sizeof(bool));
  // ``barrier`` is the highest ``var`` dimension prior
  // to ``i``, axes are not permitted to cross this barrier
  intptr_t barrier = -1;
  for (intptr_t i = 0; i < ndim; ++i) {
    // A permutation must leave dimensions that are not strided alone, so this
    // check handles those.
    intptr_t axes_i = axes[i];
    if (axes_i < 0 || axes_i >= ndim || permcheck[axes_i] != 0) {
      stringstream ss;
      ss << "Invalid axis permutation [" << axes[0];
      for (i = 1; i < ndim; ++i) {
        ss << ", " << i;
      }
      ss << "]";
      throw invalid_argument(ss.str());
    } else {
      permcheck[axes_i] = 1;
    }
    if (i != axes_i) {
      if (shape[i] >= 0 && axes_i > barrier) {
        // It's a strided dim and does not cross the barrier
        strided_dim_type_arrmeta *smd =
            reinterpret_cast<strided_dim_type_arrmeta *>(md);
        smd->dim_size = shape[axes_i];
        smd->stride = strides[axes_i];
      } else if (shape[i] < 0) {
        throw invalid_argument(
            "Cannot permute a dynd var dimension, it must remain fixed");
      } else {
        throw invalid_argument(
            "Cannot permute a strided dimension across a dynd var dimension");
      }
    } else if (shape[i] < 0) {
      barrier = i;
    }

    transformed_tp = transformed_tp.get_type_at_dimension(&md, 1);
  }

  return res;
}

nd::array nd::array::rotate(intptr_t to, intptr_t from) const
{
    if (from < to) {
        intptr_t ndim = to + 1;
        dimvector axes(ndim);
        for (intptr_t i = 0; i < from; ++i) {
            axes[i] = i;
        }
        for (intptr_t i = from; i < to; ++i) {
            axes[i] = i + 1;
        }
        axes[to] = from;

        return permute(ndim, axes.get());
    }

    if (from > to) {
        intptr_t ndim = from + 1;
        dimvector axes(ndim);
        for (intptr_t i = 0; i < to; ++i) {
            axes[i] = i;
        }
        axes[to] = from;
        for (intptr_t i = to + 1; i <= from; ++i) {
            axes[i] = i - 1;
        }

        return permute(ndim, axes.get());
    }

    return *this;
}

nd::array nd::array::transpose() const
{
    intptr_t ndim = get_ndim();
    dimvector axes(ndim);
    for (intptr_t i = 0; i < ndim; ++i) {
        axes[i] = ndim - i - 1;
    }

    return permute(ndim, axes.get());
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
            if (dt.get_kind() == expr_kind || dt.get_data_size() != e->get_data_size() ||
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
                                // All these types have the same data/arrmeta layout,
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
        const strided_dim_type *sad = array_type.tcast<strided_dim_type>();
        const strided_dim_type_arrmeta *md = reinterpret_cast<const strided_dim_type_arrmeta *>(get_arrmeta());
        const ndt::type& edt = sad->get_element_type();
        if (edt.is_pod() && (intptr_t)edt.get_data_size() == md->stride &&
                    sad->get_element_type().get_kind() != expr_kind) {
            intptr_t nbytes = md->dim_size * edt.get_data_size();
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
            array result(make_array_memory_block(result_tp.extended()->get_arrmeta_size()));
            // Copy all the array arrmeta fields
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
            strided_dim_type_arrmeta *result_md = reinterpret_cast<strided_dim_type_arrmeta *>(result.get_arrmeta());
            result_md->dim_size = nbytes / scalar_tp.get_data_size();
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
        temp = temp.ucast(ndt::make_string()).eval();
    }
    const base_string_type *esd =
                    static_cast<const base_string_type *>(temp.get_type().extended());
    return esd->get_utf8_string(temp.get_arrmeta(), temp.get_ndo()->m_data_pointer, errmode);
}

ndt::type nd::detail::array_as_type(const nd::array& lhs)
{
    if (!lhs.is_scalar()) {
        throw std::runtime_error("can only convert arrays with 0 dimensions to scalars");
    }

    nd::array temp = lhs;
    if (temp.get_type().get_type_id() != type_type_id) {
        temp = temp.ucast(ndt::make_type()).eval();
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
        o << " arrmeta:\n";
        o << "  flags: " << ndo->m_flags << " (";
        if (ndo->m_flags & read_access_flag) o << "read_access ";
        if (ndo->m_flags & write_access_flag) o << "write_access ";
        if (ndo->m_flags & immutable_access_flag) o << "immutable ";
        o << ")\n";
        if (!ndo->is_builtin_type()) {
            o << "  type-specific arrmeta:\n";
            ndo->m_type->arrmeta_debug_print(get_arrmeta(), o, indent + "   ");
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
    if (!rhs.is_null()) {
        o << "array(";
        array v = rhs.eval();
        if (v.get_ndo()->is_builtin_type()) {
          print_builtin_scalar(v.get_ndo()->get_builtin_type_id(), o,
                               v.get_ndo()->m_data_pointer);
        } else {
            if (v.get_ndo()->m_type->get_flags() & type_flag_not_host_readable) {
                v = v.to_host();
            }
            stringstream ss;
            v.get_ndo()->m_type->print_data(ss, v.get_arrmeta(),
                                            v.get_ndo()->m_data_pointer);
            print_indented(o, "      ", ss.str(), true);
        }
        o << ",\n      type=\"" << rhs.get_type() << "\")";
    } else {
        o << "array()";
    }
    return o;
}

nd::array nd::eval_raw_copy(const ndt::type& dt, const char *arrmeta, const char *data)
{
    // Allocate an output array with the canonical version of the type
    ndt::type cdt = dt.get_canonical_type();
    size_t ndim = dt.get_ndim();
    array result;
    if (ndim > 0) {
        dimvector shape(ndim);
        dt.extended()->get_shape(ndim, 0, shape.get(), arrmeta, data);
        result = nd::typed_empty(ndim, shape.get(), cdt);
        // Reorder strides of output strided dimensions in a KEEPORDER fashion
        if (dt.get_type_id() == strided_dim_type_id) {
            static_cast<const strided_dim_type *>(cdt.extended())
                ->reorder_default_constructed_strides(result.get_arrmeta(), dt,
                                                      arrmeta);
        }
    } else {
        result = nd::typed_empty(0, static_cast<const intptr_t *>(NULL), cdt);
    }

    typed_data_assign(cdt, result.get_arrmeta(),
                      result.get_readwrite_originptr(), dt, arrmeta, data,
                      &eval::default_eval_context);

    return result;
}

nd::array nd::typed_empty(intptr_t ndim, const intptr_t *shape,
                          const ndt::type &tp)
{
  if (tp.is_builtin()) {
    // This code path builds a builtin scalar type as directly as possible
    if (ndim != 0) {
      stringstream ss;
      ss << "too many dimensions provided (" << ndim
         << ") for creating dynd array of type " << tp;
      throw invalid_argument(ss.str());
    }
    char *data_ptr = NULL;
    intptr_t data_size =
        static_cast<intptr_t>(dynd::detail::builtin_data_sizes
                                  [reinterpret_cast<uintptr_t>(tp.extended())]);
    intptr_t data_alignment =
        static_cast<intptr_t>(dynd::detail::builtin_data_alignments
                                  [reinterpret_cast<uintptr_t>(tp.extended())]);
    memory_block_ptr result(
        make_array_memory_block(0, data_size, data_alignment, &data_ptr));
    array_preamble *preamble = reinterpret_cast<array_preamble *>(result.get());
    // It's a builtin type id, so no incref
    preamble->m_type = tp.extended();
    preamble->m_data_pointer = data_ptr;
    preamble->m_data_reference = NULL;
    preamble->m_flags = nd::read_access_flag | nd::write_access_flag;
    return nd::array(DYND_MOVE(result));
  } else {
    if (ndim != 0 && tp.is_scalar()) {
      stringstream ss;
      ss << "too many dimensions provided (" << ndim
         << ") for creating dynd array of type " << tp;
      throw invalid_argument(ss.str());
    }
    char *data_ptr = NULL;
    size_t arrmeta_size = tp.extended()->get_arrmeta_size();
    size_t data_size = tp.extended()->get_default_data_size(ndim, shape);
    memory_block_ptr result;
    const ndt::type &dtp = tp.get_dtype();
    if (dtp.get_kind() != memory_kind) {
      // Allocate memory the default way
      result = make_array_memory_block(
          arrmeta_size, data_size, tp.get_data_alignment(), &data_ptr);
      if (tp.get_flags() & type_flag_zeroinit) {
        memset(data_ptr, 0, data_size);
      }
    } else {
      // Allocate memory based on the memory_kind type
      result = make_array_memory_block(arrmeta_size);
      static_cast<const base_memory_type *>(dtp.extended())
          ->data_alloc(&data_ptr, data_size);
      if (tp.get_flags() & type_flag_zeroinit) {
        static_cast<const base_memory_type *>(dtp.extended())
            ->data_zeroinit(data_ptr, data_size);
      }
    }
    array_preamble *preamble = reinterpret_cast<array_preamble *>(result.get());
    preamble->m_type = ndt::type(tp).release();
    preamble->m_type->arrmeta_default_construct(
        reinterpret_cast<char *>(preamble + 1), ndim, shape);
    preamble->m_data_pointer = data_ptr;
    preamble->m_data_reference = NULL;
    preamble->m_flags = nd::read_access_flag | nd::write_access_flag;
    return nd::array(DYND_MOVE(result));
  }
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
                                        result.get_arrmeta(),
                                        rhs.get_type(), rhs.get_arrmeta());
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
                                        result.get_arrmeta(),
                                        rhs.get_type(), rhs.get_arrmeta());
        }
        return result;
    }
}

nd::array nd::typed_zeros(intptr_t ndim, const intptr_t *shape,
                          const ndt::type &tp)
{
    nd::array res = nd::typed_empty(ndim, shape, tp);
    res.val_assign(0);

    return res;
}

nd::array nd::typed_ones(intptr_t ndim, const intptr_t *shape,
                          const ndt::type &tp)
{
    nd::array res = nd::typed_empty(ndim, shape, tp);
    res.val_assign(1);

    return res;
}

nd::array nd::concatenate(const nd::array &x, const nd::array &y) {
    if (x.get_ndim() != 1 || y.get_ndim() != 1) {
        throw runtime_error("TODO: nd::concatenate is WIP");
    }

    if (x.get_dtype() != y.get_dtype()) {
        throw runtime_error("dtypes must be the same for concatenate");
    }

    nd::array res = nd::empty(x.get_dim_size() + y.get_dim_size(), x.get_dtype());
    res(irange(0, x.get_dim_size())).val_assign(x);
    res(irange(x.get_dim_size(), res.get_dim_size())).val_assign(y);

    return res;
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
    nd::array result(make_array_memory_block(dt.extended()->get_arrmeta_size(),
                        dt.get_data_size(), dt.get_data_alignment(), &data_ptr));
    // Set the bytes extents
    ((char **)data_ptr)[0] = mm_ptr;
    ((char **)data_ptr)[1] = mm_ptr + mm_size;
    // Set the array arrmeta
    array_preamble *ndo = result.get_ndo();
    ndo->m_type = dt.release();
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = access;
    // Set the bytes arrmeta, telling the system
    // about the memmapped memblock
    bytes_type_arrmeta *ndo_meta =
        reinterpret_cast<bytes_type_arrmeta *>(result.get_arrmeta());
    ndo_meta->blockref = mm.release();
    return result;
}

intptr_t nd::binary_search(const nd::array& n, const char *arrmeta, const char *data)
{
    if (n.get_ndim() == 0) {
        stringstream ss;
        ss << "cannot do a dynd binary_search on array with type " << n.get_type() << " without a leading array dimension";
        throw runtime_error(ss.str());
    }
    const char *n_arrmeta = n.get_arrmeta();
    ndt::type element_tp = n.get_type().at_single(0, &n_arrmeta);
    if (element_tp.get_arrmeta_size() == 0 || n_arrmeta == arrmeta ||
                    memcmp(n_arrmeta, arrmeta, element_tp.get_arrmeta_size()) == 0) {
        // First, a version where the arrmeta is identical, so we can
        // make do with only a single comparison kernel
        comparison_ckernel_builder k_n_less_d;
        make_comparison_kernel(&k_n_less_d, 0,
                        element_tp, n_arrmeta,
                        element_tp, n_arrmeta,
                        comparison_type_sorting_less,
                        &eval::default_eval_context);

        // TODO: support any type of array dimension
        if (n.get_type().get_type_id() != strided_dim_type_id) {
            stringstream ss;
            ss << "TODO: binary_search on array with type " << n.get_type() << " is not implemented";
            throw runtime_error(ss.str());
        }

        const char *n_data = n.get_readonly_originptr();
        intptr_t n_stride = reinterpret_cast<const strided_dim_type_arrmeta *>(n.get_arrmeta())->stride;
        intptr_t first = 0, last = n.get_dim_size();
        while (first < last) {
            intptr_t trial = first + (last - first) / 2;
            const char *trial_data = n_data + trial * n_stride;

            // In order for the data to always match up with the arrmeta, need to have
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
        // Second, a version where the arrmeta are different, so
        // we need to get a kernel for each comparison direction.
        comparison_ckernel_builder k_n_less_d, k_d_less_n;
        make_comparison_kernel(&k_n_less_d, 0,
                        element_tp, n_arrmeta,
                        element_tp, arrmeta,
                        comparison_type_sorting_less,
                        &eval::default_eval_context);
        make_comparison_kernel(&k_d_less_n, 0,
                        element_tp, arrmeta,
                        element_tp, n_arrmeta,
                        comparison_type_sorting_less,
                        &eval::default_eval_context);

        // TODO: support any type of array dimension
        if (n.get_type().get_type_id() != strided_dim_type_id) {
            stringstream ss;
            ss << "TODO: binary_search on array with type " << n.get_type() << " is not implemented";
            throw runtime_error(ss.str());
        }

        const char *n_data = n.get_readonly_originptr();
        intptr_t n_stride = reinterpret_cast<const strided_dim_type_arrmeta *>(n.get_arrmeta())->stride;
        intptr_t first = 0, last = n.get_dim_size();
        while (first < last) {
            intptr_t trial = first + (last - first) / 2;
            const char *trial_data = n_data + trial * n_stride;

            // In order for the data to always match up with the arrmeta, need to have
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
    const groupby_type *gbdt_ext = gbdt.tcast<groupby_type>();
    char *data_ptr = NULL;

    array result(make_array_memory_block(gbdt.extended()->get_arrmeta_size(),
                    gbdt.extended()->get_data_size(), gbdt.extended()->get_data_alignment(), &data_ptr));

    // Set the arrmeta for the data values
    pointer_type_arrmeta *pmeta;
    pmeta = gbdt_ext->get_data_values_pointer_arrmeta(result.get_arrmeta());
    pmeta->offset = 0;
    pmeta->blockref = data_values.get_ndo()->m_data_reference
                    ? data_values.get_ndo()->m_data_reference
                    : &data_values.get_ndo()->m_memblockdata;
    memory_block_incref(pmeta->blockref);
    data_values.get_type().extended()->arrmeta_copy_construct(reinterpret_cast<char *>(pmeta + 1),
                    data_values.get_arrmeta(), &data_values.get_ndo()->m_memblockdata);

    // Set the arrmeta for the by values
    pmeta = gbdt_ext->get_by_values_pointer_arrmeta(result.get_arrmeta());
    pmeta->offset = 0;
    pmeta->blockref = by_values_as_groups.get_ndo()->m_data_reference
                    ? by_values_as_groups.get_ndo()->m_data_reference
                    : &by_values_as_groups.get_ndo()->m_memblockdata;
    memory_block_incref(pmeta->blockref);
    by_values_as_groups.get_type().extended()->arrmeta_copy_construct(reinterpret_cast<char *>(pmeta + 1),
                    by_values_as_groups.get_arrmeta(), &by_values_as_groups.get_ndo()->m_memblockdata);

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

bool nd::is_scalar_avail(const ndt::type &tp, const char *arrmeta,
                      const char *data, const eval::eval_context *ectx)
{
    if (tp.is_scalar()) {
        if (tp.get_type_id() == option_type_id) {
            return tp.tcast<option_type>()->is_avail(arrmeta, data, ectx);
        } else if (tp.get_kind() == expr_kind &&
                   tp.value_type().get_type_id() == option_type_id) {
            nd::array tmp = nd::empty(tp.value_type());
            tmp.val_assign(tp, arrmeta, data, ectx);
            return tmp.get_type().tcast<option_type>()->is_avail(arrmeta, data,
                                                                 ectx);
        } else {
            return true;
        }
    } else {
        return false;
    }
}

void nd::assign_na(const ndt::type &tp, const char *arrmeta, char *data,
                   const eval::eval_context *ectx)
{
    if (tp.get_type_id() == option_type_id) {
        tp.tcast<option_type>()->assign_na(arrmeta, data, ectx);
    } else {
        const ndt::type& dtp = tp.get_dtype().value_type();
        if (dtp.get_type_id() == option_type_id) {
            const arrfunc_type_data *af =
                dtp.tcast<option_type>()->get_assign_na_arrfunc();
            ckernel_builder ckb;
            make_lifted_expr_ckernel(af, &ckb, 0, tp.get_ndim(), tp, arrmeta,
                                     NULL, NULL, NULL, kernel_request_single,
                                     ectx);
            ckernel_prefix *ckp = ckb.get();
            expr_single_t ckp_fn =
                ckp->get_function<expr_single_t>();
            ckp_fn(data, NULL, ckp);
        } else {
            stringstream ss;
            ss << "Cannot assign missing value token NA to dtype " << dtp;
            throw invalid_argument(ss.str());
        }
    }
}


nd::array nd::combine_into_tuple(size_t field_count, const array *field_values)
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

    ndt::type result_type = ndt::make_ctuple(field_types);
    const ctuple_type *fsd = result_type.tcast<ctuple_type>();
    char *data_ptr = NULL;

    array result(make_array_memory_block(fsd->get_arrmeta_size(),
                    fsd->get_data_size(),
                    fsd->get_data_alignment(), &data_ptr));
    // Set the array properties
    result.get_ndo()->m_type = result_type.release();
    result.get_ndo()->m_data_pointer = data_ptr;
    result.get_ndo()->m_data_reference = NULL;
    result.get_ndo()->m_flags = flags;

    // Copy all the needed arrmeta
    const uintptr_t *arrmeta_offsets = fsd->get_arrmeta_offsets_raw();
    for (size_t i = 0; i != field_count; ++i) {
        pointer_type_arrmeta *pmeta;
        pmeta = reinterpret_cast<pointer_type_arrmeta *>(result.get_arrmeta() + arrmeta_offsets[i]);
        pmeta->offset = 0;
        pmeta->blockref = field_values[i].get_ndo()->m_data_reference
                        ? field_values[i].get_ndo()->m_data_reference
                        : &field_values[i].get_ndo()->m_memblockdata;
        memory_block_incref(pmeta->blockref);

        const ndt::type& field_dt = field_values[i].get_type();
        if (field_dt.get_arrmeta_size() > 0) {
            field_dt.extended()->arrmeta_copy_construct(
                            reinterpret_cast<char *>(pmeta + 1),
                            field_values[i].get_arrmeta(),
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
    const char *arrmeta = n.get_arrmeta();
    char *data = n.get_ndo()->m_data_pointer;
    memory_block_data *dataref = NULL;
    uint64_t flags = n.get_ndo()->m_flags;
    while (dt.get_type_id() == pointer_type_id) {
        const pointer_type_arrmeta *md = reinterpret_cast<const pointer_type_arrmeta *>(arrmeta);
        const pointer_type *pd = dt.tcast<pointer_type>();
        dt = pd->get_target_type();
        arrmeta += sizeof(pointer_type_arrmeta);
        data = *reinterpret_cast<char **>(data) + md->offset;
        dataref = md->blockref;
    }
    // Create an array without the pointers
    array result(make_array_memory_block(dt.is_builtin() ? 0 : dt.extended()->get_arrmeta_size()));
    if (!dt.is_builtin()) {
        dt.extended()->arrmeta_copy_construct(result.get_arrmeta(), arrmeta, &n.get_ndo()->m_memblockdata);
    }
    result.get_ndo()->m_type = dt.release();
    result.get_ndo()->m_data_pointer = data;
    result.get_ndo()->m_data_reference = dataref ? dataref : &n.get_ndo()->m_memblockdata;
    memory_block_incref(result.get_ndo()->m_data_reference);
    result.get_ndo()->m_flags = flags;
    return result;
}
*/

