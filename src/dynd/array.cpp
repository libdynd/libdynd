//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/array_iter.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/func/assignment.hpp>
#include <dynd/func/comparison.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/types/view_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/fixed_bytes_type.hpp>
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
#include <dynd/types/categorical_type.hpp>
#include <dynd/types/builtin_type_properties.hpp>
#include <dynd/memblock/memmap_memory_block.hpp>
#include <dynd/view.hpp>

using namespace std;
using namespace dynd;

void nd::array::swap(array &rhs)
{
  m_memblock.swap(rhs.m_memblock);
}

template <class T>
inline typename std::enable_if<is_dynd_scalar<T>::value, memory_block_ptr>::type
make_builtin_scalar_array(const T &value, uint64_t flags)
{
  char *data_ptr = NULL;
  memory_block_ptr result = make_array_memory_block(0, sizeof(T), scalar_align_of<T>::value, &data_ptr);
  *reinterpret_cast<T *>(data_ptr) = value;
  array_preamble *ndo = reinterpret_cast<array_preamble *>(result.get());
  ndo->m_type = reinterpret_cast<ndt::base_type *>(type_id_of<T>::value);
  ndo->data.ptr = data_ptr;
  ndo->data.ref = NULL;
  ndo->m_flags = flags;
  return result;
}

nd::array nd::make_strided_array(const ndt::type &dtp, intptr_t ndim, const intptr_t *shape, int64_t access_flags,
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
    data_size = array_tp.extended()->get_default_data_size();
  }

  memory_block_ptr result;
  char *data_ptr = NULL;
  if (array_tp.get_kind() == memory_kind) {
    result = make_array_memory_block(array_tp.get_arrmeta_size());
    array_tp.extended<ndt::base_memory_type>()->data_alloc(&data_ptr, data_size);
  } else {
    // Allocate the array arrmeta and data in one memory block
    result = make_array_memory_block(array_tp.get_arrmeta_size(), data_size, array_tp.get_data_alignment(), &data_ptr);
  }

  if (array_tp.get_flags() & type_flag_zeroinit) {
    if (array_tp.get_kind() == memory_kind) {
      array_tp.extended<ndt::base_memory_type>()->data_zeroinit(data_ptr, data_size);
    } else {
      memset(data_ptr, 0, data_size);
    }
  }

  // Fill in the preamble arrmeta
  array_preamble *ndo = reinterpret_cast<array_preamble *>(result.get());
  ndo->m_type = array_tp.release();
  ndo->data.ptr = data_ptr;
  ndo->data.ref = NULL;
  ndo->m_flags = access_flags;

  if (!any_variable_dims) {
    // Fill in the array arrmeta with strides and sizes
    fixed_dim_type_arrmeta *meta = reinterpret_cast<fixed_dim_type_arrmeta *>(ndo + 1);
    // Use the default construction to handle the uniform_tp's arrmeta
    intptr_t stride = dtp.get_data_size();
    if (stride == 0) {
      stride = dtp.extended()->get_default_data_size();
    }
    if (!dtp.is_builtin()) {
      dtp.extended()->arrmeta_default_construct(reinterpret_cast<char *>(meta + ndim), true);
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
    ndo->m_type->arrmeta_default_construct(meta, true);
  }

  return array(result);
}

nd::array nd::make_strided_array_from_data(const ndt::type &uniform_tp, intptr_t ndim, const intptr_t *shape,
                                           const intptr_t *strides, int64_t access_flags, char *data_ptr,
                                           const memory_block_ptr &data_reference, char **out_uniform_arrmeta)
{
  if (out_uniform_arrmeta == NULL && !uniform_tp.is_builtin() && uniform_tp.extended()->get_arrmeta_size() > 0) {
    stringstream ss;
    ss << "Cannot make a strided array with type " << uniform_tp << " from a preexisting data pointer";
    throw runtime_error(ss.str());
  }

  ndt::type array_type = ndt::make_fixed_dim(ndim, shape, uniform_tp);

  // Allocate the array arrmeta and data in one memory block
  memory_block_ptr result = make_array_memory_block(array_type.get_arrmeta_size());

  // Fill in the preamble arrmeta
  array_preamble *ndo = reinterpret_cast<array_preamble *>(result.get());
  ndo->m_type = array_type.release();
  ndo->data.ptr = data_ptr;
  ndo->data.ref = data_reference.get();
  memory_block_incref(ndo->data.ref);
  ndo->m_flags = access_flags;

  // Fill in the array arrmeta with the shape and strides
  fixed_dim_type_arrmeta *meta = reinterpret_cast<fixed_dim_type_arrmeta *>(ndo + 1);
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

nd::array nd::make_pod_array(const ndt::type &pod_dt, const void *data)
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
  memory_block_ptr result = make_array_memory_block(0, size, pod_dt.get_data_alignment(), &data_ptr);

  // Fill in the preamble arrmeta
  array_preamble *ndo = reinterpret_cast<array_preamble *>(result.get());
  if (pod_dt.is_builtin()) {
    ndo->m_type = reinterpret_cast<const ndt::base_type *>(pod_dt.get_type_id());
  } else {
    ndo->m_type = pod_dt.extended();
    base_type_incref(ndo->m_type);
  }
  ndo->data.ptr = data_ptr;
  ndo->data.ref = NULL;
  ndo->m_flags = nd::read_access_flag | nd::immutable_access_flag;

  memcpy(data_ptr, data, size);

  return nd::array(result);
}

nd::array nd::make_bytes_array(const char *data, size_t len, size_t alignment)
{
  char *data_ptr = NULL, *bytes_data_ptr;
  ndt::type dt = ndt::bytes_type::make(alignment);
  nd::array result(make_array_memory_block(dt.extended()->get_arrmeta_size(), dt.get_data_size() + len + alignment - 1,
                                           dt.get_data_alignment(), &data_ptr));
  // Set the string extents
  bytes_data_ptr = inc_to_alignment(data_ptr + dt.get_data_size(), alignment);
  reinterpret_cast<bytes *>(data_ptr)->assign(bytes_data_ptr, len);
  // Copy the string data
  memcpy(bytes_data_ptr, data, len);
  // Set the array arrmeta
  array_preamble *ndo = result.get_ndo();
  ndo->m_type = dt.release();
  ndo->data.ptr = data_ptr;
  ndo->data.ref = NULL;
  ndo->m_flags = nd::read_access_flag | nd::immutable_access_flag;
  // Set the bytes arrmeta, telling the system that the bytes data was embedded
  // in the array memory
  bytes_type_arrmeta *ndo_meta = reinterpret_cast<bytes_type_arrmeta *>(result.get_arrmeta());
  ndo_meta->blockref = NULL;
  return result;
}

nd::array nd::make_string_array(const char *str, size_t len, string_encoding_t DYND_UNUSED(encoding),
                                uint64_t access_flags)
{
  char *data_ptr = NULL, *string_ptr;
  ndt::type dt = ndt::string_type::make();
  nd::array result(make_array_memory_block(dt.extended()->get_arrmeta_size(), dt.get_data_size() + len,
                                           dt.get_data_alignment(), &data_ptr));
  // Set the string extents
  string_ptr = data_ptr + dt.get_data_size();
  reinterpret_cast<string *>(data_ptr)->assign(string_ptr, len);
  // Copy the string data
  memcpy(string_ptr, str, len);
  // Set the array arrmeta
  array_preamble *ndo = result.get_ndo();
  ndo->m_type = dt.release();
  ndo->data.ptr = data_ptr;
  ndo->data.ref = NULL;
  ndo->m_flags = access_flags;
  // Set the string arrmeta, telling the system that the string data was
  // embedded in the array memory
  string_type_arrmeta *ndo_meta = reinterpret_cast<string_type_arrmeta *>(result.get_arrmeta());
  ndo_meta->blockref = NULL;
  return result;
}

nd::array nd::make_strided_string_array(const char *const *cstr_array, size_t array_size)
{
  size_t total_string_length = 0;
  for (size_t i = 0; i != array_size; ++i) {
    total_string_length += strlen(cstr_array[i]);
  }

  char *data_ptr = NULL, *string_ptr;
  string *string_arr_ptr;
  ndt::type stp = ndt::string_type::make();
  ndt::type tp = ndt::make_fixed_dim(array_size, stp);
  nd::array result(make_array_memory_block(tp.extended()->get_arrmeta_size(),
                                           array_size * stp.get_data_size() + total_string_length,
                                           tp.get_data_alignment(), &data_ptr));
  // Set the array arrmeta
  array_preamble *ndo = result.get_ndo();
  ndo->m_type = tp.release();
  ndo->data.ptr = data_ptr;
  ndo->data.ref = NULL;
  ndo->m_flags = nd::read_access_flag | nd::immutable_access_flag;
  // Get the allocator for the output string type
  fixed_dim_type_arrmeta *md = reinterpret_cast<fixed_dim_type_arrmeta *>(result.get_arrmeta());
  md->dim_size = array_size;
  md->stride = stp.get_data_size();
  string_arr_ptr = reinterpret_cast<string *>(data_ptr);
  string_ptr = data_ptr + array_size * stp.get_data_size();
  for (size_t i = 0; i < array_size; ++i) {
    size_t size = strlen(cstr_array[i]);
    memcpy(string_ptr, cstr_array[i], size);
    string_arr_ptr->assign(string_ptr, size);
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
  string *string_arr_ptr;
  ndt::type stp = ndt::string_type::make();
  ndt::type tp = ndt::make_fixed_dim(array_size, stp);
  nd::array result(make_array_memory_block(tp.extended()->get_arrmeta_size(),
                                           array_size * stp.get_data_size() + total_string_length,
                                           tp.get_data_alignment(), &data_ptr));
  // Set the array arrmeta
  array_preamble *ndo = result.get_ndo();
  ndo->m_type = tp.release();
  ndo->data.ptr = data_ptr;
  ndo->data.ref = NULL;
  ndo->m_flags = nd::read_access_flag | nd::immutable_access_flag;
  // Get the allocator for the output string type
  fixed_dim_type_arrmeta *md = reinterpret_cast<fixed_dim_type_arrmeta *>(result.get_arrmeta());
  md->dim_size = array_size;
  md->stride = stp.get_data_size();
  string_arr_ptr = reinterpret_cast<string *>(data_ptr);
  string_ptr = data_ptr + array_size * stp.get_data_size();
  for (size_t i = 0; i < array_size; ++i) {
    size_t size = str_array[i]->size();
    memcpy(string_ptr, str_array[i]->data(), size);
    string_arr_ptr->assign(string_ptr, size);
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
static nd::array make_array_clone_with_new_type(const nd::array &n, const ndt::type &new_dt)
{
  nd::array result(shallow_copy_array_memory_block(n.get_memblock()));
  array_preamble *preamble = result.get_ndo();
  // Swap in the type
  if (!preamble->is_builtin_type()) {
    base_type_decref(preamble->m_type);
  }
  preamble->m_type = new_dt.extended();
  if (!new_dt.is_builtin()) {
    base_type_incref(preamble->m_type);
  }
  return result;
}

// Constructors from C++ scalars
nd::array::array(bool1 value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(bool value)
    : m_memblock(make_builtin_scalar_array(bool1(value), nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(signed char value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(short value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(int value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(long value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(long long value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(const int128 &value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(unsigned char value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(unsigned short value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(unsigned int value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(unsigned long value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(unsigned long long value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(const uint128 &value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(float16 value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(float value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(double value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(const float128 &value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(dynd::complex<float> value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(dynd::complex<double> value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(std::complex<float> value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(std::complex<double> value)
    : m_memblock(make_builtin_scalar_array(value, nd::read_access_flag | nd::immutable_access_flag))
{
}
nd::array::array(const std::string &value)
{
  array temp = make_string_array(value.c_str(), value.size(), string_encoding_utf_8,
                                 nd::read_access_flag | nd::immutable_access_flag);
  temp.swap(*this);
}
nd::array::array(const char *cstr)
{
  array temp =
      make_string_array(cstr, strlen(cstr), string_encoding_utf_8, nd::read_access_flag | nd::immutable_access_flag);
  temp.swap(*this);
}
nd::array::array(const char *str, size_t size)
{
  array temp = make_string_array(str, size, string_encoding_utf_8, nd::read_access_flag | nd::immutable_access_flag);
  temp.swap(*this);
}
nd::array::array(const ndt::type &tp)
{
  array temp(nd::empty(ndt::make_type()));
  temp.swap(*this);
  ndt::type(tp).swap(reinterpret_cast<type_type_data *>(get_ndo()->data.ptr)->tp);
  get_ndo()->m_flags = nd::read_access_flag | nd::immutable_access_flag;
}

nd::array nd::array_rw(bool1 value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(bool value)
{
  return nd::array(make_builtin_scalar_array(bool1(value), nd::readwrite_access_flags));
}
nd::array nd::array_rw(signed char value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(short value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(int value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(long value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(long long value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(const int128 &value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(unsigned char value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(unsigned short value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(unsigned int value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(unsigned long value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(unsigned long long value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(const uint128 &value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(float16 value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(float value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(double value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(const float128 &value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(dynd::complex<float> value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(dynd::complex<double> value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(std::complex<float> value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(std::complex<double> value)
{
  return nd::array(make_builtin_scalar_array(value, nd::readwrite_access_flags));
}
nd::array nd::array_rw(const std::string &value)
{
  return make_string_array(value.c_str(), value.size(), string_encoding_utf_8, nd::readwrite_access_flags);
}
/** Construct a string from a NULL-terminated UTF8 string */
nd::array nd::array_rw(const char *cstr)
{
  return make_string_array(cstr, strlen(cstr), string_encoding_utf_8, nd::readwrite_access_flags);
}
/** Construct a string from a UTF8 buffer and specified buffer size */
nd::array nd::array_rw(const char *str, size_t size)
{
  return make_string_array(str, size, string_encoding_utf_8, nd::readwrite_access_flags);
}
nd::array nd::array_rw(const ndt::type &tp)
{
  array temp = array(nd::empty(ndt::make_type()));
  ndt::type(tp).swap(reinterpret_cast<type_type_data *>(temp.get_ndo()->data.ptr)->tp);
  return temp;
}

nd::array nd::detail::make_from_vec<ndt::type>::make(const std::vector<ndt::type> &vec)
{
  ndt::type dt = ndt::make_fixed_dim(vec.size(), ndt::make_type());
  char *data_ptr = NULL;
  array result(make_array_memory_block(dt.extended()->get_arrmeta_size(), sizeof(type_type_data) * vec.size(),
                                       dt.get_data_alignment(), &data_ptr));
  // The main array arrmeta
  array_preamble *preamble = result.get_ndo();
  preamble->data.ptr = data_ptr;
  preamble->data.ref = NULL;
  preamble->m_type = dt.release();
  preamble->m_flags = read_access_flag | immutable_access_flag;
  // The arrmeta for the strided and type parts of the type
  fixed_dim_type_arrmeta *sa_md = reinterpret_cast<fixed_dim_type_arrmeta *>(result.get_arrmeta());
  sa_md->dim_size = vec.size();
  sa_md->stride = vec.empty() ? 0 : sizeof(type_type_data);
  // The data
  type_type_data *data = reinterpret_cast<type_type_data *>(data_ptr);
  for (size_t i = 0, i_end = vec.size(); i != i_end; ++i) {
    data[i].tp = ndt::type(vec[i]).release();
  }
  return result;
}

nd::array nd::detail::make_from_vec<std::string>::make(const std::vector<std::string> &vec)
{
  // Constructor detail for making an array from a vector of strings
  size_t total_string_size = 0;
  for (size_t i = 0, i_end = vec.size(); i != i_end; ++i) {
    total_string_size += vec[i].size();
  }

  ndt::type dt = ndt::make_fixed_dim(vec.size(), ndt::string_type::make());
  char *data_ptr = NULL;
  // Make an array memory block which contains both the string pointers and
  // the string data
  array result(make_array_memory_block(dt.extended()->get_arrmeta_size(),
                                       sizeof(string) * vec.size() + total_string_size, dt.get_data_alignment(),
                                       &data_ptr));
  char *string_ptr = data_ptr + sizeof(string) * vec.size();
  // The main array arrmeta
  array_preamble *preamble = result.get_ndo();
  preamble->data.ptr = data_ptr;
  preamble->data.ref = NULL;
  preamble->m_type = dt.release();
  preamble->m_flags = read_access_flag | immutable_access_flag;
  // The arrmeta for the fixed_dim and string parts of the type
  fixed_dim_type_arrmeta *sa_md = reinterpret_cast<fixed_dim_type_arrmeta *>(result.get_arrmeta());
  sa_md->dim_size = vec.size();
  sa_md->stride = vec.empty() ? 0 : sizeof(string);
  string_type_arrmeta *s_md = reinterpret_cast<string_type_arrmeta *>(sa_md + 1);
  s_md->blockref = NULL;
  // The string pointers and data
  string *data = reinterpret_cast<string *>(data_ptr);
  for (size_t i = 0, i_end = vec.size(); i != i_end; ++i) {
    size_t size = vec[i].size();
    memcpy(string_ptr, vec[i].data(), size);
    data[i].assign(string_ptr, size);
    string_ptr += size;
  }
  return result;
}

namespace {
static void as_storage_type(const ndt::type &dt, intptr_t DYND_UNUSED(arrmeta_offset), void *DYND_UNUSED(self),
                            ndt::type &out_transformed_tp, bool &out_was_transformed)
{
  // If the type is a simple POD, switch it to a bytes type. Otherwise, keep it
  // the same so that the arrmeta layout is identical.
  if (dt.is_scalar() && dt.get_type_id() != pointer_type_id) {
    const ndt::type &storage_dt = dt.storage_type();
    if (storage_dt.is_builtin()) {
      out_transformed_tp = ndt::make_fixed_bytes(storage_dt.get_data_size(), storage_dt.get_data_alignment());
      out_was_transformed = true;
    } else if (storage_dt.is_pod() && storage_dt.extended()->get_arrmeta_size() == 0) {
      out_transformed_tp = ndt::make_fixed_bytes(storage_dt.get_data_size(), storage_dt.get_data_alignment());
      out_was_transformed = true;
    } else if (storage_dt.get_type_id() == string_type_id) {
      out_transformed_tp =
          ndt::bytes_type::make(static_cast<const ndt::string_type *>(storage_dt.extended())->get_target_alignment());
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
    dt.extended()->transform_child_types(&as_storage_type, 0, NULL, out_transformed_tp, out_was_transformed);
  }
}
} // anonymous namespace

nd::array nd::array::storage() const
{
  ndt::type storage_dt = get_type();
  bool was_transformed = false;
  as_storage_type(get_type(), 0, NULL, storage_dt, was_transformed);
  if (was_transformed) {
    return make_array_clone_with_new_type(*this, storage_dt);
  } else {
    return *this;
  }
}

nd::array &nd::array::underlying()
{
  if (get_type().get_type_id() == array_type_id) {
    return *reinterpret_cast<array *>(get_data());
  }

  return *this;
}

const nd::array &nd::array::underlying() const
{
  if (get_type().get_type_id() == array_type_id) {
    return *reinterpret_cast<const array *>(get_data());
  }

  return *this;
}

nd::array nd::array::at_array(intptr_t nindices, const irange *indices, bool collapse_leading) const
{
  if (!get_type().is_indexable()) {
    if (nindices != 0) {
      throw too_many_indices(get_type(), nindices, 0);
    }
    return *this;
  } else {
    ndt::type this_dt(get_ndo()->m_type, true);
    ndt::type dt = get_ndo()->m_type->apply_linear_index(nindices, indices, 0, this_dt, collapse_leading);
    array result;
    if (!dt.is_builtin()) {
      result.set(make_array_memory_block(dt.extended()->get_arrmeta_size()));
      result.get_ndo()->m_type = dt.extended();
      base_type_incref(result.get_ndo()->m_type);
    } else {
      result.set(make_array_memory_block(0));
      result.get_ndo()->m_type = reinterpret_cast<const ndt::base_type *>(dt.get_type_id());
    }
    result.get_ndo()->data.ptr = get_ndo()->data.ptr;
    if (get_ndo()->data.ref) {
      result.get_ndo()->data.ref = get_ndo()->data.ref;
    } else {
      // If the data reference is NULL, the data is embedded in the array itself
      result.get_ndo()->data.ref = m_memblock.get();
    }
    memory_block_incref(result.get_ndo()->data.ref);
    intptr_t offset = get_ndo()->m_type->apply_linear_index(nindices, indices, get_arrmeta(), dt, result.get_arrmeta(),
                                                            m_memblock.get(), 0, this_dt, collapse_leading,
                                                            &result.get_ndo()->data.ptr, &result.get_ndo()->data.ref);
    result.get_ndo()->data.ptr += offset;
    result.get_ndo()->m_flags = get_ndo()->m_flags;
    return result;
  }
}

void nd::array::val_assign(const array &rhs, const eval::eval_context *ectx) const
{
  // Verify read access permission
  if (!(rhs.get_flags() & read_access_flag)) {
    throw runtime_error("tried to read from a dynd array that is not readable");
  }

  /*
    const ndt::type &dst_tp = get_type();
    const ndt::type &src_tp = rhs.get_type();
    if (dst_tp.is_builtin() && src_tp.is_builtin()) {
      nd::assign(rhs, kwds("dst", *this));
    }
  */

  typed_data_assign(get_type(), get_arrmeta(), get_readwrite_originptr(), rhs.get_type(), rhs.get_arrmeta(),
                    rhs.get_readonly_originptr(), ectx);
}

void nd::array::val_assign(const ndt::type &rhs_dt, const char *rhs_arrmeta, const char *rhs_data,
                           const eval::eval_context *ectx) const
{
  typed_data_assign(get_type(), get_arrmeta(), get_readwrite_originptr(), rhs_dt, rhs_arrmeta, rhs_data, ectx);
}

void nd::array::flag_as_immutable()
{
  // If it's already immutable, everything's ok
  if ((get_flags() & immutable_access_flag) != 0) {
    return;
  }

  // Check that nobody else is peeking into our data
  bool ok = true;
  if (m_memblock.get()->m_use_count != 1) {
    // More than one reference to the array itself
    ok = false;
  } else if (get_ndo()->data.ref != NULL && (get_ndo()->data.ref->m_use_count != 1 ||
                                             !(get_ndo()->data.ref->m_type == fixed_size_pod_memory_block_type ||
                                               get_ndo()->data.ref->m_type == pod_memory_block_type))) {
    // More than one reference to the array's data, or the reference is to
    // something
    // other than a memblock owning its data, such as an external memblock.
    ok = false;
  } else if (!get_ndo()->is_builtin_type() && !get_ndo()->m_type->is_unique_data_owner(get_arrmeta())) {
    ok = false;
  }

  if (ok) {
    // Finalize any allocated data in the arrmeta
    if (!is_builtin_type(get_ndo()->m_type)) {
      get_ndo()->m_type->arrmeta_finalize_buffers(get_arrmeta());
    }
    // Clear the write flag, and set the immutable flag
    get_ndo()->m_flags = (get_ndo()->m_flags & ~(uint64_t)write_access_flag) | immutable_access_flag;
  } else {
    stringstream ss;
    ss << "Unable to flag array of type " << get_type() << " as immutable, because ";
    ss << "it does not uniquely own all of its data";
    throw runtime_error(ss.str());
  }
}

nd::array nd::array::p(const char *property_name) const
{
  if (!is_null()) {
    ndt::type dt = get_type();
    const std::pair<std::string, gfunc::callable> *properties;
    size_t count;
    if (!dt.is_builtin()) {
      dt.extended()->get_dynamic_array_properties(&properties, &count);
    } else {
      get_builtin_type_dynamic_array_properties(dt.get_type_id(), &properties, &count);
    }
    // TODO: We probably want to make some kind of acceleration structure for
    // the name lookup
    if (count > 0) {
      for (size_t i = 0; i < count; ++i) {
        if (properties[i].first == property_name) {
          return properties[i].second.call(*this);
        }
      }
    }
  }

  stringstream ss;
  ss << "dynd array does not have property " << property_name;
  throw runtime_error(ss.str());
}

nd::array nd::array::p(const std::string &property_name) const
{
  if (!is_null()) {
    ndt::type dt = get_type();
    const std::pair<std::string, gfunc::callable> *properties;
    size_t count;
    if (!dt.is_builtin()) {
      dt.extended()->get_dynamic_array_properties(&properties, &count);
    } else {
      get_builtin_type_dynamic_array_properties(dt.get_type_id(), &properties, &count);
    }
    // TODO: We probably want to make some kind of acceleration structure for
    // the name lookup
    if (count > 0) {
      for (size_t i = 0; i < count; ++i) {
        if (properties[i].first == property_name) {
          return properties[i].second.call(*this);
        }
      }
    }
  }

  stringstream ss;
  ss << "dynd array does not have property " << property_name;
  throw runtime_error(ss.str());
}

const gfunc::callable &nd::array::find_dynamic_function(const char *function_name) const
{
  ndt::type dt = get_type();
  if (!dt.is_builtin()) {
    const std::pair<std::string, gfunc::callable> *properties;
    size_t count;
    dt.extended()->get_dynamic_array_functions(&properties, &count);
    // TODO: We probably want to make some kind of acceleration structure for
    // the name lookup
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
  const ndt::type &current_tp = get_type();
  if (!current_tp.is_expression()) {
    return *this;
  } else {
    // Create a canonical type for the result
    const ndt::type &dt = current_tp.get_canonical_type();
    array result(nd::empty(dt));
    if (dt.get_type_id() == fixed_dim_type_id) {
      // Reorder strides of output strided dimensions in a KEEPORDER fashion
      dt.extended<ndt::fixed_dim_type>()->reorder_default_constructed_strides(result.get_arrmeta(), get_type(),
                                                                              get_arrmeta());
    }
    result.val_assign(*this, ectx);
    return result;
  }
}

nd::array nd::array::eval_immutable(const eval::eval_context *ectx) const
{
  const ndt::type &current_tp = get_type();
  if ((get_access_flags() & immutable_access_flag) && !current_tp.is_expression()) {
    return *this;
  } else {
    // Create a canonical type for the result
    const ndt::type &dt = current_tp.get_canonical_type();
    array result(nd::empty(dt));
    if (dt.get_type_id() == fixed_dim_type_id) {
      // Reorder strides of output strided dimensions in a KEEPORDER fashion
      dt.extended<ndt::fixed_dim_type>()->reorder_default_constructed_strides(result.get_arrmeta(), get_type(),
                                                                              get_arrmeta());
    }
    result.val_assign(*this, ectx);
    result.get_ndo()->m_flags = immutable_access_flag | read_access_flag;
    return result;
  }
}

nd::array nd::array::eval_copy(uint32_t access_flags, const eval::eval_context *ectx) const
{
  const ndt::type &current_tp = get_type();
  const ndt::type &dt = current_tp.get_canonical_type();
  array result(nd::empty(dt));
  if (dt.get_type_id() == fixed_dim_type_id) {
    // Reorder strides of output strided dimensions in a KEEPORDER fashion
    dt.extended<ndt::fixed_dim_type>()->reorder_default_constructed_strides(result.get_arrmeta(), get_type(),
                                                                            get_arrmeta());
  }
  result.val_assign(*this, ectx);
  // If the access_flags are 0, use the defaults
  access_flags = access_flags ? access_flags : (int32_t)nd::default_access_flags;
  // If the access_flags are just readonly, add immutable
  // because we just created a unique instance
  access_flags =
      (access_flags != nd::read_access_flag) ? access_flags : (nd::read_access_flag | nd::immutable_access_flag);
  result.get_ndo()->m_flags = access_flags;
  return result;
}

nd::array nd::array::to_host() const
{
  array result = empty(get_type().without_memory_type());
  result.val_assign(*this);

  return result;
}

#ifdef DYND_CUDA

nd::array nd::array::to_cuda_host(unsigned int cuda_host_flags) const
{
  array result = empty(make_cuda_host(get_type().without_memory_type(), cuda_host_flags));
  result.val_assign(*this);

  return result;
}

nd::array nd::array::to_cuda_device() const
{
  array result = empty(make_cuda_device(get_type().without_memory_type()));
  result.val_assign(*this);

  return result;
}

#endif // DYND_CUDA

bool nd::array::is_missing() const
{
  ndt::type tp = get_type();
  if (tp.get_type_id() == option_type_id) {
    return !tp.extended<ndt::option_type>()->is_avail(get_arrmeta(), get_readonly_originptr(),
                                                      &eval::default_eval_context);
  }

  return false;
}

void nd::array::assign_na()
{
  ndt::type tp = get_type();
  if (tp.get_type_id() == option_type_id) {
    tp.extended<ndt::option_type>()->assign_na(get_arrmeta(), get_readwrite_originptr(), &eval::default_eval_context);
  }
}

bool nd::array::equals_exact(const array &rhs) const
{
  if (get_ndo() == rhs.get_ndo()) {
    return true;
  } else if (get_type() != rhs.get_type()) {
    return false;
  } else if (get_ndim() == 0) {
    return (*this == rhs).as<bool>();
  } else if (get_type().get_type_id() == var_dim_type_id) {
    // If there's a leading var dimension, convert it to strided and compare
    // (Note: this is an inefficient hack)
    ndt::type tp = ndt::make_fixed_dim(get_shape()[0], get_type().extended<ndt::base_dim_type>()->get_element_type());
    return nd::view(*this, tp).equals_exact(nd::view(rhs, tp));
  } else {
    // First compare the shape, to avoid triggering an exception in common cases
    size_t ndim = get_ndim();
    if (ndim == 1 && get_dim_size() == 0 && rhs.get_dim_size() == 0) {
      return true;
    }
    dimvector shape0(ndim), shape1(ndim);
    get_shape(shape0.get());
    rhs.get_shape(shape1.get());
    if (memcmp(shape0.get(), shape1.get(), ndim * sizeof(intptr_t)) != 0) {
      return false;
    }
    try
    {
      array_iter<0, 2> iter(*this, rhs);
      if (!iter.empty()) {
        do {
          const char *const src[2] = {iter.data<0>(), iter.data<1>()};
          ndt::type tp[2] = {iter.get_uniform_dtype<0>(), iter.get_uniform_dtype<1>()};
          const char *arrmeta[2] = {iter.arrmeta<0>(), iter.arrmeta<1>()};
          ndt::type dst_tp = ndt::type::make<bool1>();
          if ((*not_equal::get().get())(dst_tp, 2, tp, arrmeta, const_cast<char *const *>(src), 0, NULL,
                                        std::map<std::string, ndt::type>()).as<bool>()) {
            return false;
          }
        } while (iter.next());
      }
      return true;
    }
    catch (const broadcast_error &)
    {
      // If there's a broadcast error in a variable-sized dimension, return
      // false for it too
      return false;
    }
  }
}

nd::array nd::array::cast(const ndt::type &tp) const
{
  // Use the ucast function specifying to replace all dimensions
  return ucast(tp, get_type().get_ndim());
}

namespace {
struct cast_dtype_extra {
  cast_dtype_extra(const ndt::type &tp, size_t ru) : replacement_tp(tp), replace_ndim(ru), out_can_view_data(true)
  {
  }
  const ndt::type &replacement_tp;
  intptr_t replace_ndim;
  bool out_can_view_data;
};
static void cast_dtype(const ndt::type &dt, intptr_t DYND_UNUSED(arrmeta_offset), void *extra,
                       ndt::type &out_transformed_tp, bool &out_was_transformed)
{
  cast_dtype_extra *e = reinterpret_cast<cast_dtype_extra *>(extra);
  intptr_t replace_ndim = e->replace_ndim;
  if (dt.get_ndim() > replace_ndim) {
    dt.extended()->transform_child_types(&cast_dtype, 0, extra, out_transformed_tp, out_was_transformed);
  } else {
    if (replace_ndim > 0) {
      // If the dimension we're replacing doesn't change, then
      // avoid creating the convert type at this level
      if (dt.get_type_id() == e->replacement_tp.get_type_id()) {
        bool can_keep_dim = false;
        ndt::type child_dt, child_replacement_tp;
        switch (dt.get_type_id()) {
        /*
                case cfixed_dim_type_id: {
                  const cfixed_dim_type *dt_fdd =
           dt.extended<cfixed_dim_type>();
                  const cfixed_dim_type *r_fdd = static_cast<const
           cfixed_dim_type *>(
                      e->replacement_tp.extended());
                  if (dt_fdd->get_fixed_dim_size() ==
           r_fdd->get_fixed_dim_size() &&
                      dt_fdd->get_fixed_stride() == r_fdd->get_fixed_stride()) {
                    can_keep_dim = true;
                    child_dt = dt_fdd->get_element_type();
                    child_replacement_tp = r_fdd->get_element_type();
                  }
                }
        */
        case var_dim_type_id: {
          const ndt::base_dim_type *dt_budd = dt.extended<ndt::base_dim_type>();
          const ndt::base_dim_type *r_budd = static_cast<const ndt::base_dim_type *>(e->replacement_tp.extended());
          can_keep_dim = true;
          child_dt = dt_budd->get_element_type();
          child_replacement_tp = r_budd->get_element_type();
          break;
        }
        default:
          break;
        }
        if (can_keep_dim) {
          cast_dtype_extra extra_child(child_replacement_tp, replace_ndim - 1);
          dt.extended()->transform_child_types(&cast_dtype, 0, &extra_child, out_transformed_tp, out_was_transformed);
          return;
        }
      }
    }
    out_transformed_tp = ndt::convert_type::make(e->replacement_tp, dt);
    // Only flag the transformation if this actually created a convert type
    if (out_transformed_tp.extended() != e->replacement_tp.extended()) {
      out_was_transformed = true;
      e->out_can_view_data = false;
    }
  }
}
} // anonymous namespace

nd::array nd::array::ucast(const ndt::type &scalar_tp, intptr_t replace_ndim) const
{
  // This creates a type which has a convert type for every scalar of different
  // type.
  // The result has the exact same arrmeta and data, so we just have to swap in
  // the new
  // type in a shallow copy.
  ndt::type replaced_tp;
  bool was_transformed = false;
  cast_dtype_extra extra(scalar_tp, replace_ndim);
  cast_dtype(get_type(), 0, &extra, replaced_tp, was_transformed);
  if (was_transformed) {
    return make_array_clone_with_new_type(*this, replaced_tp);
  } else {
    return *this;
  }
}

nd::array nd::array::view(const ndt::type &tp) const
{
  return nd::view(*this, tp);
}

nd::array nd::array::uview(const ndt::type &uniform_dt, intptr_t replace_ndim) const
{
  // Use the view function specifying to replace all dimensions
  return view(get_type().with_replaced_dtype(uniform_dt, replace_ndim));
}

nd::array nd::array::adapt(const ndt::type &tp, const std::string &adapt_op)
{
  return uview(ndt::adapt_type::make(get_dtype(), tp, adapt_op), 0);
}

namespace {
struct permute_dims_data {
  intptr_t ndim, i;
  const intptr_t *axes;
  char *arrmeta;
};
static void permute_type_dims(const ndt::type &tp, intptr_t arrmeta_offset, void *extra, ndt::type &out_transformed_tp,
                              bool &out_was_transformed)
{
  permute_dims_data *pdd = reinterpret_cast<permute_dims_data *>(extra);
  intptr_t i = pdd->i;
  if (pdd->axes[i] == i) {
    // Stationary axis
    if (pdd->i == pdd->ndim - 1) {
      // No more perm dimensions left, leave type as is
      out_transformed_tp = tp;
    } else {
      if (tp.get_kind() == dim_kind) {
        ++pdd->i;
      }
      tp.extended()->transform_child_types(&permute_type_dims, arrmeta_offset, extra, out_transformed_tp,
                                           out_was_transformed);
      pdd->i = i;
    }
  } else {
    // Find the smallest interval of mutually permuted axes
    intptr_t max_i = pdd->axes[i], loop_i = i + 1;
    while (loop_i <= max_i && loop_i < pdd->ndim) {
      if (pdd->axes[loop_i] > max_i) {
        max_i = pdd->axes[loop_i];
      }
      ++loop_i;
    }
    // We must have enough consecutive strided dimensions
    if (tp.get_strided_ndim() < max_i - i + 1) {
      stringstream ss;
      ss << "Cannot permute non-strided dimensions in type " << tp;
      throw invalid_argument(ss.str());
    }
    ndt::type subtp = tp.extended<ndt::base_dim_type>()->get_element_type();
    for (loop_i = i + 1; loop_i <= max_i; ++loop_i) {
      subtp = subtp.extended<ndt::base_dim_type>()->get_element_type();
    }
    intptr_t perm_ndim = max_i - i + 1;
    // If there are more permutation axes left, process the subtype
    if (max_i < pdd->ndim - 1) {
      pdd->i = max_i + 1;
      tp.extended()->transform_child_types(&permute_type_dims,
                                           arrmeta_offset + perm_ndim * sizeof(fixed_dim_type_arrmeta), extra, subtp,
                                           out_was_transformed);
    }
    // Apply the permutation
    dimvector shape(perm_ndim), permuted_shape(perm_ndim);
    shortvector<size_stride_t> perm_arrmeta(perm_ndim);
    size_stride_t *original_arrmeta = reinterpret_cast<size_stride_t *>(pdd->arrmeta + arrmeta_offset);
    memcpy(perm_arrmeta.get(), original_arrmeta, perm_ndim * sizeof(size_stride_t));
    tp.extended()->get_shape(max_i - i + 1, 0, shape.get(), NULL, NULL);
    for (loop_i = i; loop_i <= max_i; ++loop_i) {
      intptr_t srcidx = pdd->axes[loop_i] - i;
      permuted_shape[loop_i - i] = shape[srcidx];
      original_arrmeta[loop_i - i] = perm_arrmeta[srcidx];
    }
    out_transformed_tp = ndt::make_type(perm_ndim, permuted_shape.get(), subtp);
    out_was_transformed = true;
  }
}
} // anonymous namespace

nd::array nd::array::permute(intptr_t ndim, const intptr_t *axes) const
{
  if (ndim > get_ndim()) {
    stringstream ss;
    ss << "Too many dimensions provided for axis permutation, got " << ndim << " for type " << get_type();
    throw invalid_argument(ss.str());
  }
  if (!is_valid_perm(ndim, axes)) {
    stringstream ss;
    ss << "Invalid permutation provided to dynd axis permute: [";
    for (intptr_t i = 0; i < ndim; ++i) {
      ss << axes[i] << (i != ndim - 1 ? " " : "");
    }
    ss << "]";
    throw invalid_argument(ss.str());
  }

  // Make a shallow copy of the arrmeta. When we permute the type,
  // its arrmeta has identical structure, so we can fix it up
  // while we're transforming the type.
  nd::array res(shallow_copy_array_memory_block(get_memblock()));

  ndt::type transformed_tp;
  bool was_transformed = false;
  permute_dims_data pdd;
  pdd.ndim = ndim;
  pdd.i = 0;
  pdd.axes = axes;
  pdd.arrmeta = res.get_arrmeta();
  permute_type_dims(get_type(), 0, &pdd, transformed_tp, was_transformed);

  // We can now substitute our transformed type into
  // the result array
  base_type_decref(res.get_ndo()->m_type);
  res.get_ndo()->m_type = transformed_tp.extended();
  base_type_incref(res.get_ndo()->m_type);

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
  replace_compatible_dtype_extra(const ndt::type &tp, intptr_t replace_ndim_)
      : replacement_tp(tp), replace_ndim(replace_ndim_)
  {
  }
  const ndt::type &replacement_tp;
  intptr_t replace_ndim;
};
static void replace_compatible_dtype(const ndt::type &tp, intptr_t DYND_UNUSED(arrmeta_offset), void *extra,
                                     ndt::type &out_transformed_tp, bool &out_was_transformed)
{
  const replace_compatible_dtype_extra *e = reinterpret_cast<const replace_compatible_dtype_extra *>(extra);
  const ndt::type &replacement_tp = e->replacement_tp;
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
      out_was_transformed = true;
    }
  } else {
    tp.extended()->transform_child_types(&replace_compatible_dtype, 0, extra, out_transformed_tp, out_was_transformed);
  }
}
} // anonymous namespace

nd::array nd::array::replace_dtype(const ndt::type &replacement_tp, intptr_t replace_ndim) const
{
  // This creates a type which swaps in the new dtype for
  // the existing one. It raises an error if the data layout
  // is incompatible
  ndt::type replaced_tp;
  bool was_transformed = false;
  replace_compatible_dtype_extra extra(replacement_tp, replace_ndim);
  replace_compatible_dtype(get_type(), 0, &extra, replaced_tp, was_transformed);
  if (was_transformed) {
    return make_array_clone_with_new_type(*this, replaced_tp);
  } else {
    return *this;
  }
}

nd::array nd::array::new_axis(intptr_t i, intptr_t new_ndim) const
{
  ndt::type src_tp = get_type();
  ndt::type dst_tp = src_tp.with_new_axis(i, new_ndim);

  // This is taken from view_concrete in view.cpp
  nd::array res(make_array_memory_block(dst_tp.get_arrmeta_size()));
  res.get_ndo()->data.ptr = get_ndo()->data.ptr;
  if (get_ndo()->data.ref == NULL) {
    res.get_ndo()->data.ref = get_memblock().release();
  } else {
    res.get_ndo()->data.ref = get_data_memblock().release();
  }
  res.get_ndo()->m_type = ndt::type(dst_tp).release();
  res.get_ndo()->m_flags = get_ndo()->m_flags;

  char *src_arrmeta = get_ndo()->get_arrmeta();
  char *dst_arrmeta = res.get_arrmeta();
  for (intptr_t j = 0; j < i; ++j) {
    dst_tp.extended<ndt::base_dim_type>()->arrmeta_copy_construct_onedim(dst_arrmeta, src_arrmeta, NULL);
    src_tp = src_tp.get_type_at_dimension(&src_arrmeta, 1);
    dst_tp = dst_tp.get_type_at_dimension(&dst_arrmeta, 1);
  }
  for (intptr_t j = 0; j < new_ndim; ++j) {
    size_stride_t *smd = reinterpret_cast<size_stride_t *>(dst_arrmeta);
    smd->dim_size = 1;
    smd->stride = 0; // Should this not be zero?
    dst_tp = dst_tp.get_type_at_dimension(&dst_arrmeta, 1);
  }
  if (!dst_tp.is_builtin()) {
    dst_tp.extended()->arrmeta_copy_construct(dst_arrmeta, src_arrmeta, NULL);
  }

  return res;
}

namespace {
static void view_scalar_types(const ndt::type &dt, intptr_t DYND_UNUSED(arrmeta_offset), void *extra,
                              ndt::type &out_transformed_tp, bool &out_was_transformed)
{
  if (dt.is_scalar()) {
    const ndt::type *e = reinterpret_cast<const ndt::type *>(extra);
    // If things aren't simple, use a view_type
    if (dt.get_kind() == expr_kind || dt.get_data_size() != e->get_data_size() || !dt.is_pod() || !e->is_pod()) {
      // Some special cases that have the same memory layouts
      switch (dt.get_type_id()) {
      case string_type_id:
      case bytes_type_id:
        switch (e->get_type_id()) {
        case string_type_id:
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
      out_transformed_tp = ndt::view_type::make(*e, dt);
      out_was_transformed = true;
    } else {
      out_transformed_tp = *e;
      if (dt != *e) {
        out_was_transformed = true;
      }
    }
  } else {
    dt.extended()->transform_child_types(&view_scalar_types, 0, extra, out_transformed_tp, out_was_transformed);
  }
}
} // anonymous namespace

nd::array nd::array::view_scalars(const ndt::type &scalar_tp) const
{
  const ndt::type &array_type = get_type();
  size_t uniform_ndim = array_type.get_ndim();
  // First check if we're dealing with a simple one dimensional block of memory
  // we can reinterpret
  // at will.
  if (uniform_ndim == 1 && array_type.get_type_id() == fixed_dim_type_id) {
    const ndt::fixed_dim_type *sad = array_type.extended<ndt::fixed_dim_type>();
    const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(get_arrmeta());
    const ndt::type &edt = sad->get_element_type();
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
      // Create the result array, adjusting the type if the data isn't aligned
      // correctly
      char *data_ptr = get_ndo()->data.ptr;
      ndt::type result_tp;
      intptr_t dim_size = nbytes / scalar_tp.get_data_size();
      if ((((uintptr_t)data_ptr) & (scalar_tp.get_data_alignment() - 1)) == 0) {
        result_tp = ndt::make_fixed_dim(dim_size, scalar_tp);
      } else {
        result_tp = ndt::make_fixed_dim(dim_size, make_unaligned(scalar_tp));
      }
      array result(make_array_memory_block(result_tp.extended()->get_arrmeta_size()));
      // Copy all the array arrmeta fields
      result.get_ndo()->data.ptr = get_ndo()->data.ptr;
      if (get_ndo()->data.ref) {
        result.get_ndo()->data.ref = get_ndo()->data.ref;
      } else {
        result.get_ndo()->data.ref = m_memblock.get();
      }
      memory_block_incref(result.get_ndo()->data.ref);
      result.get_ndo()->m_type = result_tp.release();
      result.get_ndo()->m_flags = get_ndo()->m_flags;
      // The result has one strided ndarray field
      fixed_dim_type_arrmeta *result_md = reinterpret_cast<fixed_dim_type_arrmeta *>(result.get_arrmeta());
      result_md->dim_size = dim_size;
      result_md->stride = scalar_tp.get_data_size();
      return result;
    }
  }

  // Transform the scalars into view types
  ndt::type viewed_tp;
  bool was_transformed = false;
  view_scalar_types(get_type(), 0, const_cast<void *>(reinterpret_cast<const void *>(&scalar_tp)), viewed_tp,
                    was_transformed);
  return make_array_clone_with_new_type(*this, viewed_tp);
}

std::string nd::detail::array_as_string(const nd::array &lhs, assign_error_mode errmode)
{
  if (!lhs.is_scalar()) {
    throw std::runtime_error("can only convert arrays with 0 dimensions to scalars");
  }

  nd::array temp = lhs;
  if (temp.get_type().get_kind() != string_kind) {
    temp = temp.ucast(ndt::string_type::make()).eval();
  }
  const ndt::base_string_type *esd = static_cast<const ndt::base_string_type *>(temp.get_type().extended());
  return esd->get_utf8_string(temp.get_arrmeta(), temp.get_ndo()->data.ptr, errmode);
}

ndt::type nd::detail::array_as_type(const nd::array &lhs)
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

void nd::array::debug_print(std::ostream &o, const std::string &indent) const
{
  o << indent << "------ array\n";
  if (m_memblock.get()) {
    const array_preamble *ndo = get_ndo();
    o << " address: " << (void *)m_memblock.get() << "\n";
    o << " refcount: " << ndo->m_memblockdata.m_use_count << "\n";
    o << " type:\n";
    o << "  pointer: " << (void *)ndo->m_type << "\n";
    o << "  type: " << get_type() << "\n";
    if (!get_type().is_builtin()) {
      o << "  type refcount: " << get_type().extended()->get_use_count() << "\n";
    }
    o << " arrmeta:\n";
    o << "  flags: " << ndo->m_flags << " (";
    if (ndo->m_flags & read_access_flag)
      o << "read_access ";
    if (ndo->m_flags & write_access_flag)
      o << "write_access ";
    if (ndo->m_flags & immutable_access_flag)
      o << "immutable ";
    o << ")\n";
    if (!ndo->is_builtin_type()) {
      o << "  type-specific arrmeta:\n";
      ndo->m_type->arrmeta_debug_print(get_arrmeta(), o, indent + "   ");
    }
    o << " data:\n";
    o << "   pointer: " << (void *)ndo->data.ptr << "\n";
    o << "   reference: " << (void *)ndo->data.ref;
    if (ndo->data.ref == NULL) {
      o << " (embedded in array memory)\n";
    } else {
      o << "\n";
    }
    if (ndo->data.ref != NULL) {
      memory_block_debug_print(ndo->data.ref, o, "    ");
    }
  } else {
    o << indent << "NULL\n";
  }
  o << indent << "------" << endl;
}

std::ostream &nd::operator<<(std::ostream &o, const array &rhs)
{
  if (!rhs.is_null()) {
    o << "array(";
    array v = rhs.eval();
    if (v.get_ndo()->is_builtin_type()) {
      print_builtin_scalar(v.get_ndo()->get_builtin_type_id(), o, v.get_ndo()->data.ptr);
    } else {
      stringstream ss;
      v.get_ndo()->m_type->print_data(ss, v.get_arrmeta(), v.get_ndo()->data.ptr);
      print_indented(o, "      ", ss.str(), true);
    }
    o << ",\n      type=\"" << rhs.get_type() << "\")";
  } else {
    o << "array()";
  }
  return o;
}

nd::array nd::as_struct()
{
  return empty(ndt::struct_type::make());
}

nd::array nd::as_struct(std::size_t size, const char **names, const array *values)
{
  std::vector<ndt::type> types(size);
  for (std::size_t i = 0; i < size; ++i) {
    types[i] = values[i].get_type();
  }

  array res = empty(ndt::struct_type::make(make_strided_string_array(names, size), array(types)));
  for (std::size_t i = 0; i < size; ++i) {
    res(i).val_assign(values[i]);
  }

  return res;
}

nd::array nd::eval_raw_copy(const ndt::type &dt, const char *arrmeta, const char *data)
{
  // Allocate an output array with the canonical version of the type
  ndt::type cdt = dt.get_canonical_type();
  size_t ndim = dt.get_ndim();
  array result;
  if (ndim > 0) {
    result = nd::empty(cdt);
    // Reorder strides of output strided dimensions in a KEEPORDER fashion
    if (cdt.get_type_id() == fixed_dim_type_id) {
      cdt.extended<ndt::fixed_dim_type>()->reorder_default_constructed_strides(result.get_arrmeta(), dt, arrmeta);
    }
  } else {
    result = nd::empty(cdt);
  }

  typed_data_assign(cdt, result.get_arrmeta(), result.get_readwrite_originptr(), dt, arrmeta, data,
                    &eval::default_eval_context);

  return result;
}

nd::array nd::empty_shell(const ndt::type &tp)
{
  if (tp.is_builtin()) {
    char *data_ptr = NULL;
    intptr_t data_size =
        static_cast<intptr_t>(dynd::ndt::detail::builtin_data_sizes[reinterpret_cast<uintptr_t>(tp.extended())]);
    intptr_t data_alignment =
        static_cast<intptr_t>(dynd::ndt::detail::builtin_data_alignments[reinterpret_cast<uintptr_t>(tp.extended())]);
    memory_block_ptr result(make_array_memory_block(0, data_size, data_alignment, &data_ptr));
    array_preamble *preamble = reinterpret_cast<array_preamble *>(result.get());
    // It's a builtin type id, so no incref
    preamble->m_type = tp.extended();
    preamble->data.ptr = data_ptr;
    preamble->data.ref = NULL;
    preamble->m_flags = nd::read_access_flag | nd::write_access_flag;
    return nd::array(std::move(result));
  } else if (!tp.is_symbolic()) {
    char *data_ptr = NULL;
    size_t arrmeta_size = tp.extended()->get_arrmeta_size();
    size_t data_size = tp.extended()->get_default_data_size();
    memory_block_ptr result;
    if (tp.get_kind() != memory_kind) {
      // Allocate memory the default way
      result = make_array_memory_block(arrmeta_size, data_size, tp.get_data_alignment(), &data_ptr);
      if (tp.get_flags() & type_flag_zeroinit) {
        memset(data_ptr, 0, data_size);
      }
      if (tp.get_flags() & type_flag_construct) {
        tp.extended()->data_construct(NULL, data_ptr);
      }
    } else {
      // Allocate memory based on the memory_kind type
      result = make_array_memory_block(arrmeta_size);
      tp.extended<ndt::base_memory_type>()->data_alloc(&data_ptr, data_size);
      if (tp.get_flags() & type_flag_zeroinit) {
        tp.extended<ndt::base_memory_type>()->data_zeroinit(data_ptr, data_size);
      }
    }
    array_preamble *preamble = reinterpret_cast<array_preamble *>(result.get());
    preamble->m_type = ndt::type(tp).release();
    preamble->data.ptr = data_ptr;
    preamble->data.ref = NULL;
    preamble->m_flags = nd::read_access_flag | nd::write_access_flag;
    return nd::array(std::move(result));
  } else {
    stringstream ss;
    ss << "Cannot create a dynd array with symbolic type " << tp;
    throw type_error(ss.str());
  }
}

nd::array nd::empty(const ndt::type &tp)
{
  // Create an empty shell
  nd::array res = nd::empty_shell(tp);
  // Construct the arrmeta with default settings
  if (tp.get_arrmeta_size() > 0) {
    array_preamble *preamble = res.get_ndo();
    preamble->m_type->arrmeta_default_construct(reinterpret_cast<char *>(preamble + 1), true);
  }
  return res;
}

nd::array nd::empty_like(const nd::array &rhs, const ndt::type &uniform_tp)
{
  if (rhs.get_ndim() == 0) {
    return nd::empty(uniform_tp);
  } else {
    size_t ndim = rhs.get_type().extended()->get_ndim();
    dimvector shape(ndim);
    rhs.get_shape(shape.get());
    array result(make_strided_array(uniform_tp, ndim, shape.get()));
    // Reorder strides of output strided dimensions in a KEEPORDER fashion
    if (result.get_type().get_type_id() == fixed_dim_type_id) {
      result.get_type().extended<ndt::fixed_dim_type>()->reorder_default_constructed_strides(
          result.get_arrmeta(), rhs.get_type(), rhs.get_arrmeta());
    }
    return result;
  }
}

nd::array nd::empty_like(const nd::array &rhs)
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
    intptr_t ndim = dt.extended()->get_ndim();
    dimvector shape(ndim);
    rhs.get_shape(shape.get());
    nd::array result(make_strided_array(dt.get_dtype(), ndim, shape.get()));
    // Reorder strides of output strided dimensions in a KEEPORDER fashion
    if (result.get_type().get_type_id() == fixed_dim_type_id) {
      result.get_type().extended<ndt::fixed_dim_type>()->reorder_default_constructed_strides(
          result.get_arrmeta(), rhs.get_type(), rhs.get_arrmeta());
    }
    return result;
  }
}

nd::array nd::concatenate(const nd::array &x, const nd::array &y)
{
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

nd::array nd::reshape(const nd::array &a, const nd::array &shape)
{
  intptr_t ndim = shape.get_dim_size();

  intptr_t old_ndim = a.get_ndim();
  dimvector old_shape(old_ndim);
  a.get_shape(old_shape.get());

  intptr_t old_size = 1;
  for (intptr_t i = 0; i < old_ndim; ++i) {
    old_size *= old_shape[i];
  }
  intptr_t size = 1;
  for (intptr_t i = 0; i < ndim; ++i) {
    size *= shape(i).as<intptr_t>();
  }

  if (old_size != size) {
    stringstream ss;
    ss << "dynd reshape: cannot reshape to a different total number of "
          "elements, from " << old_size << " to " << size;
    throw invalid_argument(ss.str());
  }

  dimvector strides(ndim);
  strides[ndim - 1] = a.get_dtype().get_data_size();
  for (intptr_t i = ndim - 2; i >= 0; --i) {
    strides[i] = shape(i + 1).as<intptr_t>() * strides[i + 1];
  }

  dimvector shape_copy(ndim);
  for (intptr_t i = 0; i < ndim; ++i) {
    shape_copy[i] = shape(i).as<intptr_t>();
  }

  return make_strided_array_from_data(a.get_dtype(), ndim, shape_copy.get(), strides.get(), a.get_flags(),
                                      a.get_readwrite_originptr(), a.get_memblock(), NULL);
}

nd::array nd::memmap(const std::string &filename, intptr_t begin, intptr_t end, uint32_t access)
{
  if (access == 0) {
    access = nd::default_access_flags;
  }

  char *mm_ptr = NULL;
  intptr_t mm_size = 0;
  // Create a memory mapped memblock of the file
  memory_block_ptr mm = make_memmap_memory_block(filename, access, &mm_ptr, &mm_size, begin, end);
  // Create a bytes array referring to the data.
  ndt::type dt = ndt::bytes_type::make(1);
  char *data_ptr = 0;
  nd::array result(make_array_memory_block(dt.extended()->get_arrmeta_size(), dt.get_data_size(),
                                           dt.get_data_alignment(), &data_ptr));
  // Set the bytes extents
  reinterpret_cast<bytes *>(data_ptr)->assign(mm_ptr, mm_size);
  // Set the array arrmeta
  array_preamble *ndo = result.get_ndo();
  ndo->m_type = dt.release();
  ndo->data.ptr = data_ptr;
  ndo->data.ref = NULL;
  ndo->m_flags = access;
  // Set the bytes arrmeta, telling the system
  // about the memmapped memblock
  bytes_type_arrmeta *ndo_meta = reinterpret_cast<bytes_type_arrmeta *>(result.get_arrmeta());
  ndo_meta->blockref = mm.release();
  return result;
}

bool nd::is_scalar_avail(const ndt::type &tp, const char *arrmeta, const char *data, const eval::eval_context *ectx)
{
  if (tp.is_scalar()) {
    if (tp.get_type_id() == option_type_id) {
      return tp.extended<ndt::option_type>()->is_avail(arrmeta, data, ectx);
    } else if (tp.get_kind() == expr_kind && tp.value_type().get_type_id() == option_type_id) {
      nd::array tmp = nd::empty(tp.value_type());
      tmp.val_assign(tp, arrmeta, data, ectx);
      return tmp.get_type().extended<ndt::option_type>()->is_avail(arrmeta, data, ectx);
    } else {
      return true;
    }
  } else {
    return false;
  }
}

void nd::assign_na(const ndt::type &tp, const char *arrmeta, char *data, const eval::eval_context *ectx)
{
  if (tp.get_type_id() == option_type_id) {
    tp.extended<ndt::option_type>()->assign_na(arrmeta, data, ectx);
  } else {
    const ndt::type &dtp = tp.get_dtype().value_type();
    if (dtp.get_type_id() == option_type_id) {
      throw std::runtime_error("nd::assign_na is not yet implemented");
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
    field_types[i] = ndt::pointer_type::make(field_values[i].get_type());
  }
  // The flags are the intersection of all the input flags
  uint64_t flags = field_values[0].get_flags();
  for (size_t i = 1; i != field_count; ++i) {
    flags &= field_values[i].get_flags();
  }

  ndt::type result_type = ndt::tuple_type::make(field_types);
  const ndt::tuple_type *fsd = result_type.extended<ndt::tuple_type>();
  char *data_ptr = NULL;

  array result(make_array_memory_block(fsd->get_arrmeta_size(), fsd->get_default_data_size(), fsd->get_data_alignment(),
                                       &data_ptr));
  // Set the array properties
  result.get_ndo()->m_type = result_type.release();
  result.get_ndo()->data.ptr = data_ptr;
  result.get_ndo()->data.ref = NULL;
  result.get_ndo()->m_flags = flags;

  // Set the data offsets arrmeta for the tuple type. It's a bunch of pointer
  // types, so the offsets are pretty simple.
  intptr_t *data_offsets = reinterpret_cast<intptr_t *>(result.get_arrmeta());
  for (size_t i = 0; i != field_count; ++i) {
    data_offsets[i] = i * sizeof(void *);
  }

  // Copy all the needed arrmeta
  const uintptr_t *arrmeta_offsets = fsd->get_arrmeta_offsets_raw();
  for (size_t i = 0; i != field_count; ++i) {
    pointer_type_arrmeta *pmeta;
    pmeta = reinterpret_cast<pointer_type_arrmeta *>(result.get_arrmeta() + arrmeta_offsets[i]);
    pmeta->offset = 0;
    pmeta->blockref = field_values[i].get_ndo()->data.ref ? field_values[i].get_ndo()->data.ref
                                                          : &field_values[i].get_ndo()->m_memblockdata;
    memory_block_incref(pmeta->blockref);

    const ndt::type &field_dt = field_values[i].get_type();
    if (field_dt.get_arrmeta_size() > 0) {
      field_dt.extended()->arrmeta_copy_construct(reinterpret_cast<char *>(pmeta + 1), field_values[i].get_arrmeta(),
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

void nd::forward_as_array(const ndt::type &tp, char *arrmeta, char *data, const nd::array &val)
{

  if (tp.is_builtin() || tp.get_type_id() == callable_type_id) {
    memcpy(data, val.get_readonly_originptr(), tp.get_data_size());
  } else {
    pointer_type_arrmeta *am = reinterpret_cast<pointer_type_arrmeta *>(arrmeta);
    // Insert the reference in the destination pointer's arrmeta
    am->blockref = val.get_data_memblock().get();
    memory_block_incref(am->blockref);
    // Copy the rest of the arrmeta after the pointer's arrmeta
    const ndt::type &val_tp = val.get_type();
    if (val_tp.get_arrmeta_size() > 0) {
      val_tp.extended()->arrmeta_copy_construct(arrmeta + sizeof(pointer_type_arrmeta), val.get_arrmeta(),
                                                val.get_memblock().get());
    }
    // Copy the pointer
    *reinterpret_cast<char **>(data) = const_cast<char *>(val.get_readonly_originptr());
  }
}

void nd::forward_as_array(const ndt::type &tp, char *arrmeta, char *data, const nd::callable &value)
{
  forward_as_array(tp, arrmeta, data, static_cast<nd::array>(value));
}
