//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/exceptions.hpp>
#include <dynd/type.hpp>
#include <dynd/type_registry.hpp>
#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/base_dim_type.hpp>
#include <dynd/types/base_memory_type.hpp>
#include <dynd/types/bool_kind_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/categorical_kind_type.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/complex_kind_type.hpp>
#include <dynd/types/datashape_parser.hpp>
#include <dynd/types/fixed_bytes_kind_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/fixed_string_kind_type.hpp>
#include <dynd/types/float_kind_type.hpp>
#include <dynd/types/int_kind_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/scalar_kind_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/uint_kind_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>

#include <algorithm>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <vector>

using namespace std;
using namespace dynd;

char *dynd::iterdata_broadcasting_terminator_incr(iterdata_common *iterdata, intptr_t DYND_UNUSED(level)) {
  // This repeats the same data over and over again, broadcasting additional
  // leftmost dimensions
  iterdata_broadcasting_terminator *id = reinterpret_cast<iterdata_broadcasting_terminator *>(iterdata);
  return id->data;
}

char *dynd::iterdata_broadcasting_terminator_adv(iterdata_common *iterdata, intptr_t DYND_UNUSED(level),
                                                 intptr_t DYND_UNUSED(i)) {
  // This repeats the same data over and over again, broadcasting additional
  // leftmost dimensions
  iterdata_broadcasting_terminator *id = reinterpret_cast<iterdata_broadcasting_terminator *>(iterdata);
  return id->data;
}

char *dynd::iterdata_broadcasting_terminator_reset(iterdata_common *iterdata, char *data, intptr_t DYND_UNUSED(level)) {
  iterdata_broadcasting_terminator *id = reinterpret_cast<iterdata_broadcasting_terminator *>(iterdata);
  id->data = data;
  return data;
}

ndt::type::type(const std::string &rep) { type_from_datashape(rep).swap(*this); }

ndt::type::type(const char *rep_begin, const char *rep_end) { type_from_datashape(rep_begin, rep_end).swap(*this); }

size_t ndt::type::get_data_alignment() const {
  switch (reinterpret_cast<uintptr_t>(m_ptr)) {
  case uninitialized_id:
    return 1;
  case bool_id:
    return alignof(bool1);
  case int8_id:
    return alignof(int8_t);
  case int16_id:
    return alignof(int16_t);
  case int32_id:
    return alignof(int32_t);
  case int64_id:
    return alignof(int64_t);
  case int128_id:
    return alignof(int128);
  case uint8_id:
    return alignof(uint8_t);
  case uint16_id:
    return alignof(uint16_t);
  case uint32_id:
    return alignof(uint32_t);
  case uint64_id:
    return alignof(uint64_t);
  case uint128_id:
    return alignof(uint128);
  case float16_id:
    return alignof(float16);
  case float32_id:
    return alignof(float);
  case float64_id:
    return alignof(double);
  case float128_id:
    return alignof(float128);
  case complex_float32_id:
    return alignof(complex<float>);
  case complex_float64_id:
    return alignof(complex<double>);
  case void_id:
    return 1;
  default:
    return m_ptr->get_data_alignment();
  }
}

size_t ndt::type::get_data_size() const {
  switch (reinterpret_cast<uintptr_t>(m_ptr)) {
  case uninitialized_id:
    return 0;
  case bool_id:
    return sizeof(bool1);
  case int8_id:
    return sizeof(int8_t);
  case int16_id:
    return sizeof(int16_t);
  case int32_id:
    return sizeof(int32_t);
  case int64_id:
    return sizeof(int64_t);
  case int128_id:
    return sizeof(int128);
  case uint8_id:
    return sizeof(uint8_t);
  case uint16_id:
    return sizeof(uint16_t);
  case uint32_id:
    return sizeof(uint32_t);
  case uint64_id:
    return sizeof(uint64_t);
  case uint128_id:
    return sizeof(uint128);
  case float16_id:
    return sizeof(float16);
  case float32_id:
    return sizeof(float);
  case float64_id:
    return sizeof(double);
  case float128_id:
    return sizeof(float128);
  case complex_float32_id:
    return sizeof(complex<float>);
  case complex_float64_id:
    return sizeof(complex<double>);
  case void_id:
    return 0;
  default:
    return m_ptr->get_data_size();
  }
}

size_t ndt::type::get_default_data_size() const {
  switch (reinterpret_cast<uintptr_t>(m_ptr)) {
  case uninitialized_id:
    return 0;
  case bool_id:
    return sizeof(bool1);
  case int8_id:
    return sizeof(int8_t);
  case int16_id:
    return sizeof(int16_t);
  case int32_id:
    return sizeof(int32_t);
  case int64_id:
    return sizeof(int64_t);
  case int128_id:
    return sizeof(int128);
  case uint8_id:
    return sizeof(uint8_t);
  case uint16_id:
    return sizeof(uint16_t);
  case uint32_id:
    return sizeof(uint32_t);
  case uint64_id:
    return sizeof(uint64_t);
  case uint128_id:
    return sizeof(uint128);
  case float16_id:
    return sizeof(float16);
  case float32_id:
    return sizeof(float);
  case float64_id:
    return sizeof(double);
  case float128_id:
    return sizeof(float128);
  case complex_float32_id:
    return sizeof(complex<float>);
  case complex_float64_id:
    return sizeof(complex<double>);
  case void_id:
    return 0;
  default:
    return m_ptr->get_default_data_size();
  }
}

ndt::type ndt::type::at_array(int nindices, const irange *indices) const {
  if (this->is_builtin()) {
    if (nindices == 0) {
      return *this;
    } else {
      throw too_many_indices(*this, nindices, 0);
    }
  } else {
    return m_ptr->apply_linear_index(nindices, indices, 0, *this, true);
  }
}

bool ndt::type::match(const type &other, std::map<std::string, type> &tp_vars) const {
  return m_ptr == other.m_ptr || (!is_builtin() && m_ptr->match(other, tp_vars));
}

ndt::type ndt::type::apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i,
                                        const ndt::type &root_tp, bool leading_dimension) const {
  if (is_builtin()) {
    if (nindices == 0) {
      return *this;
    } else {
      throw too_many_indices(*this, nindices + current_i, current_i);
    }
  } else {
    return m_ptr->apply_linear_index(nindices, indices, current_i, root_tp, leading_dimension);
  }
}

namespace {
struct replace_scalar_type_extra {
  replace_scalar_type_extra(const ndt::type &dt) : scalar_tp(dt) {}
  const ndt::type &scalar_tp;
};
static void replace_scalar_types(const ndt::type &dt, intptr_t DYND_UNUSED(arrmeta_offset), void *extra,
                                 ndt::type &out_transformed_tp, bool &out_was_transformed) {
  //  const replace_scalar_type_extra *e = reinterpret_cast<const replace_scalar_type_extra *>(extra);
  if (!dt.is_indexable()) {
    throw std::runtime_error("trying to make convert_type");
    //    out_transformed_tp = ndt::convert_type::make(e->scalar_tp, dt);
    out_was_transformed = true;
  } else {
    dt.extended()->transform_child_types(&replace_scalar_types, 0, extra, out_transformed_tp, out_was_transformed);
  }
}
} // anonymous namespace

ndt::type ndt::type::with_replaced_scalar_types(const ndt::type &scalar_tp) const {
  ndt::type result;
  bool was_transformed;
  replace_scalar_type_extra extra(scalar_tp);
  replace_scalar_types(*this, 0, &extra, result, was_transformed);
  return result;
}

namespace {
struct replace_dtype_extra {
  replace_dtype_extra(const ndt::type &replacement_tp, intptr_t replace_ndim)
      : m_replacement_tp(replacement_tp), m_replace_ndim(replace_ndim) {}
  const ndt::type &m_replacement_tp;
  intptr_t m_replace_ndim;
};
static void replace_dtype(const ndt::type &tp, intptr_t DYND_UNUSED(arrmeta_offset), void *extra,
                          ndt::type &out_transformed_tp, bool &out_was_transformed) {
  const replace_dtype_extra *e = reinterpret_cast<const replace_dtype_extra *>(extra);
  if (tp.get_ndim() == e->m_replace_ndim) {
    out_transformed_tp = e->m_replacement_tp;
    out_was_transformed = true;
  } else {
    tp.extended()->transform_child_types(&replace_dtype, 0, extra, out_transformed_tp, out_was_transformed);
  }
}
} // anonymous namespace

ndt::type ndt::type::with_replaced_dtype(const ndt::type &replacement_tp, intptr_t replace_ndim) const {
  ndt::type result;
  bool was_transformed;
  replace_dtype_extra extra(replacement_tp, replace_ndim);
  replace_dtype(*this, 0, &extra, result, was_transformed);
  return result;
}

ndt::type ndt::type::without_memory_type() const {
  if (get_base_id() == memory_id) {
    return extended<base_memory_type>()->get_element_type();
  }

  return *this;
}

const ndt::type &ndt::type::storage_type() const {
  // Only expr_kind types have different storage_type
  if (is_builtin() || get_base_id() != expr_kind_id) {
    return *this;
  }

  return extended<base_expr_type>()->get_storage_type();
}

const ndt::type &ndt::type::value_type() const {
  // Only expr_kind types have different value_type
  if (is_builtin() || get_base_id() != expr_kind_id) {
    return *this;
  }

  return extended<base_expr_type>()->get_value_type();
}

ndt::type ndt::type::with_new_axis(intptr_t i, intptr_t new_ndim) const {
  ndt::type tp = without_memory_type();

  tp = tp.with_replaced_dtype(ndt::pow(ndt::make_fixed_dim(1, tp.get_type_at_dimension(NULL, i)), new_ndim),
                              tp.get_ndim() - i);
  if (get_base_id() == memory_id) {
    tp = extended<base_memory_type>()->with_replaced_storage_type(tp);
  }

  return tp;
}

intptr_t ndt::type::get_dim_size(const char *arrmeta, const char *data) const {
  if (get_base_id() == dim_kind_id) {
    return static_cast<const base_dim_type *>(get())->get_dim_size(arrmeta, data);
  } else if (get_id() == struct_id) {
    return static_cast<const struct_type *>(get())->get_field_count();
  } else if (get_ndim() > 0) {
    intptr_t dim_size = -1;
    get()->get_shape(1, 0, &dim_size, arrmeta, data);
    if (dim_size >= 0) {
      return dim_size;
    }
  }

  std::stringstream ss;
  ss << "Scalar dynd array of type " << *this << " has no length";
  throw std::invalid_argument(ss.str());
}

intptr_t ndt::type::get_size(const char *arrmeta) const {
  if (is_scalar()) {
    return 1;
  }

  return extended<base_dim_type>()->get_size(arrmeta);
}

bool ndt::type::get_as_strided(const char *arrmeta, intptr_t *out_dim_size, intptr_t *out_stride, ndt::type *out_el_tp,
                               const char **out_el_arrmeta) const {
  if (get_base_id() == memory_id) {
    bool res = without_memory_type().get_as_strided(arrmeta, out_dim_size, out_stride, out_el_tp, out_el_arrmeta);
    *out_el_tp = extended<base_memory_type>()->with_replaced_storage_type(*out_el_tp);
    return res;
  }

  if (get_strided_ndim() >= 1) {
    *out_dim_size = reinterpret_cast<const size_stride_t *>(arrmeta)->dim_size;
    *out_stride = reinterpret_cast<const size_stride_t *>(arrmeta)->stride;
    *out_el_tp = extended<base_dim_type>()->get_element_type();
    *out_el_arrmeta = arrmeta + sizeof(fixed_dim_type_arrmeta);
    return true;
  } else {
    return false;
  }
}

std::map<std::string, std::pair<ndt::type, const char *>> ndt::type::get_properties() const {
  std::map<std::string, std::pair<ndt::type, const char *>> properties;
  if (!is_builtin()) {
    return m_ptr->get_dynamic_type_properties();
  }

  return properties;
}

bool ndt::type::get_as_strided(const char *arrmeta, intptr_t ndim, const size_stride_t **out_size_stride,
                               ndt::type *out_el_tp, const char **out_el_arrmeta) const {
  if (get_strided_ndim() >= ndim) {
    *out_size_stride = reinterpret_cast<const size_stride_t *>(arrmeta);
    *out_el_arrmeta = arrmeta + ndim * sizeof(fixed_dim_type_arrmeta);
    *out_el_tp = *this;
    while (ndim-- > 0) {
      *out_el_tp = out_el_tp->extended<base_dim_type>()->get_element_type();
    }
    return true;
  } else {
    return false;
  }
}
bool ndt::type::data_layout_compatible_with(const ndt::type &rhs) const {
  if (extended() == rhs.extended()) {
    // If they're trivially identical, quickly return true
    return true;
  }
  if (get_data_size() != rhs.get_data_size() || get_arrmeta_size() != rhs.get_arrmeta_size()) {
    // The size of the data and arrmeta must be the same
    return false;
  }
  if (get_arrmeta_size() == 0 && is_pod() && rhs.is_pod()) {
    // If both are POD with no arrmeta, then they're compatible
    return true;
  }
  if (get_base_id() == expr_kind_id || rhs.get_base_id() == expr_kind_id) {
    // If either is an expression type, check compatibility with
    // the storage types
    return storage_type().data_layout_compatible_with(rhs.storage_type());
  }
  // Rules for the rest of the types
  switch (get_id()) {
  case string_id:
  case bytes_id:
    switch (rhs.get_id()) {
    case string_id:
    case bytes_id:
      // All of string, bytes, json are compatible
      return true;
    default:
      return false;
    }
  case fixed_dim_id:
    if (rhs.get_id() == fixed_dim_id) {
      return extended<fixed_dim_type>()->get_fixed_dim_size() == rhs.extended<fixed_dim_type>()->get_fixed_dim_size() &&
             extended<fixed_dim_type>()->get_element_type().data_layout_compatible_with(
                 rhs.extended<fixed_dim_type>()->get_element_type());
    }
    break;
  case var_dim_id:
    // For var dimensions, it's data layout
    // compatible if the element is
    if (rhs.get_id() == var_dim_id) {
      const base_dim_type *budd = static_cast<const base_dim_type *>(extended());
      const base_dim_type *rhs_budd = rhs.extended<base_dim_type>();
      return budd->get_element_type().data_layout_compatible_with(rhs_budd->get_element_type());
    }
    break;
  default:
    break;
  }
  return false;
}

std::ostream &dynd::ndt::operator<<(std::ostream &o, const ndt::type &rhs) {
  switch (rhs.get_id()) {
  case uninitialized_id:
    o << "uninitialized";
    break;
  case bool_id:
    o << "bool";
    break;
  case int8_id:
    o << "int8";
    break;
  case int16_id:
    o << "int16";
    break;
  case int32_id:
    o << "int32";
    break;
  case int64_id:
    o << "int64";
    break;
  case int128_id:
    o << "int128";
    break;
  case uint8_id:
    o << "uint8";
    break;
  case uint16_id:
    o << "uint16";
    break;
  case uint32_id:
    o << "uint32";
    break;
  case uint64_id:
    o << "uint64";
    break;
  case uint128_id:
    o << "uint128";
    break;
  case float16_id:
    o << "float16";
    break;
  case float32_id:
    o << "float32";
    break;
  case float64_id:
    o << "float64";
    break;
  case float128_id:
    o << "float128";
    break;
  case complex_float32_id:
    o << "complex[float32]";
    break;
  case complex_float64_id:
    o << "complex[float64]";
    break;
  case void_id:
    o << "void";
    break;
  default:
    rhs.extended()->print_type(o);
    break;
  }

  return o;
}

ndt::type ndt::make_type(intptr_t ndim, const intptr_t *shape, const ndt::type &dtp) {
  if (ndim > 0) {
    ndt::type result_tp =
        shape[ndim - 1] >= 0 ? ndt::make_fixed_dim(shape[ndim - 1], dtp) : ndt::make_type<ndt::var_dim_type>(dtp);
    for (intptr_t i = ndim - 2; i >= 0; --i) {
      if (shape[i] >= 0) {
        result_tp = ndt::make_fixed_dim(shape[i], result_tp);
      } else {
        result_tp = ndt::make_type<ndt::var_dim_type>(result_tp);
      }
    }
    return result_tp;
  } else {
    return dtp;
  }
}

ndt::type ndt::make_type(intptr_t ndim, const intptr_t *shape, const ndt::type &dtp, bool &out_any_var) {
  if (ndim > 0) {
    ndt::type result_tp = dtp;
    for (intptr_t i = ndim - 1; i >= 0; --i) {
      if (shape[i] >= 0) {
        result_tp = ndt::make_fixed_dim(shape[i], result_tp);
      } else {
        result_tp = ndt::make_type<ndt::var_dim_type>(result_tp);
        out_any_var = true;
      }
    }
    return result_tp;
  } else {
    return dtp;
  }
}

DYNDT_API ndt::type ndt::pow(const type &base_tp, size_t exponent) {
  switch (exponent) {
  case 0:
    return base_tp.extended<base_dim_type>()->get_element_type();
  case 1:
    return base_tp;
  default:
    return base_tp.extended<base_dim_type>()->with_element_type(pow(base_tp, exponent - 1));
  }
}

template <class T, class Tas>
static void print_as(std::ostream &o, const char *data) {
  T value;
  memcpy(&value, data, sizeof(value));
  o << static_cast<Tas>(value);
}

void dynd::hexadecimal_print(std::ostream &o, char value) {
  static char hexadecimal[] = "0123456789abcdef";
  unsigned char v = (unsigned char)value;
  o << hexadecimal[v >> 4] << hexadecimal[v & 0x0f];
}

void dynd::hexadecimal_print(std::ostream &o, unsigned char value) { hexadecimal_print(o, static_cast<char>(value)); }

void dynd::hexadecimal_print(std::ostream &o, unsigned short value) {
  // Standard printing is in big-endian order
  hexadecimal_print(o, static_cast<char>((value >> 8) & 0xff));
  hexadecimal_print(o, static_cast<char>(value & 0xff));
}

void dynd::hexadecimal_print(std::ostream &o, unsigned int value) {
  // Standard printing is in big-endian order
  hexadecimal_print(o, static_cast<char>(value >> 24));
  hexadecimal_print(o, static_cast<char>((value >> 16) & 0xff));
  hexadecimal_print(o, static_cast<char>((value >> 8) & 0xff));
  hexadecimal_print(o, static_cast<char>(value & 0xff));
}

void dynd::hexadecimal_print(std::ostream &o, unsigned long value) {
  if (sizeof(unsigned int) == sizeof(unsigned long)) {
    hexadecimal_print(o, static_cast<unsigned int>(value));
  } else {
    hexadecimal_print(o, static_cast<unsigned long long>(value));
  }
}

void dynd::hexadecimal_print(std::ostream &o, unsigned long long value) {
  // Standard printing is in big-endian order
  hexadecimal_print(o, static_cast<char>(value >> 56));
  hexadecimal_print(o, static_cast<char>((value >> 48) & 0xff));
  hexadecimal_print(o, static_cast<char>((value >> 40) & 0xff));
  hexadecimal_print(o, static_cast<char>((value >> 32) & 0xff));
  hexadecimal_print(o, static_cast<char>((value >> 24) & 0xff));
  hexadecimal_print(o, static_cast<char>((value >> 16) & 0xff));
  hexadecimal_print(o, static_cast<char>((value >> 8) & 0xff));
  hexadecimal_print(o, static_cast<char>(value & 0xff));
}

void dynd::hexadecimal_print(std::ostream &o, const char *data, intptr_t element_size) {
  for (int i = 0; i < element_size; ++i, ++data) {
    hexadecimal_print(o, *data);
  }
}

void dynd::hexadecimal_print_summarized(std::ostream &o, const char *data, intptr_t element_size,
                                        intptr_t summary_size) {
  if (element_size * 2 <= summary_size) {
    hexadecimal_print(o, data, element_size);
  } else {
    // Print a summary
    intptr_t size = max(summary_size / 4 - 1, (intptr_t)1);
    hexadecimal_print(o, data, size);
    o << " ... ";
    size = max(summary_size / 4 - size - 1, (intptr_t)1);
    hexadecimal_print(o, data + element_size - size, size);
  }
}

static intptr_t line_count(const std::string &s) {
  return 1 + count_if(s.begin(), s.end(), bind(equal_to<char>(), '\n', placeholders::_1));
}

static void summarize_stats(const std::string &s, intptr_t &num_rows, intptr_t &max_num_cols) {
  num_rows += line_count(s);
  max_num_cols = max(max_num_cols, (intptr_t)s.size());
}

void dynd::print_indented(ostream &o, const std::string &indent, const std::string &s, bool skipfirstline) {
  const char *begin = s.data();
  const char *end = s.data() + s.size();
  const char *cur = begin;
  while (cur != end) {
    const char *next = find_if(cur, end, bind(equal_to<char>(), '\n', placeholders::_1));
    if (*next == '\n')
      ++next;
    if (!skipfirstline || cur != begin)
      o << indent;
    o.write(cur, next - cur);
    cur = next;
  }
}

// TODO Move the magic numbers into parameters
void dynd::strided_array_summarized(std::ostream &o, const ndt::type &tp, const char *arrmeta, const char *data,
                                    intptr_t dim_size, intptr_t stride) {
  const int leading_count = 7, trailing_count = 3, row_threshold = 10, col_threshold = 30, packing_cols = 75;

  vector<std::string> leading, trailing;
  intptr_t ilead = 0, itrail = dim_size - 1;
  intptr_t num_rows = 0, max_num_cols = 0;
  // Get leading strings
  for (; ilead < leading_count && ilead < dim_size; ++ilead) {
    stringstream ss;
    tp.print_data(ss, arrmeta, data + ilead * stride);
    leading.push_back(ss.str());
    summarize_stats(leading.back(), num_rows, max_num_cols);
  }
  // Get trailing strings
  for (itrail = max(dim_size - trailing_count, ilead + 1); itrail < dim_size; ++itrail) {
    stringstream ss;
    tp.print_data(ss, arrmeta, data + itrail * stride);
    trailing.push_back(ss.str());
    summarize_stats(trailing.back(), num_rows, max_num_cols);
  }
  itrail = dim_size - trailing.size() - 1;

  // Select between two printing strategies depending on what we got
  if ((size_t)num_rows > (leading.size() + trailing.size()) || max_num_cols > col_threshold) {
    // Trim the leading/trailing vectors until we get to our threshold
    while (num_rows > row_threshold && (trailing.size() > 1 || leading.size() > 1)) {
      if (trailing.size() > 1) {
        num_rows -= line_count(trailing.front());
        trailing.erase(trailing.begin());
      }
      if (num_rows > row_threshold && leading.size() > 1) {
        if (trailing.empty()) {
          trailing.insert(trailing.begin(), leading.back());
        } else {
          num_rows -= line_count(leading.back());
        }
        leading.pop_back();
      }
    }
    // Print the [leading, ..., trailing]
    o << "[";
    print_indented(o, " ", leading.front(), true);
    for (size_t i = 1; i < leading.size(); ++i) {
      o << ",\n";
      print_indented(o, " ", leading[i]);
    }
    if (leading.size() != (size_t)dim_size) {
      if ((size_t)dim_size > (leading.size() + trailing.size())) {
        o << ",\n ...\n";
      }
      if (!trailing.empty()) {
        print_indented(o, " ", trailing.front());
      }
      for (size_t i = 1; i < trailing.size(); ++i) {
        o << ",\n";
        print_indented(o, " ", trailing[i]);
      }
    }
    o << "]";
  } else {
    // Pack the values in a regular grid
    // Keep getting more strings until we use up our column budget.
    intptr_t total_cols = (max_num_cols + 2) * (leading.size() + trailing.size());
    while (ilead <= itrail && total_cols < packing_cols * row_threshold) {
      if (ilead <= itrail) {
        stringstream ss;
        tp.print_data(ss, arrmeta, data + ilead++ * stride);
        leading.push_back(ss.str());
        summarize_stats(leading.back(), num_rows, max_num_cols);
      }
      if (ilead <= itrail) {
        stringstream ss;
        tp.print_data(ss, arrmeta, data + itrail-- * stride);
        trailing.insert(trailing.begin(), ss.str());
        summarize_stats(trailing.front(), num_rows, max_num_cols);
      }
      total_cols = (max_num_cols + 2) * (leading.size() + trailing.size());
    }

    intptr_t per_row = packing_cols / (max_num_cols + 2);

    if (leading.size() + trailing.size() == (size_t)dim_size) {
      // Combine the lists if the total size is covered
      copy(trailing.begin(), trailing.end(), back_inserter(leading));
      trailing.clear();
    } else {
      // Remove partial rows if the total size is not covered
      if (leading.size() > (size_t)per_row && leading.size() % per_row != 0) {
        leading.erase(leading.begin() + (leading.size() / per_row) * per_row, leading.end());
      }
      if (trailing.size() > (size_t)per_row && trailing.size() % per_row != 0) {
        trailing.erase(trailing.begin(), trailing.begin() + trailing.size() % per_row);
      }
    }

    intptr_t i = 0, j;
    intptr_t i_size = leading.size();
    if (!i_size)
      o << '[';
    while (i < i_size) {
      o << ((i == 0) ? "[" : " ") << setw(max_num_cols) << leading[i] << setw(0);
      ++i;
      for (j = 1; j < per_row && i < i_size; ++j, ++i) {
        o << ", " << setw(max_num_cols) << leading[i] << setw(0);
      }
      if (i < i_size - 1) {
        o << ",\n";
      }
    }
    if (leading.size() != (size_t)dim_size) {
      i = 0;
      i_size = trailing.size();
      o << ",\n ...\n";
      while (i < i_size) {
        o << " " << setw(max_num_cols) << trailing[i] << setw(0);
        ++i;
        for (j = 1; j < per_row && i < i_size; ++j, ++i) {
          o << ", " << setw(max_num_cols) << trailing[i] << setw(0);
        }
        if (i < i_size - 1) {
          o << ",\n";
        }
      }
    }
    o << "]";
  }
}

void dynd::print_builtin_scalar(type_id_t type_id, std::ostream &o, const char *data) {
  switch (type_id) {
  case bool_id:
    o << (*data ? "True" : "False");
    break;
  case int8_id:
    print_as<int8, int32>(o, data);
    break;
  case int16_id:
    print_as<int16, int32>(o, data);
    break;
  case int32_id:
    print_as<int32, int32>(o, data);
    break;
  case int64_id:
    print_as<int64, int64>(o, data);
    break;
  case int128_id:
    print_as<int128, int128>(o, data);
    break;
  case uint8_id:
    print_as<uint8, uint32>(o, data);
    break;
  case uint16_id:
    print_as<uint16, uint32>(o, data);
    break;
  case uint32_id:
    print_as<uint32, uint32>(o, data);
    break;
  case uint64_id:
    print_as<uint64, uint64>(o, data);
    break;
  case uint128_id:
    print_as<uint128, uint128>(o, data);
    break;
  case float16_id:
    print_as<float16, float32>(o, data);
    break;
  case float32_id:
    print_as<float32, float32>(o, data);
    break;
  case float64_id:
    print_as<float64, float64>(o, data);
    break;
  case float128_id:
    print_as<float128, float128>(o, data);
    break;
  case complex_float32_id:
    print_as<complex<float>, complex<float>>(o, data);
    break;
  case complex_float64_id:
    print_as<complex<double>, complex<double>>(o, data);
    break;
  case void_id:
    o << "(void)";
    break;
  default:
    stringstream ss;
    ss << "printing of dynd builtin type id " << type_id << " isn't supported yet";
    throw dynd::type_error(ss.str());
  }
}

void ndt::type::print_data(std::ostream &o, const char *arrmeta, const char *data) const {
  if (is_builtin()) {
    print_builtin_scalar(get_id(), o, data);
  } else {
    extended()->print_data(o, arrmeta, data);
  }
}

type_id_t ndt::type::get_base_id() const { return dynd::detail::infos()[get_id()].base_id; }

// Returns true if the destination type can represent *all* the values
// of the source type, false otherwise. This is used, for example,
// to skip any overflow checks when doing value assignments between differing
// types.
bool dynd::is_lossless_assignment(const ndt::type &dst_tp, const ndt::type &src_tp) {
  if (dst_tp.is_builtin() && src_tp.is_builtin()) {
    switch (src_tp.get_base_id()) {
    case bool_kind_id:
      switch (dst_tp.get_base_id()) {
      case bool_kind_id:
      case int_kind_id:
      case uint_kind_id:
      case float_kind_id:
      case complex_kind_id:
        return true;
      case bytes_kind_id:
        return false;
      default:
        break;
      }
      break;
    case int_kind_id:
      switch (dst_tp.get_base_id()) {
      case bool_kind_id:
        return false;
      case int_kind_id:
        return dst_tp.get_data_size() >= src_tp.get_data_size();
      case uint_kind_id:
        return false;
      case float_kind_id:
        return dst_tp.get_data_size() > src_tp.get_data_size();
      case complex_kind_id:
        return dst_tp.get_data_size() > 2 * src_tp.get_data_size();
      case bytes_kind_id:
        return false;
      default:
        break;
      }
      break;
    case uint_kind_id:
      switch (dst_tp.get_base_id()) {
      case bool_kind_id:
        return false;
      case int_kind_id:
        return dst_tp.get_data_size() > src_tp.get_data_size();
      case uint_kind_id:
        return dst_tp.get_data_size() >= src_tp.get_data_size();
      case float_kind_id:
        return dst_tp.get_data_size() > src_tp.get_data_size();
      case complex_kind_id:
        return dst_tp.get_data_size() > 2 * src_tp.get_data_size();
      case bytes_kind_id:
        return false;
      default:
        break;
      }
      break;
    case float_kind_id:
      switch (dst_tp.get_base_id()) {
      case bool_kind_id:
	return false;
      case int_kind_id:
	return false;
      case uint_kind_id:
        return false;
      case float_kind_id:
        return dst_tp.get_data_size() >= src_tp.get_data_size();
      case complex_kind_id:
        return dst_tp.get_data_size() >= 2 * src_tp.get_data_size();
      case bytes_kind_id:
        return false;
      default:
        break;
      }
      break;
    case complex_kind_id:
      switch (dst_tp.get_base_id()) {
      case bool_kind_id:
	return false;
      case int_kind_id:
	return false;
      case uint_kind_id:
	return false;
      case float_kind_id:
        return false;
      case complex_kind_id:
        return dst_tp.get_data_size() >= src_tp.get_data_size();
      case bytes_kind_id:
        return false;
      default:
        break;
      }
      break;
    case string_kind_id:
      switch (dst_tp.get_base_id()) {
      case bool_kind_id:
	return false;
      case int_kind_id:
	return false;
      case uint_kind_id:
        return false;
      case float_kind_id:
        return false;
      case complex_kind_id:
        return false;
      case bytes_kind_id:
        return false;
      default:
        break;
      }
      break;
    case bytes_kind_id:
      return dst_tp.get_base_id() == bytes_kind_id && dst_tp.get_data_size() == src_tp.get_data_size();
    default:
      break;
    }

    throw std::runtime_error("unhandled built-in case in is_lossless_assignmently");
  }

  // Use the available base_type to check the casting
  if (!dst_tp.is_builtin()) {
    // Call with dst_dt (the first parameter) first
    return dst_tp.extended()->is_lossless_assignment(dst_tp, src_tp);
  } else {
    // Fall back to src_dt if the dst's extended is NULL
    return src_tp.extended()->is_lossless_assignment(dst_tp, src_tp);
  }
}

constexpr size_t ndt::trivial_traits::metadata_size;
