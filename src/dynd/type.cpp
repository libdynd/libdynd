//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/base_dim_type.hpp>
#include <dynd/types/base_memory_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/type_pattern_match.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/buffer_storage.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/make_callable.hpp>
#include <dynd/func/call_callable.hpp>
#include <dynd/func/arrfunc.hpp>

#include <dynd/types/convert_type.hpp>
#include <dynd/types/datashape_parser.hpp>

#include <sstream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <functional>
#include <iterator>
#include <iomanip>

using namespace std;
using namespace dynd;

char *dynd::iterdata_broadcasting_terminator_incr(iterdata_common *iterdata,
                                                  intptr_t DYND_UNUSED(level))
{
  // This repeats the same data over and over again, broadcasting additional
  // leftmost dimensions
  iterdata_broadcasting_terminator *id =
      reinterpret_cast<iterdata_broadcasting_terminator *>(iterdata);
  return id->data;
}

char *dynd::iterdata_broadcasting_terminator_adv(iterdata_common *iterdata,
                                                 intptr_t DYND_UNUSED(level),
                                                 intptr_t DYND_UNUSED(i))
{
  // This repeats the same data over and over again, broadcasting additional
  // leftmost dimensions
  iterdata_broadcasting_terminator *id =
      reinterpret_cast<iterdata_broadcasting_terminator *>(iterdata);
  return id->data;
}

char *dynd::iterdata_broadcasting_terminator_reset(iterdata_common *iterdata,
                                                   char *data,
                                                   intptr_t DYND_UNUSED(level))
{
  iterdata_broadcasting_terminator *id =
      reinterpret_cast<iterdata_broadcasting_terminator *>(iterdata);
  id->data = data;
  return data;
}

const ndt::type ndt::static_builtin_types[builtin_type_id_count] = {
    ndt::type(uninitialized_type_id),   ndt::type(bool_type_id),
    ndt::type(int8_type_id),            ndt::type(int16_type_id),
    ndt::type(int32_type_id),           ndt::type(int64_type_id),
    ndt::type(int128_type_id),          ndt::type(uint8_type_id),
    ndt::type(uint16_type_id),          ndt::type(uint32_type_id),
    ndt::type(uint64_type_id),          ndt::type(uint128_type_id),
    ndt::type(float16_type_id),         ndt::type(float32_type_id),
    ndt::type(float64_type_id),         ndt::type(float128_type_id),
    ndt::type(complex_float32_type_id), ndt::type(complex_float64_type_id),
    ndt::type(void_type_id)};

ndt::type::type(const std::string &rep) : m_extended(NULL)
{
  type_from_datashape(rep).swap(*this);
}

ndt::type::type(const char *rep_begin, const char *rep_end) : m_extended(NULL)
{
  type_from_datashape(rep_begin, rep_end).swap(*this);
}

ndt::type ndt::type::at_array(int nindices, const irange *indices) const
{
  if (this->is_builtin()) {
    if (nindices == 0) {
      return *this;
    } else {
      throw too_many_indices(*this, nindices, 0);
    }
  } else {
    return m_extended->apply_linear_index(nindices, indices, 0, *this, true);
  }
}

bool ndt::type::matches(const char *arrmeta, const ndt::type &other,
                        std::map<nd::string, ndt::type> &tp_vars) const
{
  // dispatch to patterns first
  // then categories
  // then concrete

  if (other.is_symbolic()) {
    return other.extended()->matches(arrmeta, *this, tp_vars);
  } else if (is_builtin()) {
    if (other.is_builtin()) {
      return get_type_id() == other.get_type_id();
    } else {
      return other.extended()->matches(arrmeta, *this, tp_vars);
    }
  }

  return extended()->matches(arrmeta, other, tp_vars);
}

bool ndt::type::matches(const ndt::type &other) const
{
  if (extended() == other.extended()) {
    return true;
  } else {
    std::map<nd::string, ndt::type> tp_vars;
    return matches(NULL, other, tp_vars);
  }
}

nd::array ndt::type::p(const char *property_name) const
{
  if (!is_builtin()) {
    const std::pair<std::string, gfunc::callable> *properties;
    size_t count;
    extended()->get_dynamic_type_properties(&properties, &count);
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
  ss << "dynd type does not have property " << property_name;
  throw runtime_error(ss.str());
}

nd::array ndt::type::p(const std::string &property_name) const
{
  if (!is_builtin()) {
    const std::pair<std::string, gfunc::callable> *properties;
    size_t count;
    extended()->get_dynamic_type_properties(&properties, &count);
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
  ss << "dynd type does not have property " << property_name;
  throw runtime_error(ss.str());
}

ndt::type ndt::type::apply_linear_index(intptr_t nindices,
                                        const irange *indices, size_t current_i,
                                        const ndt::type &root_tp,
                                        bool leading_dimension) const
{
  if (is_builtin()) {
    if (nindices == 0) {
      return *this;
    } else {
      throw too_many_indices(*this, nindices + current_i, current_i);
    }
  } else {
    return m_extended->apply_linear_index(nindices, indices, current_i, root_tp,
                                          leading_dimension);
  }
}

namespace {
struct replace_scalar_type_extra {
  replace_scalar_type_extra(const ndt::type &dt) : scalar_tp(dt) {}
  const ndt::type &scalar_tp;
};
static void replace_scalar_types(const ndt::type &dt,
                                 intptr_t DYND_UNUSED(arrmeta_offset),
                                 void *extra, ndt::type &out_transformed_tp,
                                 bool &out_was_transformed)
{
  const replace_scalar_type_extra *e =
      reinterpret_cast<const replace_scalar_type_extra *>(extra);
  if (dt.is_scalar()) {
    out_transformed_tp = ndt::make_convert(e->scalar_tp, dt);
    out_was_transformed = true;
  } else {
    dt.extended()->transform_child_types(&replace_scalar_types, 0, extra,
                                         out_transformed_tp,
                                         out_was_transformed);
  }
}
} // anonymous namespace

ndt::type
ndt::type::with_replaced_scalar_types(const ndt::type &scalar_tp) const
{
  ndt::type result;
  bool was_transformed;
  replace_scalar_type_extra extra(scalar_tp);
  replace_scalar_types(*this, 0, &extra, result, was_transformed);
  return result;
}

namespace {
struct replace_dtype_extra {
  replace_dtype_extra(const ndt::type &replacement_tp, intptr_t replace_ndim)
      : m_replacement_tp(replacement_tp), m_replace_ndim(replace_ndim)
  {
  }
  const ndt::type &m_replacement_tp;
  intptr_t m_replace_ndim;
};
static void replace_dtype(const ndt::type &tp,
                          intptr_t DYND_UNUSED(arrmeta_offset), void *extra,
                          ndt::type &out_transformed_tp,
                          bool &out_was_transformed)
{
  const replace_dtype_extra *e =
      reinterpret_cast<const replace_dtype_extra *>(extra);
  if (tp.get_ndim() == e->m_replace_ndim) {
    out_transformed_tp = e->m_replacement_tp;
    out_was_transformed = true;
  } else {
    tp.extended()->transform_child_types(
        &replace_dtype, 0, extra, out_transformed_tp, out_was_transformed);
  }
}
} // anonymous namespace

ndt::type ndt::type::with_replaced_dtype(const ndt::type &replacement_tp,
                                         intptr_t replace_ndim) const
{
  ndt::type result;
  bool was_transformed;
  replace_dtype_extra extra(replacement_tp, replace_ndim);
  replace_dtype(*this, 0, &extra, result, was_transformed);
  return result;
}

ndt::type ndt::type::without_memory_type() const
{
  if (get_kind() == memory_kind) {
    return extended<base_memory_type>()->get_element_type();
  } else {
    return *this;
  }
}

ndt::type ndt::type::with_new_axis(intptr_t i, intptr_t new_ndim) const
{
  return with_replaced_dtype(
      ndt::make_fixed_dim(1, get_type_at_dimension(NULL, i), new_ndim),
      get_ndim() - i);
}

intptr_t ndt::type::get_dim_size(const char *arrmeta, const char *data) const
{
  if (get_kind() == dim_kind) {
    return static_cast<const base_dim_type *>(m_extended)
        ->get_dim_size(arrmeta, data);
  } else if (get_kind() == struct_kind) {
    return static_cast<const base_struct_type *>(m_extended)->get_field_count();
  } else if (get_ndim() > 0) {
    intptr_t dim_size = -1;
    m_extended->get_shape(1, 0, &dim_size, arrmeta, data);
    if (dim_size >= 0) {
      return dim_size;
    }
  }

  std::stringstream ss;
  ss << "Scalar dynd array of type " << *this << " has no length";
  throw std::invalid_argument(ss.str());
}

bool ndt::type::get_as_strided(const char *arrmeta, intptr_t *out_dim_size,
                               intptr_t *out_stride, ndt::type *out_el_tp,
                               const char **out_el_arrmeta) const
{
  if (get_kind() == memory_kind) {
    bool res = without_memory_type().get_as_strided(
        arrmeta, out_dim_size, out_stride, out_el_tp, out_el_arrmeta);
    *out_el_tp =
        extended<base_memory_type>()->with_replaced_storage_type(*out_el_tp);
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

bool ndt::type::get_as_strided(const char *arrmeta, intptr_t ndim,
                               const size_stride_t **out_size_stride,
                               ndt::type *out_el_tp,
                               const char **out_el_arrmeta) const
{
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
bool ndt::type::data_layout_compatible_with(const ndt::type &rhs) const
{
  if (extended() == rhs.extended()) {
    // If they're trivially identical, quickly return true
    return true;
  }
  if (get_data_size() != rhs.get_data_size() ||
      get_arrmeta_size() != rhs.get_arrmeta_size()) {
    // The size of the data and arrmeta must be the same
    return false;
  }
  if (get_arrmeta_size() == 0 && is_pod() && rhs.is_pod()) {
    // If both are POD with no arrmeta, then they're compatible
    return true;
  }
  if (get_kind() == expr_kind || rhs.get_kind() == expr_kind) {
    // If either is an expression type, check compatibility with
    // the storage types
    return storage_type().data_layout_compatible_with(rhs.storage_type());
  }
  // Rules for the rest of the types
  switch (get_type_id()) {
  case string_type_id:
  case bytes_type_id:
  case json_type_id:
    switch (rhs.get_type_id()) {
    case string_type_id:
    case bytes_type_id:
    case json_type_id:
      // All of string, bytes, json are compatible
      return true;
    default:
      return false;
    }
  case cfixed_dim_type_id:
    // For fixed dimensions, it's data layout compatible if
    // the shape and strides match, and the element is data
    // layout compatible.
    if (rhs.get_type_id() == cfixed_dim_type_id) {
      const cfixed_dim_type *fdd =
          static_cast<const cfixed_dim_type *>(extended());
      const cfixed_dim_type *rhs_fdd = rhs.extended<cfixed_dim_type>();
      return fdd->get_fixed_dim_size() == rhs_fdd->get_fixed_dim_size() &&
             fdd->get_fixed_stride() == rhs_fdd->get_fixed_stride() &&
             fdd->get_element_type().data_layout_compatible_with(
                 rhs_fdd->get_element_type());
    }
    break;
  case fixed_dim_type_id:
    if (rhs.get_type_id() == fixed_dim_type_id) {
      return extended<fixed_dim_type>()->get_fixed_dim_size() ==
                 rhs.extended<fixed_dim_type>()->get_fixed_dim_size() &&
             extended<fixed_dim_type>()
                 ->get_element_type()
                 .data_layout_compatible_with(
                     rhs.extended<fixed_dim_type>()->get_element_type());
    }
    break;
  case var_dim_type_id:
    // For var dimensions, it's data layout
    // compatible if the element is
    if (rhs.get_type_id() == var_dim_type_id) {
      const base_dim_type *budd =
          static_cast<const base_dim_type *>(extended());
      const base_dim_type *rhs_budd = rhs.extended<base_dim_type>();
      return budd->get_element_type().data_layout_compatible_with(
          rhs_budd->get_element_type());
    }
    break;
  default:
    break;
  }
  return false;
}

std::ostream &dynd::ndt::operator<<(std::ostream &o, const ndt::type &rhs)
{
  switch (rhs.get_type_id()) {
  case uninitialized_type_id:
    o << "uninitialized";
    break;
  case bool_type_id:
    o << "bool";
    break;
  case int8_type_id:
    o << "int8";
    break;
  case int16_type_id:
    o << "int16";
    break;
  case int32_type_id:
    o << "int32";
    break;
  case int64_type_id:
    o << "int64";
    break;
  case int128_type_id:
    o << "int128";
    break;
  case uint8_type_id:
    o << "uint8";
    break;
  case uint16_type_id:
    o << "uint16";
    break;
  case uint32_type_id:
    o << "uint32";
    break;
  case uint64_type_id:
    o << "uint64";
    break;
  case uint128_type_id:
    o << "uint128";
    break;
  case float16_type_id:
    o << "float16";
    break;
  case float32_type_id:
    o << "float32";
    break;
  case float64_type_id:
    o << "float64";
    break;
  case float128_type_id:
    o << "float128";
    break;
  case complex_float32_type_id:
    o << "complex[float32]";
    break;
  case complex_float64_type_id:
    o << "complex[float64]";
    break;
  case void_type_id:
    o << "void";
    break;
  default:
    rhs.extended()->print_type(o);
    break;
  }

  return o;
}

ndt::type ndt::make_type(intptr_t ndim, const intptr_t *shape,
                         const ndt::type &dtp)
{
  if (ndim > 0) {
    ndt::type result_tp = shape[ndim - 1] >= 0
                              ? ndt::make_fixed_dim(shape[ndim - 1], dtp)
                              : ndt::make_var_dim(dtp);
    for (intptr_t i = ndim - 2; i >= 0; --i) {
      if (shape[i] >= 0) {
        result_tp = ndt::make_fixed_dim(shape[i], result_tp);
      } else {
        result_tp = ndt::make_var_dim(result_tp);
      }
    }
    return result_tp;
  } else {
    return dtp;
  }
}

ndt::type ndt::make_type(intptr_t ndim, const intptr_t *shape,
                         const ndt::type &dtp, bool &out_any_var)
{
  if (ndim > 0) {
    ndt::type result_tp = dtp;
    for (intptr_t i = ndim - 1; i >= 0; --i) {
      if (shape[i] >= 0) {
        result_tp = ndt::make_fixed_dim(shape[i], result_tp);
      } else {
        result_tp = ndt::make_var_dim(result_tp);
        out_any_var = true;
      }
    }
    return result_tp;
  } else {
    return dtp;
  }
}

ndt::type ndt::as_type(const nd::array &value) { return value.get_type(); }

ndt::type ndt::as_type(const nd::arrfunc &value)
{
  return value.get_array_type();
}

ndt::type ndt::get_forward_type(const nd::array &val)
{
  if (val.get_type().get_type_id() == arrfunc_type_id) {
    return val.get_type();
  }
  /*
    if ((val.get_access_flags() & nd::write_access_flag) == 0) {
      throw std::runtime_error("TODO: how to handle readonly/immutable arrays "
                               "in struct/tuple packing");
    }
  */

  return make_pointer(val.get_type());
}

ndt::type ndt::get_forward_type(const nd::arrfunc &value)
{
  return value.get_array_type();
}

template <class T, class Tas>
static void print_as(std::ostream &o, const char *data)
{
  T value;
  memcpy(&value, data, sizeof(value));
  o << static_cast<Tas>(value);
}

void dynd::hexadecimal_print(std::ostream &o, char value)
{
  static char hexadecimal[] = "0123456789abcdef";
  unsigned char v = (unsigned char)value;
  o << hexadecimal[v >> 4] << hexadecimal[v & 0x0f];
}

void dynd::hexadecimal_print(std::ostream &o, unsigned char value)
{
  hexadecimal_print(o, static_cast<char>(value));
}

void dynd::hexadecimal_print(std::ostream &o, unsigned short value)
{
  // Standard printing is in big-endian order
  hexadecimal_print(o, static_cast<char>((value >> 8) & 0xff));
  hexadecimal_print(o, static_cast<char>(value & 0xff));
}

void dynd::hexadecimal_print(std::ostream &o, unsigned int value)
{
  // Standard printing is in big-endian order
  hexadecimal_print(o, static_cast<char>(value >> 24));
  hexadecimal_print(o, static_cast<char>((value >> 16) & 0xff));
  hexadecimal_print(o, static_cast<char>((value >> 8) & 0xff));
  hexadecimal_print(o, static_cast<char>(value & 0xff));
}

void dynd::hexadecimal_print(std::ostream &o, unsigned long value)
{
  if (sizeof(unsigned int) == sizeof(unsigned long)) {
    hexadecimal_print(o, static_cast<unsigned int>(value));
  } else {
    hexadecimal_print(o, static_cast<unsigned long long>(value));
  }
}

void dynd::hexadecimal_print(std::ostream &o, unsigned long long value)
{
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

void dynd::hexadecimal_print(std::ostream &o, const char *data,
                             intptr_t element_size)
{
  for (int i = 0; i < element_size; ++i, ++data) {
    hexadecimal_print(o, *data);
  }
}

void dynd::hexadecimal_print_summarized(std::ostream &o, const char *data,
                                        intptr_t element_size,
                                        intptr_t summary_size)
{
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

static intptr_t line_count(const string &s)
{
  return 1 + count_if(s.begin(), s.end(), bind1st(equal_to<char>(), '\n'));
}

static void summarize_stats(const string &s, intptr_t &num_rows,
                            intptr_t &max_num_cols)
{
  num_rows += line_count(s);
  max_num_cols = max(max_num_cols, (intptr_t)s.size());
}

void dynd::print_indented(ostream &o, const string &indent, const string &s,
                          bool skipfirstline)
{
  const char *begin = s.data();
  const char *end = s.data() + s.size();
  const char *cur = begin;
  while (cur != end) {
    const char *next = find_if(cur, end, bind1st(equal_to<char>(), '\n'));
    if (*next == '\n')
      ++next;
    if (!skipfirstline || cur != begin)
      o << indent;
    o.write(cur, next - cur);
    cur = next;
  }
}

// TODO Move the magic numbers into parameters
void dynd::strided_array_summarized(std::ostream &o, const ndt::type &tp,
                                    const char *arrmeta, const char *data,
                                    intptr_t dim_size, intptr_t stride)
{
  const int leading_count = 7, trailing_count = 3, row_threshold = 10,
            col_threshold = 30, packing_cols = 75;

  vector<string> leading, trailing;
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
  for (itrail = max(dim_size - trailing_count, ilead + 1); itrail < dim_size;
       ++itrail) {
    stringstream ss;
    tp.print_data(ss, arrmeta, data + itrail * stride);
    trailing.push_back(ss.str());
    summarize_stats(trailing.back(), num_rows, max_num_cols);
  }
  itrail = dim_size - trailing.size() - 1;

  // Select between two printing strategies depending on what we got
  if ((size_t)num_rows > (leading.size() + trailing.size()) ||
      max_num_cols > col_threshold) {
    // Trim the leading/trailing vectors until we get to our threshold
    while (num_rows > row_threshold &&
           (trailing.size() > 1 || leading.size() > 1)) {
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
    intptr_t total_cols =
        (max_num_cols + 2) * (leading.size() + trailing.size());
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
        leading.erase(leading.begin() + (leading.size() / per_row) * per_row,
                      leading.end());
      }
      if (trailing.size() > (size_t)per_row && trailing.size() % per_row != 0) {
        trailing.erase(trailing.begin(),
                       trailing.begin() + trailing.size() % per_row);
      }
    }

    intptr_t i = 0, j;
    intptr_t i_size = leading.size();
    if (!i_size)
      o << '[';
    while (i < i_size) {
      o << ((i == 0) ? "[" : " ") << setw(max_num_cols) << leading[i]
        << setw(0);
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

void dynd::print_builtin_scalar(type_id_t type_id, std::ostream &o,
                                const char *data)
{
  switch (type_id) {
  case bool_type_id:
    o << (*data ? "True" : "False");
    break;
  case int8_type_id:
    print_as<int8_t, int32_t>(o, data);
    break;
  case int16_type_id:
    print_as<int16_t, int32_t>(o, data);
    break;
  case int32_type_id:
    print_as<int32_t, int32_t>(o, data);
    break;
  case int64_type_id:
    print_as<int64_t, int64_t>(o, data);
    break;
  case int128_type_id:
    print_as<dynd_int128, dynd_int128>(o, data);
    break;
  case uint8_type_id:
    print_as<uint8_t, uint32_t>(o, data);
    break;
  case uint16_type_id:
    print_as<uint16_t, uint32_t>(o, data);
    break;
  case uint32_type_id:
    print_as<uint32_t, uint32_t>(o, data);
    break;
  case uint64_type_id:
    print_as<uint64_t, uint64_t>(o, data);
    break;
  case uint128_type_id:
    print_as<dynd_uint128, dynd_uint128>(o, data);
    break;
  case float16_type_id:
    print_as<dynd_float16, float>(o, data);
    break;
  case float32_type_id:
    print_as<float, float>(o, data);
    break;
  case float64_type_id:
    print_as<double, double>(o, data);
    break;
  case float128_type_id:
    print_as<dynd_float128, dynd_float128>(o, data);
    break;
  case complex_float32_type_id:
    print_as<complex<float>, complex<float>>(o, data);
    break;
  case complex_float64_type_id:
    print_as<complex<double>, complex<double>>(o, data);
    break;
  case void_type_id:
    o << "(void)";
    break;
  default:
    stringstream ss;
    ss << "printing of dynd builtin type id " << type_id
       << " isn't supported yet";
    throw dynd::type_error(ss.str());
  }
}

void ndt::type::print_data(std::ostream &o, const char *arrmeta,
                           const char *data) const
{
  if (is_builtin()) {
    print_builtin_scalar(get_type_id(), o, data);
  } else {
    extended()->print_data(o, arrmeta, data);
  }
}
