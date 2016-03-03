//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/callable.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/parse.hpp>
#include <dynd/math.hpp>
#include <dynd/option.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

ndt::option_type::option_type(const type &value_tp)
    : base_type(option_id, value_tp.get_data_size(), value_tp.get_data_alignment(),
                value_tp.get_flags() & (type_flags_value_inherited | type_flags_operand_inherited),
                value_tp.get_arrmeta_size(), value_tp.get_ndim(), 0),
      m_value_tp(value_tp)
{
  if (value_tp.get_id() == option_id) {
    stringstream ss;
    ss << "Cannot construct an option type out of " << value_tp << ", it is already an option type";
    throw type_error(ss.str());
  }
}

void ndt::option_type::get_vars(std::unordered_set<std::string> &vars) const { m_value_tp.get_vars(vars); }

bool ndt::option_type::is_avail(const char *arrmeta, const char *data,
                                const eval::eval_context *DYND_UNUSED(ectx)) const
{
  if (m_value_tp.is_builtin()) {
    switch (m_value_tp.get_id()) {
    // Just use the known value assignments for these builtins
    case bool_id:
      return *reinterpret_cast<const unsigned char *>(data) <= 1;
    case int8_id:
      return *reinterpret_cast<const int8_t *>(data) != DYND_INT8_NA;
    case int16_id:
      return *reinterpret_cast<const int16_t *>(data) != DYND_INT16_NA;
    case int32_id:
      return *reinterpret_cast<const int32_t *>(data) != DYND_INT32_NA;
    case uint32_id:
      return *reinterpret_cast<const uint32_t *>(data) != DYND_UINT32_NA;
    case int64_id:
      return *reinterpret_cast<const int64_t *>(data) != DYND_INT64_NA;
    case int128_id:
      return *reinterpret_cast<const int128 *>(data) != DYND_INT128_NA;
    case float32_id:
      return !isnan(*reinterpret_cast<const float *>(data));
    case float64_id:
      return !isnan(*reinterpret_cast<const double *>(data));
    case complex_float32_id:
      return reinterpret_cast<const uint32_t *>(data)[0] != DYND_FLOAT32_NA_AS_UINT ||
             reinterpret_cast<const uint32_t *>(data)[1] != DYND_FLOAT32_NA_AS_UINT;
    case complex_float64_id:
      return reinterpret_cast<const uint64_t *>(data)[0] != DYND_FLOAT64_NA_AS_UINT ||
             reinterpret_cast<const uint64_t *>(data)[1] != DYND_FLOAT64_NA_AS_UINT;
    default:
      return false;
    }
  }
  else {
    nd::kernel_builder ckb;
    nd::callable &af = nd::is_na::get();
    type src_tp[1] = {type(this, true)};
    af.get()->instantiate(af->static_data(), NULL, &ckb, make_type<bool1>(), NULL, 1, src_tp, &arrmeta,
                          kernel_request_single, 0, NULL, std::map<std::string, type>());
    nd::kernel_prefix *ckp = ckb.get();
    char result;
    ckp->get_function<kernel_single_t>()(ckp, &result, const_cast<char **>(&data));
    return result == 0;
  }
}

void ndt::option_type::assign_na(const char *arrmeta, char *data, const eval::eval_context *DYND_UNUSED(ectx)) const
{
  if (m_value_tp.is_builtin()) {
    switch (m_value_tp.get_id()) {
    // Just use the known value assignments for these builtins
    case bool_id:
      *data = 2;
      return;
    case int8_id:
      *reinterpret_cast<int8_t *>(data) = DYND_INT8_NA;
      return;
    case int16_id:
      *reinterpret_cast<int16_t *>(data) = DYND_INT16_NA;
      return;
    case int32_id:
      *reinterpret_cast<int32_t *>(data) = DYND_INT32_NA;
      return;
    case int64_id:
      *reinterpret_cast<int64_t *>(data) = DYND_INT64_NA;
      return;
    case int128_id:
      *reinterpret_cast<int128 *>(data) = DYND_INT128_NA;
      return;
    case float32_id:
      *reinterpret_cast<uint32_t *>(data) = DYND_FLOAT32_NA_AS_UINT;
      return;
    case float64_id:
      *reinterpret_cast<uint64_t *>(data) = DYND_FLOAT64_NA_AS_UINT;
      return;
    case complex_float32_id:
      reinterpret_cast<uint32_t *>(data)[0] = DYND_FLOAT32_NA_AS_UINT;
      reinterpret_cast<uint32_t *>(data)[1] = DYND_FLOAT32_NA_AS_UINT;
      return;
    case complex_float64_id:
      reinterpret_cast<uint64_t *>(data)[0] = DYND_FLOAT64_NA_AS_UINT;
      reinterpret_cast<uint64_t *>(data)[1] = DYND_FLOAT64_NA_AS_UINT;
      return;
    default:
      break;
    }
  }
  else {
    nd::kernel_builder ckb;
    nd::callable &af = nd::assign_na::get();
    af.get()->instantiate(af->static_data(), NULL, &ckb, type(this, true), arrmeta, 0, NULL, NULL,
                          kernel_request_single, 0, NULL, std::map<std::string, type>());
    nd::kernel_prefix *ckp = ckb.get();
    ckp->get_function<kernel_single_t>()(ckp, data, NULL);
  }
}

void ndt::option_type::print_type(std::ostream &o) const { o << "?" << m_value_tp; }

void ndt::option_type::print_data(std::ostream &o, const char *arrmeta, const char *data) const
{
  if (is_avail(arrmeta, data, &eval::default_eval_context)) {
    m_value_tp.print_data(o, arrmeta, data);
  }
  else {
    o << "NA";
  }
}

bool ndt::option_type::is_expression() const
{
  // Even though the pointer is an instance of an base_expr_type,
  // we'll only call it an expression if the target is.
  return m_value_tp.is_expression();
}

bool ndt::option_type::is_unique_data_owner(const char *arrmeta) const
{
  if (m_value_tp.get_flags() & type_flag_blockref) {
    return m_value_tp.extended()->is_unique_data_owner(arrmeta);
  }
  return true;
}

void ndt::option_type::transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                                             type &out_transformed_tp, bool &out_was_transformed) const
{
  type tmp_tp;
  bool was_transformed = false;
  transform_fn(m_value_tp, arrmeta_offset + 0, extra, tmp_tp, was_transformed);
  if (was_transformed) {
    out_transformed_tp = make_type<option_type>(tmp_tp);
    out_was_transformed = true;
  }
  else {
    out_transformed_tp = type(this, true);
  }
}

ndt::type ndt::option_type::get_canonical_type() const
{
  return make_type<option_type>(m_value_tp.get_canonical_type());
}

void ndt::option_type::set_from_utf8_string(const char *arrmeta, char *data, const char *utf8_begin,
                                            const char *utf8_end, const eval::eval_context *ectx) const
{
  if (m_value_tp.get_base_id() != string_kind_id && parse_na(utf8_begin, utf8_end)) {
    assign_na(arrmeta, data, ectx);
  }
  else {
    if (m_value_tp.is_builtin()) {
      if (m_value_tp.unchecked_get_builtin_id() == bool_id) {
        *reinterpret_cast<bool1 *>(data) = parse<bool>(utf8_begin, utf8_end);
      }
      else {
        string_to_number(data, m_value_tp.unchecked_get_builtin_id(), utf8_begin, utf8_end, ectx->errmode);
      }
    }
    else {
      m_value_tp.extended()->set_from_utf8_string(arrmeta, data, utf8_begin, utf8_end, ectx);
    }
  }
}

ndt::type ndt::option_type::get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim) const
{
  if (i == 0) {
    return type(this, true);
  }
  else {
    return m_value_tp.get_type_at_dimension(inout_arrmeta, i, total_ndim);
  }
}

bool ndt::option_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
{
  if (dst_tp.extended() == this) {
    return ::is_lossless_assignment(m_value_tp, src_tp);
  }
  else {
    return ::is_lossless_assignment(dst_tp, m_value_tp);
  }
}

bool ndt::option_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_id() != option_id) {
    return false;
  }
  else {
    const option_type *ot = static_cast<const option_type *>(&rhs);
    return m_value_tp == ot->m_value_tp;
  }
}

void ndt::option_type::arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const
{
  if (!m_value_tp.is_builtin()) {
    m_value_tp.extended()->arrmeta_default_construct(arrmeta, blockref_alloc);
  }
}

void ndt::option_type::arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                              const intrusive_ptr<memory_block_data> &embedded_reference) const
{
  if (!m_value_tp.is_builtin()) {
    m_value_tp.extended()->arrmeta_copy_construct(dst_arrmeta, src_arrmeta, embedded_reference);
  }
}

void ndt::option_type::arrmeta_reset_buffers(char *arrmeta) const
{
  if (!m_value_tp.is_builtin()) {
    m_value_tp.extended()->arrmeta_reset_buffers(arrmeta);
  }
}

void ndt::option_type::arrmeta_finalize_buffers(char *arrmeta) const
{
  if (!m_value_tp.is_builtin()) {
    m_value_tp.extended()->arrmeta_finalize_buffers(arrmeta);
  }
}

void ndt::option_type::arrmeta_destruct(char *arrmeta) const
{
  if (!m_value_tp.is_builtin()) {
    m_value_tp.extended()->arrmeta_destruct(arrmeta);
  }
}

void ndt::option_type::arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const
{
  o << indent << "option arrmeta\n";
  if (!m_value_tp.is_builtin()) {
    m_value_tp.extended()->arrmeta_debug_print(arrmeta, o, indent + " ");
  }
}

void ndt::option_type::data_destruct(const char *arrmeta, char *data) const
{
  m_value_tp.extended()->data_destruct(arrmeta, data);
}

void ndt::option_type::data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const
{
  m_value_tp.extended()->data_destruct_strided(arrmeta, data, stride, count);
}

bool ndt::option_type::match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const
{
  if (candidate_tp.get_id() != option_id) {
    return false;
  }

  return m_value_tp.match(candidate_tp.extended<option_type>()->m_value_tp, tp_vars);
}

std::map<std::string, std::pair<ndt::type, void *>> ndt::option_type::get_dynamic_type_properties() const
{
  std::map<std::string, std::pair<ndt::type, void *>> properties;
  properties["value_type"] = {ndt::type("type"), (void *)(&m_value_tp)};

  return properties;
}
