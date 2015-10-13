//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/kernels/option_assignment_kernels.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/parser_util.hpp>
#include <dynd/func/apply.hpp>
#include <dynd/math.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

ndt::option_type::option_type(const type &value_tp)
    : base_type(option_type_id, option_kind, value_tp.get_data_size(), value_tp.get_data_alignment(),
                value_tp.get_flags() & (type_flags_value_inherited | type_flags_operand_inherited),
                value_tp.get_arrmeta_size(), value_tp.get_ndim(), 0),
      m_value_tp(value_tp)
{
  if (value_tp.get_type_id() == option_type_id) {
    stringstream ss;
    ss << "Cannot construct an option type out of " << value_tp << ", it is already an option type";
    throw type_error(ss.str());
  }
}

ndt::option_type::~option_type()
{
}

void ndt::option_type::get_vars(std::unordered_set<std::string> &vars) const
{
  m_value_tp.get_vars(vars);
}

bool ndt::option_type::is_avail(const char *arrmeta, const char *data, const eval::eval_context *ectx) const
{
  if (m_value_tp.is_builtin()) {
    switch (m_value_tp.get_type_id()) {
    // Just use the known value assignments for these builtins
    case bool_type_id:
      return *reinterpret_cast<const unsigned char *>(data) <= 1;
    case int8_type_id:
      return *reinterpret_cast<const int8_t *>(data) != DYND_INT8_NA;
    case int16_type_id:
      return *reinterpret_cast<const int16_t *>(data) != DYND_INT16_NA;
    case int32_type_id:
      return *reinterpret_cast<const int32_t *>(data) != DYND_INT32_NA;
    case int64_type_id:
      return *reinterpret_cast<const int64_t *>(data) != DYND_INT64_NA;
    case int128_type_id:
      return *reinterpret_cast<const int128 *>(data) != DYND_INT128_NA;
    case float32_type_id:
      return !isnan(*reinterpret_cast<const float *>(data));
    case float64_type_id:
      return !isnan(*reinterpret_cast<const double *>(data));
    case complex_float32_type_id:
      return reinterpret_cast<const uint32_t *>(data)[0] != DYND_FLOAT32_NA_AS_UINT ||
             reinterpret_cast<const uint32_t *>(data)[1] != DYND_FLOAT32_NA_AS_UINT;
    case complex_float64_type_id:
      return reinterpret_cast<const uint64_t *>(data)[0] != DYND_FLOAT64_NA_AS_UINT ||
             reinterpret_cast<const uint64_t *>(data)[1] != DYND_FLOAT64_NA_AS_UINT;
    default:
      return false;
    }
  } else {
    ckernel_builder<kernel_request_host> ckb;
    nd::callable &af = get_is_avail();
    type src_tp[1] = {type(this, true)};
    af.get()->instantiate(NULL, 0, NULL, &ckb, 0, type::make<bool1>(), NULL, 1, src_tp, &arrmeta, kernel_request_single,
                          ectx, 0, NULL, std::map<std::string, type>());
    ckernel_prefix *ckp = ckb.get();
    char result;
    ckp->get_function<expr_single_t>()(ckp, &result, const_cast<char **>(&data));
    return result != 0;
  }
}

void ndt::option_type::assign_na(const char *arrmeta, char *data, const eval::eval_context *ectx) const
{
  if (m_value_tp.is_builtin()) {
    switch (m_value_tp.get_type_id()) {
    // Just use the known value assignments for these builtins
    case bool_type_id:
      *data = 2;
      return;
    case int8_type_id:
      *reinterpret_cast<int8_t *>(data) = DYND_INT8_NA;
      return;
    case int16_type_id:
      *reinterpret_cast<int16_t *>(data) = DYND_INT16_NA;
      return;
    case int32_type_id:
      *reinterpret_cast<int32_t *>(data) = DYND_INT32_NA;
      return;
    case int64_type_id:
      *reinterpret_cast<int64_t *>(data) = DYND_INT64_NA;
      return;
    case int128_type_id:
      *reinterpret_cast<int128 *>(data) = DYND_INT128_NA;
      return;
    case float32_type_id:
      *reinterpret_cast<uint32_t *>(data) = DYND_FLOAT32_NA_AS_UINT;
      return;
    case float64_type_id:
      *reinterpret_cast<uint64_t *>(data) = DYND_FLOAT64_NA_AS_UINT;
      return;
    case complex_float32_type_id:
      reinterpret_cast<uint32_t *>(data)[0] = DYND_FLOAT32_NA_AS_UINT;
      reinterpret_cast<uint32_t *>(data)[1] = DYND_FLOAT32_NA_AS_UINT;
      return;
    case complex_float64_type_id:
      reinterpret_cast<uint64_t *>(data)[0] = DYND_FLOAT64_NA_AS_UINT;
      reinterpret_cast<uint64_t *>(data)[1] = DYND_FLOAT64_NA_AS_UINT;
      return;
    default:
      break;
    }
  } else {
    ckernel_builder<kernel_request_host> ckb;
    nd::callable &af = get_assign_na();
    af.get()->instantiate(NULL, 0, NULL, &ckb, 0, type(this, true), arrmeta, 0, NULL, NULL, kernel_request_single, ectx,
                          0, NULL, std::map<std::string, type>());
    ckernel_prefix *ckp = ckb.get();
    ckp->get_function<expr_single_t>()(ckp, data, NULL);
  }
}

void ndt::option_type::print_data(std::ostream &o, const char *arrmeta, const char *data) const
{
  if (is_avail(arrmeta, data, &eval::default_eval_context)) {
    m_value_tp.print_data(o, arrmeta, data);
  } else {
    o << "NA";
  }
}

void ndt::option_type::print_type(std::ostream &o) const
{
  o << "?" << m_value_tp;
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
    out_transformed_tp = make(tmp_tp);
    out_was_transformed = true;
  } else {
    out_transformed_tp = type(this, true);
  }
}

ndt::type ndt::option_type::get_canonical_type() const
{
  return make(m_value_tp.get_canonical_type());
}

void ndt::option_type::set_from_utf8_string(const char *arrmeta, char *data, const char *utf8_begin,
                                            const char *utf8_end, const eval::eval_context *ectx) const
{
  if (m_value_tp.get_kind() != string_kind && m_value_tp.get_kind() != dynamic_kind &&
      parse::matches_option_type_na_token(utf8_begin, utf8_end)) {
    assign_na(arrmeta, data, ectx);
  } else {
    if (m_value_tp.is_builtin()) {
      if (m_value_tp.unchecked_get_builtin_type_id() == bool_type_id) {
        parse::string_to_bool(data, utf8_begin, utf8_end, false, ectx->errmode);
      } else {
        parse::string_to_number(data, m_value_tp.unchecked_get_builtin_type_id(), utf8_begin, utf8_end, false,
                                ectx->errmode);
      }
    } else {
      m_value_tp.extended()->set_from_utf8_string(arrmeta, data, utf8_begin, utf8_end, ectx);
    }
  }
}

ndt::type ndt::option_type::get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim) const
{
  if (i == 0) {
    return type(this, true);
  } else {
    return m_value_tp.get_type_at_dimension(inout_arrmeta, i, total_ndim);
  }
}

bool ndt::option_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
{
  if (dst_tp.extended() == this) {
    return ::is_lossless_assignment(m_value_tp, src_tp);
  } else {
    return ::is_lossless_assignment(dst_tp, m_value_tp);
  }
}

bool ndt::option_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != option_type_id) {
    return false;
  } else {
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
                                              memory_block_data *embedded_reference) const
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

intptr_t ndt::option_type::make_assignment_kernel(void *ckb, intptr_t ckb_offset, const type &dst_tp,
                                                  const char *dst_arrmeta, const type &src_tp, const char *src_arrmeta,
                                                  kernel_request_t kernreq, const eval::eval_context *ectx) const
{
  // Let expression types resolve themselves first
  if (this == dst_tp.extended() && src_tp.get_kind() == expr_kind) {
    return src_tp.extended()->make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta, kernreq,
                                                     ectx);
  }

  return kernels::make_option_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta, kernreq,
                                                ectx);
}

bool ndt::option_type::match(const char *arrmeta, const type &candidate_tp, const char *candidate_arrmeta,
                             std::map<std::string, type> &tp_vars) const
{
  if (candidate_tp.get_type_id() != option_type_id) {
    return false;
  }

  return m_value_tp.match(arrmeta, candidate_tp.extended<option_type>()->m_value_tp, candidate_arrmeta, tp_vars);
}

static ndt::type property_get_value_type(ndt::type tp)
{
  const ndt::option_type *pd = tp.extended<ndt::option_type>();
  return pd->get_value_type();
}

/*
static nd::array property_get_is_avail(const ndt::type &tp)
{
  const ndt::option_type *pd = tp.extended<ndt::option_type>();
  return pd->get_is_avail();
}

static nd::array property_get_assign_na(const ndt::type &tp)
{
  const ndt::option_type *pd = tp.extended<ndt::option_type>();
  return pd->get_assign_na();
}
*/

void ndt::option_type::get_dynamic_type_properties(const std::pair<std::string, nd::callable> **out_properties,
                                                   size_t *out_count) const
{
  static pair<std::string, nd::callable> type_properties[] = {
      pair<std::string, nd::callable>("value_type", nd::functional::apply(&property_get_value_type, "self")),
      //      pair<string, gfunc::callable>(
      //        "is_avail", gfunc::make_callable(&property_get_is_avail,
      //        "self")),
      //  pair<string, gfunc::callable>(
      //    "assign_na", gfunc::make_callable(&property_get_assign_na, "self")),
  };

  *out_properties = type_properties;
  *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}

namespace {
// TODO: use the PP meta stuff, but DYND_PP_LEN_MAX is set to 8 right now,
// would need to be 19
struct static_options {
  ndt::option_type bt1;
  ndt::option_type bt2;
  ndt::option_type bt3;
  ndt::option_type bt4;
  ndt::option_type bt5;
  ndt::option_type bt6;
  ndt::option_type bt7;
  ndt::option_type bt8;
  ndt::option_type bt9;
  ndt::option_type bt10;
  ndt::option_type bt11;
  ndt::option_type bt12;
  ndt::option_type bt13;
  ndt::option_type bt14;
  ndt::option_type bt15;
  ndt::option_type bt16;
  ndt::option_type bt17;
  ndt::option_type bt18;

  ndt::type static_builtins_instance[builtin_type_id_count];

  static_options()
      : bt1(ndt::type((type_id_t)1)), bt2(ndt::type((type_id_t)2)), bt3(ndt::type((type_id_t)3)),
        bt4(ndt::type((type_id_t)4)), bt5(ndt::type((type_id_t)5)), bt6(ndt::type((type_id_t)6)),
        bt7(ndt::type((type_id_t)7)), bt8(ndt::type((type_id_t)8)), bt9(ndt::type((type_id_t)9)),
        bt10(ndt::type((type_id_t)10)), bt11(ndt::type((type_id_t)11)), bt12(ndt::type((type_id_t)12)),
        bt13(ndt::type((type_id_t)13)), bt14(ndt::type((type_id_t)14)), bt15(ndt::type((type_id_t)15)),
        bt16(ndt::type((type_id_t)16)), bt17(ndt::type((type_id_t)17)), bt18(ndt::type((type_id_t)18))
  {
    static_builtins_instance[1] = ndt::type(&bt1, true);
    static_builtins_instance[2] = ndt::type(&bt2, true);
    static_builtins_instance[3] = ndt::type(&bt3, true);
    static_builtins_instance[4] = ndt::type(&bt4, true);
    static_builtins_instance[5] = ndt::type(&bt5, true);
    static_builtins_instance[6] = ndt::type(&bt6, true);
    static_builtins_instance[7] = ndt::type(&bt7, true);
    static_builtins_instance[8] = ndt::type(&bt8, true);
    static_builtins_instance[9] = ndt::type(&bt9, true);
    static_builtins_instance[10] = ndt::type(&bt10, true);
    static_builtins_instance[11] = ndt::type(&bt11, true);
    static_builtins_instance[12] = ndt::type(&bt12, true);
    static_builtins_instance[13] = ndt::type(&bt13, true);
    static_builtins_instance[14] = ndt::type(&bt14, true);
    static_builtins_instance[15] = ndt::type(&bt15, true);
    static_builtins_instance[16] = ndt::type(&bt16, true);
    static_builtins_instance[17] = ndt::type(&bt17, true);
    static_builtins_instance[18] = ndt::type(&bt18, true);
  }
};
} // anonymous namespace

ndt::type ndt::option_type::make(const type &value_tp)
{
  // Static instances of the types, which have a reference
  // count > 0 for the lifetime of the program. This static
  // construction is inside a function to ensure correct creation
  // order during startup.
  static static_options so;

  if (value_tp.is_builtin()) {
    return so.static_builtins_instance[value_tp.get_type_id()];
  } else {
    return type(new option_type(value_tp), false);
  }
}
