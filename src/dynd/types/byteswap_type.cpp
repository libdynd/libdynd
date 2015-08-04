//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/byteswap_type.hpp>
#include <dynd/types/fixed_bytes_type.hpp>
#include <dynd/buffer_storage.hpp>
#include <dynd/kernels/byteswap_kernels.hpp>

using namespace std;
using namespace dynd;

ndt::byteswap_type::byteswap_type(const type &value_type)
    : base_expr_type(byteswap_type_id, expr_kind, value_type.get_data_size(),
                     value_type.get_data_alignment(), type_flag_none, 0),
      m_value_type(value_type),
      m_operand_type(make_fixed_bytes(value_type.get_data_size(),
                                      value_type.get_data_alignment()))
{
  if (!value_type.is_builtin()) {
    throw dynd::type_error(
        "byteswap_type: Only built-in types are supported presently");
  }
}

ndt::byteswap_type::byteswap_type(const type &value_type,
                                  const type &operand_type)
    : base_expr_type(byteswap_type_id, expr_kind, operand_type.get_data_size(),
                     operand_type.get_data_alignment(), type_flag_none, 0),
      m_value_type(value_type), m_operand_type(operand_type)
{
  // Only a bytes type be the operand to the byteswap
  if (operand_type.value_type().get_type_id() != fixed_bytes_type_id) {
    std::stringstream ss;
    ss << "byteswap_type: The operand to the type must have a value type of "
          "bytes, not " << operand_type.value_type();
    throw dynd::type_error(ss.str());
  }
  // Automatically realign if needed
  if (operand_type.value_type().get_data_alignment() <
      value_type.get_data_alignment()) {
    m_operand_type = view_type::make(
        operand_type, make_fixed_bytes(operand_type.get_data_size(),
                                       value_type.get_data_alignment()));
  }
}

ndt::byteswap_type::~byteswap_type()
{
}

void ndt::byteswap_type::print_data(std::ostream &DYND_UNUSED(o),
                                    const char *DYND_UNUSED(arrmeta),
                                    const char *DYND_UNUSED(data)) const
{
  throw runtime_error(
      "internal error: byteswap_type::print_data isn't supposed to be called");
}

void ndt::byteswap_type::print_type(std::ostream &o) const
{
  o << "byteswap[" << m_value_type;
  if (m_operand_type.get_type_id() != fixed_bytes_type_id) {
    o << ", " << m_operand_type;
  }
  o << "]";
}

bool ndt::byteswap_type::is_lossless_assignment(const type &dst_tp,
                                                const type &src_tp) const
{
  // Treat this type as the value type for whether assignment is always lossless
  if (src_tp.extended() == this) {
    return ::dynd::is_lossless_assignment(dst_tp, m_value_type);
  } else {
    return ::dynd::is_lossless_assignment(m_value_type, src_tp);
  }
}

bool ndt::byteswap_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != byteswap_type_id) {
    return false;
  } else {
    const byteswap_type *dt = static_cast<const byteswap_type *>(&rhs);
    return m_value_type == dt->m_value_type;
  }
}

ndt::type ndt::byteswap_type::with_replaced_storage_type(
    const type &replacement_type) const
{
  if (m_operand_type.get_kind() != expr_kind) {
    // If there's no expression in the operand, just try substituting (the
    // constructor will error-check)
    return type(new byteswap_type(m_value_type, replacement_type), false);
  } else {
    // With an expression operand, replace it farther down the chain
    return type(
        new byteswap_type(m_value_type,
                          reinterpret_cast<const base_expr_type *>(
                              replacement_type.extended())
                              ->with_replaced_storage_type(replacement_type)),
        false);
  }
}

size_t ndt::byteswap_type::make_operand_to_value_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const char *DYND_UNUSED(dst_arrmeta),
    const char *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
    const eval::eval_context *DYND_UNUSED(ectx)) const
{
  if (m_value_type.get_kind() != complex_kind) {
    return make_byteswap_assignment_function(
        ckb, ckb_offset, m_value_type.get_data_size(),
        m_value_type.get_data_alignment(), kernreq);
  } else {
    return make_pairwise_byteswap_assignment_function(
        ckb, ckb_offset, m_value_type.get_data_size(),
        m_value_type.get_data_alignment(), kernreq);
  }
}

size_t ndt::byteswap_type::make_value_to_operand_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const char *DYND_UNUSED(dst_arrmeta),
    const char *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
    const eval::eval_context *DYND_UNUSED(ectx)) const
{
  if (m_value_type.get_kind() != complex_kind) {
    return make_byteswap_assignment_function(
        ckb, ckb_offset, m_value_type.get_data_size(),
        m_value_type.get_data_alignment(), kernreq);
  } else {
    return make_pairwise_byteswap_assignment_function(
        ckb, ckb_offset, m_value_type.get_data_size(),
        m_value_type.get_data_alignment(), kernreq);
  }
}
