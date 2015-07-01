//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/types/adapt_type.hpp>
#include <dynd/func/chain.hpp>

using namespace std;
using namespace dynd;

ndt::adapt_type::adapt_type(const type &operand_type, const type &value_type,
                            const nd::string &op)
    : base_expr_type(
          adapt_type_id, expr_kind, operand_type.get_data_size(),
          operand_type.get_data_alignment(),
          inherited_flags(value_type.get_flags(), operand_type.get_flags()), 0),
      m_value_type(value_type), m_operand_type(operand_type), m_op(op)
{
  if (!value_type.is_builtin() &&
      value_type.extended()->adapt_type(operand_type.value_type(), op,
                                        m_forward, m_reverse)) {
  } else if (!operand_type.value_type().is_builtin() &&
             operand_type.value_type().extended()->reverse_adapt_type(
                 value_type, op, m_forward, m_reverse)) {
  } else {
    stringstream ss;
    ss << "Cannot create type ";
    print_type(ss);
    throw type_error(ss.str());
  }

  // If the operand is an expression, make a buffering arrfunc
  if (m_operand_type.get_kind() == expr_kind && !m_forward.is_null() &&
      m_operand_type != m_forward.get_type()->get_pos_type(0)) {
    m_forward = nd::functional::chain(
        make_arrfunc_from_assignment(m_forward.get_type()->get_pos_type(0),
                                     m_operand_type, assign_error_default),
        m_forward, m_forward.get_type()->get_pos_type(0));
  }
}

ndt::adapt_type::~adapt_type() {}

void ndt::adapt_type::print_data(std::ostream &DYND_UNUSED(o),
                                 const char *DYND_UNUSED(arrmeta),
                                 const char *DYND_UNUSED(data)) const
{
  throw runtime_error(
      "internal error: adapt_type::print_data isn't supposed to be called");
}

void ndt::adapt_type::print_type(std::ostream &o) const
{
  o << "adapt[(" << m_operand_type << ") -> " << m_value_type << ", ";
  print_escaped_utf8_string(o, m_op, true);
  o << "]";
}

bool
ndt::adapt_type::is_lossless_assignment(const type &DYND_UNUSED(dst_tp),
                                        const type &DYND_UNUSED(src_tp)) const
{
  return false;
}

bool ndt::adapt_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != adapt_type_id) {
    return false;
  } else {
    const adapt_type *dt = static_cast<const adapt_type *>(&rhs);
    return m_value_type == dt->m_value_type &&
           m_operand_type == dt->m_operand_type && m_op == dt->m_op;
  }
}

ndt::type
ndt::adapt_type::with_replaced_storage_type(const type &replacement_type) const
{
  if (m_operand_type.get_kind() != expr_kind) {
    // If there's no expression in the operand, just try substituting (the
    // constructor will error-check)
    return type(new adapt_type(replacement_type, m_value_type, m_op), false);
  } else {
    // With an expression operand, replace it farther down the chain
    return type(
        new adapt_type(reinterpret_cast<const base_expr_type *>(
                           replacement_type.extended())
                           ->with_replaced_storage_type(replacement_type),
                       m_value_type, m_op),
        false);
  }
}

size_t ndt::adapt_type::make_operand_to_value_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
    const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx) const
{
  const arrfunc_type_data *af = m_forward.get();
  if (af != NULL) {
    return af->instantiate(af, m_forward.get_type(), 0, NULL, ckb, ckb_offset,
                           m_value_type, dst_arrmeta, -1, &m_operand_type,
                           &src_arrmeta, kernreq, ectx, nd::array(),
                           std::map<nd::string, type>());
  } else {
    stringstream ss;
    ss << "Cannot apply ";
    print_type(ss);
    ss << "in a forward direction";
    throw type_error(ss.str());
  }
}

size_t ndt::adapt_type::make_value_to_operand_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
    const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx) const
{
  const arrfunc_type_data *af = m_reverse.get();
  if (af != NULL) {
    return af->instantiate(af, m_reverse.get_type(), 0, NULL, ckb, ckb_offset,
                           m_operand_type, src_arrmeta, -1, &m_value_type,
                           &dst_arrmeta, kernreq, ectx, nd::array(),
                           std::map<nd::string, type>());
  } else {
    stringstream ss;
    ss << "Cannot apply ";
    print_type(ss);
    ss << "in a reverse direction";
    throw type_error(ss.str());
  }
}
