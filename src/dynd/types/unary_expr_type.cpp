//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/unary_expr_type.hpp>
#include <dynd/shortvector.hpp>
#include <dynd/types/builtin_type_properties.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/func/callable.hpp>

using namespace std;
using namespace dynd;

ndt::unary_expr_type::unary_expr_type(const type &value_type, const type &operand_type,
                                      const expr_kernel_generator *kgen)
    : base_expr_type(unary_expr_type_id, expr_kind, operand_type.get_data_size(), operand_type.get_data_alignment(),
                     inherited_flags(value_type.get_flags(), operand_type.get_flags()), operand_type.get_arrmeta_size(),
                     value_type.get_ndim()),
      m_value_type(value_type), m_operand_type(operand_type), m_kgen(kgen)
{
}

ndt::unary_expr_type::~unary_expr_type() { expr_kernel_generator_decref(m_kgen); }

void ndt::unary_expr_type::print_data(std::ostream &DYND_UNUSED(o), const char *DYND_UNUSED(arrmeta),
                                      const char *DYND_UNUSED(data)) const
{
  throw runtime_error("internal error: unary_expr_type::print_data isn't "
                      "supposed to be called");
}

void ndt::unary_expr_type::print_type(std::ostream &o) const
{
  o << "expr<";
  o << m_value_type;
  o << ", op0=" << m_operand_type;
  o << ", expr=";
  m_kgen->print_type(o);
  o << ">";
}

ndt::type ndt::unary_expr_type::apply_linear_index(intptr_t nindices, const irange *DYND_UNUSED(indices),
                                                   size_t current_i, const type &DYND_UNUSED(root_tp),
                                                   bool DYND_UNUSED(leading_dimension)) const
{
  if (m_kgen->is_elwise()) {
    // Scalar behavior
    if (nindices == 0) {
      return type(this, true);
    }
    else {
      throw too_many_indices(type(this, true), current_i + nindices, current_i);
    }
  }
  else {
    throw runtime_error("unary_expr_type::apply_linear_index is only "
                        "implemented for elwise kernel generators");
  }
}

intptr_t ndt::unary_expr_type::apply_linear_index(intptr_t nindices, const irange *DYND_UNUSED(indices),
                                                  const char *arrmeta, const type &DYND_UNUSED(result_tp),
                                                  char *out_arrmeta,
                                                  const intrusive_ptr<memory_block_data> &embedded_reference,
                                                  size_t current_i, const type &DYND_UNUSED(root_tp),
                                                  bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
                                                  intrusive_ptr<memory_block_data> &DYND_UNUSED(inout_dataref)) const
{
  if (m_kgen->is_elwise()) {
    // Scalar behavior
    if (nindices == 0) {
      // Copy any arrmeta verbatim
      if (get_arrmeta_size() > 0) {
        m_operand_type.extended()->arrmeta_copy_construct(out_arrmeta, arrmeta, embedded_reference);
      }
      return 0;
    }
    else {
      throw too_many_indices(type(this, true), current_i + nindices, current_i);
    }
  }
  else {
    throw runtime_error("unary_expr_type::apply_linear_index is only "
                        "implemented for elwise kernel generators");
  }
}

bool ndt::unary_expr_type::is_lossless_assignment(const type &DYND_UNUSED(dst_tp),
                                                  const type &DYND_UNUSED(src_tp)) const
{
  return false;
}

bool ndt::unary_expr_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_type_id() != unary_expr_type_id) {
    return false;
  }
  else {
    const unary_expr_type *dt = static_cast<const unary_expr_type *>(&rhs);
    return m_value_type == dt->m_value_type && m_operand_type == dt->m_operand_type && m_kgen == dt->m_kgen;
  }
}

size_t ndt::unary_expr_type::make_operand_to_value_assignment_kernel(void *ckb, intptr_t ckb_offset,
                                                                     const char *dst_arrmeta, const char *src_arrmeta,
                                                                     kernel_request_t kernreq,
                                                                     const eval::eval_context *ectx) const
{
  // As a special case, when src_count == 1, the kernel generated
  // is a expr_single_t/expr_strided_t instead of
  // expr_single_t/expr_strided_t
  return m_kgen->make_expr_kernel(ckb, ckb_offset, m_value_type, dst_arrmeta, 1, &m_operand_type.value_type(),
                                  &src_arrmeta, kernreq, ectx);
}

size_t ndt::unary_expr_type::make_value_to_operand_assignment_kernel(void *DYND_UNUSED(ckb),
                                                                     intptr_t DYND_UNUSED(ckb_offset),
                                                                     const char *DYND_UNUSED(dst_arrmeta),
                                                                     const char *DYND_UNUSED(src_arrmeta),
                                                                     kernel_request_t DYND_UNUSED(kernreq),
                                                                     const eval::eval_context *DYND_UNUSED(ectx)) const
{
  throw runtime_error("Cannot assign to a dynd unary_expr object value");
}

ndt::type ndt::unary_expr_type::with_replaced_storage_type(const type &DYND_UNUSED(replacement_type)) const
{
  throw runtime_error("TODO: implement unary_expr_type::with_replaced_storage_type");
}

void ndt::unary_expr_type::get_dynamic_array_properties(const std::pair<std::string, nd::callable> **out_properties,
                                                        size_t *out_count) const
{
  const type &udt = m_value_type.get_dtype();
  if (!udt.is_builtin()) {
    udt.extended()->get_dynamic_array_properties(out_properties, out_count);
  }
  else {
    get_builtin_type_dynamic_array_properties(udt.get_type_id(), out_properties, out_count);
  }
}

void ndt::unary_expr_type::get_dynamic_array_functions(std::map<std::string, nd::callable> &functions) const
{
  const type &udt = m_value_type.get_dtype();
  if (!udt.is_builtin()) {
    udt.extended()->get_dynamic_array_functions(functions);
  }
  else {
    // get_builtin_type_dynamic_array_functions(udt.get_type_id(),
    // out_functions, out_count);
  }
}
