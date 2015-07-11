//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/type.hpp>
#include <dynd/kernels/expression_assignment_kernels.hpp>
#include <dynd/kernels/expression_comparison_kernels.hpp>

using namespace std;
using namespace dynd;

ndt::base_expr_type::~base_expr_type() {}

bool ndt::base_expr_type::is_expression() const { return true; }

ndt::type ndt::base_expr_type::get_canonical_type() const
{
  return get_value_type();
}

void ndt::base_expr_type::arrmeta_default_construct(char *arrmeta,
                                                    bool blockref_alloc) const
{
  const type &dt = get_operand_type();
  if (!dt.is_builtin()) {
    dt.extended()->arrmeta_default_construct(arrmeta, blockref_alloc);
  }
}

void ndt::base_expr_type::arrmeta_copy_construct(
    char *dst_arrmeta, const char *src_arrmeta,
    memory_block_data *embedded_reference) const
{
  const type &dt = get_operand_type();
  if (!dt.is_builtin()) {
    dt.extended()->arrmeta_copy_construct(dst_arrmeta, src_arrmeta,
                                          embedded_reference);
  }
}

void ndt::base_expr_type::arrmeta_destruct(char *arrmeta) const
{
  const type &dt = get_operand_type();
  if (!dt.is_builtin()) {
    dt.extended()->arrmeta_destruct(arrmeta);
  }
}

void ndt::base_expr_type::arrmeta_debug_print(const char *arrmeta,
                                              std::ostream &o,
                                              const std::string &indent) const
{
  const type &dt = get_operand_type();
  if (!dt.is_builtin()) {
    dt.extended()->arrmeta_debug_print(arrmeta, o, indent);
  }
}

size_t ndt::base_expr_type::get_iterdata_size(intptr_t DYND_UNUSED(ndim)) const
{
  return 0;
}

size_t ndt::base_expr_type::make_operand_to_value_assignment_kernel(
    void *DYND_UNUSED(ckb), intptr_t DYND_UNUSED(ckb_offset),
    const char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    kernel_request_t DYND_UNUSED(kernreq),
    const eval::eval_context *DYND_UNUSED(ectx)) const
{
  stringstream ss;
  ss << "dynd type " << type(this, true)
     << " does not support reading of its values";
  throw dynd::type_error(ss.str());
}

size_t ndt::base_expr_type::make_value_to_operand_assignment_kernel(
    void *DYND_UNUSED(ckb), intptr_t DYND_UNUSED(ckb_offset),
    const char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    kernel_request_t DYND_UNUSED(kernreq),
    const eval::eval_context *DYND_UNUSED(ectx)) const
{
  stringstream ss;
  ss << "dynd type " << type(this, true)
     << " does not support writing to its values";
  throw dynd::type_error(ss.str());
}

intptr_t ndt::base_expr_type::make_assignment_kernel(
    const arrfunc_type *DYND_UNUSED(af_tp), void *ckb, intptr_t ckb_offset,
    const type &dst_tp, const char *dst_arrmeta, const type &src_tp,
    const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &DYND_UNUSED(kwds)) const
{
  return make_expression_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta,
                                           src_tp, src_arrmeta, kernreq, ectx);
}

size_t ndt::base_expr_type::make_comparison_kernel(
    void *ckb, intptr_t ckb_offset, const type &src0_dt,
    const char *src0_arrmeta, const type &src1_dt, const char *src1_arrmeta,
    comparison_type_t comptype, const eval::eval_context *ectx) const
{
  return make_expression_comparison_kernel(ckb, ckb_offset, src0_dt,
                                           src0_arrmeta, src1_dt, src1_arrmeta,
                                           comptype, ectx);
}