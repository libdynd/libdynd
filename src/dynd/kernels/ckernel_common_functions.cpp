//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>

using namespace std;
using namespace dynd;

void kernels::destroy_trivial_parent_ckernel(ckernel_prefix *self)
{
  self->get_child_ckernel(sizeof(ckernel_prefix))->destroy();
}

namespace {
struct constant_value_assignment_ck
    : nd::base_kernel<constant_value_assignment_ck, 0> {
  // Pointer to the array inside of `constant`
  const char *m_constant_data;
  // Reference which owns the constant value to assign
  nd::array m_constant;

  ~constant_value_assignment_ck()
  {
    // Destroy the child ckernel
    get_child_ckernel()->destroy();
  }

  void single(char *dst, char *const *DYND_UNUSED(src))
  {
    ckernel_prefix *child = get_child_ckernel();
    expr_single_t child_fn = child->get_function<expr_single_t>();
    child_fn(child, dst, const_cast<char *const *>(&m_constant_data));
  }

  void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
               const intptr_t *DYND_UNUSED(src_stride), size_t count)
  {
    ckernel_prefix *child = get_child_ckernel();
    expr_strided_t child_fn = child->get_function<expr_strided_t>();
    intptr_t zero_stride = 0;
    child_fn(child, dst, dst_stride,
             const_cast<char *const *>(&m_constant_data), &zero_stride, count);
  }
};
} // anonymous namespace

size_t kernels::make_constant_value_assignment_ckernel(
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const nd::array &constant,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
  typedef constant_value_assignment_ck self_type;
  // Initialize the ckernel
  self_type *self = self_type::make(ckb, kernreq, ckb_offset);
  // Store the constant data
  self->m_constant = constant.cast(dst_tp).eval_immutable(ectx);
  self->m_constant_data = self->m_constant.get_readonly_originptr();
  // Create the child assignment ckernel
  return make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta,
                                self->m_constant.get_type(),
                                self->m_constant.get_arrmeta(), kernreq, ectx);
}
