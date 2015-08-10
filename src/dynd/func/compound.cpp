//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/compound_kernel.hpp>
#include <dynd/func/compound.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::functional::left_compound(const callable &child)
{
  return callable::make<left_compound_kernel>(
      ndt::callable_type::make(child.get_type()->get_return_type(),
                               child.get_type()->get_pos_types()(irange() < 1)),
      child, 0);
}

nd::callable nd::functional::right_compound(const callable &child)
{
  return callable::make<right_compound_kernel>(
      ndt::callable_type::make(
          child.get_type()->get_return_type(),
          child.get_type()->get_pos_types()(1 >= irange())),
      child, 0);
}

intptr_t nd::functional::wrap_binary_as_unary_reduction_ckernel(
    void *ckb, intptr_t ckb_offset, bool right_associative,
    kernel_request_t kernreq)
{
  // Add an adapter kernel which converts the binary expr kernel to an expr
  // kernel
  ckernel_prefix *ckp =
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
          ->alloc_ck<ckernel_prefix>(ckb_offset);
  ckp->destructor = &kernels::destroy_trivial_parent_ckernel;
  if (right_associative) {
    ckp->set_expr_function(kernreq, &left_compound_kernel::single_wrapper,
                           &left_compound_kernel::strided_wrapper);
  } else {
    ckp->set_expr_function(kernreq, &right_compound_kernel::single_wrapper,
                           &left_compound_kernel::strided_wrapper);
  }
  return ckb_offset;
}
