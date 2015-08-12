//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/compound.hpp>
#include <dynd/func/reduction.hpp>
#include <dynd/kernels/make_lifted_reduction_ckernel.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::functional::reduction(const callable &child)
{
  // Validate the input elwise_reduction callable
  if (child.is_null()) {
    throw runtime_error(
        "lift_reduction_callable: 'elwise_reduction' may not be empty");
  }
  const ndt::callable_type *elwise_reduction_tp = child.get_type();
  if (elwise_reduction_tp->get_npos() != 1 &&
      !(elwise_reduction_tp->get_npos() == 2 &&
        elwise_reduction_tp->get_pos_type(0) ==
            elwise_reduction_tp->get_pos_type(1) &&
        elwise_reduction_tp->get_pos_type(0) ==
            elwise_reduction_tp->get_return_type())) {
    stringstream ss;
    ss << "lift_reduction_callable: 'elwise_reduction' must contain a"
          " unary operation ckernel or a binary expr ckernel with all "
          "equal types, its prototype is " << elwise_reduction_tp;
    throw invalid_argument(ss.str());
  }

  if (elwise_reduction_tp->get_npos() == 2) {
    if (right_associative) {
      return reduction(left_compound(child));
    }

    return reduction(right_compound(child));
  }

  return callable::make<reduction_kernel>(
      ndt::callable_type::make(
          ndt::make_ellipsis_dim("Dims", child.get_type()->get_return_type()),
          {ndt::make_ellipsis_dim("Dims", child.get_type()->get_pos_type(0))},
          {"axes", "identity", "keepdims"},
          {ndt::type("Fixed * int32"),
           ndt::option_type::make(child.get_type()->get_return_type()),
           ndt::option_type::make(ndt::type::make<bool>())}),
      reduction_kernel::static_data_type(child),
      sizeof(reduction_kernel::data_type));
}
