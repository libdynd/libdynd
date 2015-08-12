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

using namespace std;
using namespace dynd;

nd::callable nd::functional::reduction(const callable &child,
                                       const ndt::type &src0_tp, bool keepdims,
                                       const vector<intptr_t> &axes,
                                       const array &reduction_identity,
                                       callable_property properties)
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
      return reduction(left_compound(child), src0_tp, keepdims, axes,
                       reduction_identity, properties);
    }

    return reduction(right_compound(child), src0_tp, keepdims, axes,
                     reduction_identity, properties);
  }

  reduction_kernel::static_data_type self(child, axes, properties);

  if (!reduction_identity.is_null()) {
    if (reduction_identity.is_immutable() &&
        reduction_identity.get_type() ==
            elwise_reduction_tp->get_return_type()) {
      self.reduction_identity = reduction_identity;
    } else {
      self.reduction_identity = empty(elwise_reduction_tp->get_return_type());
      self.reduction_identity.vals() = reduction_identity;
      self.reduction_identity.flag_as_immutable();
    }
  }

  intptr_t reduction_ndim =
      src0_tp.get_ndim() - child.get_type()->get_return_type().get_ndim();

  // Figure out the result type
  ndt::type dst_tp = child.get_type()->get_return_type();
  for (intptr_t i = reduction_ndim - 1; i >= 0; --i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
      if (keepdims) {
        dst_tp = ndt::make_fixed_dim(1, dst_tp);
      }
    } else {
      ndt::type subtype = src0_tp.get_type_at_dimension(NULL, i);
      dst_tp =
          subtype.extended<ndt::base_dim_type>()->with_element_type(dst_tp);
    }
  }
  return callable::make<reduction_kernel>(
      ndt::callable_type::make(
          dst_tp, {src0_tp}, {"keepdims"},
          {ndt::option_type::make(ndt::type::make<bool>())}),
      self, sizeof(reduction_kernel::data_type));
}
