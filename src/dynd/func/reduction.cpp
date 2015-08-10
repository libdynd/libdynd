//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/compound.hpp>
#include <dynd/func/reduction.hpp>
#include <dynd/kernels/make_lifted_reduction_ckernel.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::functional::reduction(
    const callable &elwise_reduction_arr, const ndt::type &lifted_arr_type,
    const callable &dst_initialization_arr, bool keepdims,
    intptr_t reduction_ndim, const vector<int> &axes, bool associative,
    bool commutative, bool right_associative, const array &reduction_identity)
{
  // Validate the input elwise_reduction callable
  if (elwise_reduction_arr.is_null()) {
    throw runtime_error(
        "lift_reduction_callable: 'elwise_reduction' may not be empty");
  }
  const ndt::callable_type *elwise_reduction_tp =
      elwise_reduction_arr.get_type();
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
      return reduction(left_compound(elwise_reduction_arr), lifted_arr_type,
                       dst_initialization_arr, keepdims, reduction_ndim, axes,
                       associative, commutative, right_associative,
                       reduction_identity);
    }

    return reduction(right_compound(elwise_reduction_arr), lifted_arr_type,
                     dst_initialization_arr, keepdims, reduction_ndim, axes,
                     associative, commutative, right_associative,
                     reduction_identity);
  }

  // Figure out the result type
  ndt::type lifted_dst_type = elwise_reduction_tp->get_return_type();
  for (intptr_t i = reduction_ndim - 1; i >= 0; --i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
      if (keepdims) {
        lifted_dst_type = ndt::make_fixed_dim(1, lifted_dst_type);
      }
    } else {
      ndt::type subtype = lifted_arr_type.get_type_at_dimension(NULL, i);
      switch (subtype.get_type_id()) {
      case fixed_dim_type_id:
        if (subtype.get_kind() == kind_kind) {
          lifted_dst_type = ndt::make_fixed_dim_kind(lifted_dst_type);
        } else {
          lifted_dst_type = ndt::make_fixed_dim(
              subtype.extended<ndt::fixed_dim_type>()->get_fixed_dim_size(),
              lifted_dst_type);
        }
        break;
      case var_dim_type_id:
        lifted_dst_type = ndt::var_dim_type::make(lifted_dst_type);
        break;
      default: {
        stringstream ss;
        ss << "lift_reduction_callable: don't know how to process ";
        ss << "dimension of type " << subtype;
        throw type_error(ss.str());
      }
      }
    }
  }

  std::shared_ptr<reduction_kernel::stored_data_type> self =
      make_shared<reduction_kernel::stored_data_type>();
  self->child_elwise_reduction = elwise_reduction_arr;
  self->child_dst_initialization = dst_initialization_arr;
  if (!reduction_identity.is_null()) {
    if (reduction_identity.is_immutable() &&
        reduction_identity.get_type() ==
            elwise_reduction_tp->get_return_type()) {
      self->reduction_identity = reduction_identity;
    } else {
      self->reduction_identity = empty(elwise_reduction_tp->get_return_type());
      self->reduction_identity.vals() = reduction_identity;
      self->reduction_identity.flag_as_immutable();
    }
  }
  self->data_types[0] = lifted_dst_type;
  self->data_types[1] = lifted_arr_type;
  self->reduction_ndim = reduction_ndim;
  self->associative = associative;
  self->commutative = commutative;
  self->right_associative = right_associative;
  self->reduction_dimflags = axes;

  return callable::make<reduction_kernel>(
      ndt::callable_type::make(lifted_dst_type, lifted_arr_type), self, 0);
}
