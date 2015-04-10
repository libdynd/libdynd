//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/lift_reduction_arrfunc.hpp>
#include <dynd/kernels/make_lifted_reduction_ckernel.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

namespace {

struct lifted_reduction_arrfunc_data {
  // Pointer to the child arrfunc
  nd::arrfunc child_elwise_reduction;
  nd::arrfunc child_dst_initialization;
  nd::array reduction_identity;
  // The types of the child ckernel and this one
  const ndt::type *child_data_types;
  ndt::type data_types[2];
  intptr_t reduction_ndim;
  bool associative, commutative, right_associative;
  shortvector<bool> reduction_dimflags;
};

static void delete_lifted_reduction_arrfunc_data(arrfunc_type_data *self_af)
{
  lifted_reduction_arrfunc_data *self =
      *self_af->get_data_as<lifted_reduction_arrfunc_data *>();
  delete self;
}

static intptr_t instantiate_lifted_reduction_arrfunc_data(
    const arrfunc_type_data *af_self, const arrfunc_type *DYND_UNUSED(af_tp),
    char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta,
    intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &DYND_UNUSED(kwds),
    const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  lifted_reduction_arrfunc_data *data =
      *af_self->get_data_as<lifted_reduction_arrfunc_data *>();
  return make_lifted_reduction_ckernel(
      data->child_elwise_reduction.get(),
      data->child_elwise_reduction.get_type(),
      data->child_dst_initialization.get(),
      data->child_dst_initialization.get_type(), ckb, ckb_offset, dst_tp,
      dst_arrmeta, src_tp[0], src_arrmeta[0], data->reduction_ndim,
      data->reduction_dimflags.get(), data->associative, data->commutative,
      data->right_associative, data->reduction_identity,
      static_cast<dynd::kernel_request_t>(kernreq), ectx);
}

} // anonymous namespace

nd::arrfunc dynd::lift_reduction_arrfunc(
    const nd::arrfunc &elwise_reduction_arr, const ndt::type &lifted_arr_type,
    const nd::arrfunc &dst_initialization_arr, bool keepdims,
    intptr_t reduction_ndim, const bool *reduction_dimflags, bool associative,
    bool commutative, bool right_associative,
    const nd::array &reduction_identity)
{
  // Validate the input elwise_reduction arrfunc
  if (elwise_reduction_arr.is_null()) {
    throw runtime_error(
        "lift_reduction_arrfunc: 'elwise_reduction' may not be empty");
  }
  const arrfunc_type *elwise_reduction_tp = elwise_reduction_arr.get_type();
  if (elwise_reduction_tp->get_npos() != 1 &&
      !(elwise_reduction_tp->get_npos() == 2 &&
        elwise_reduction_tp->get_pos_type(0) ==
            elwise_reduction_tp->get_pos_type(1) &&
        elwise_reduction_tp->get_pos_type(0) ==
            elwise_reduction_tp->get_return_type())) {
    stringstream ss;
    ss << "lift_reduction_arrfunc: 'elwise_reduction' must contain a"
          " unary operation ckernel or a binary expr ckernel with all "
          "equal types, its prototype is " << elwise_reduction_tp;
    throw invalid_argument(ss.str());
  }

  // Figure out the result type
  ndt::type lifted_dst_type = elwise_reduction_tp->get_return_type();
  for (intptr_t i = reduction_ndim - 1; i >= 0; --i) {
    if (reduction_dimflags[i]) {
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
              subtype.extended<fixed_dim_type>()->get_fixed_dim_size(),
              lifted_dst_type);
        }
        break;
      case cfixed_dim_type_id:
        lifted_dst_type = ndt::make_fixed_dim(
            subtype.extended<cfixed_dim_type>()->get_fixed_dim_size(),
            lifted_dst_type);
        break;
      case var_dim_type_id:
        lifted_dst_type = ndt::make_var_dim(lifted_dst_type);
        break;
      default: {
        stringstream ss;
        ss << "lift_reduction_arrfunc: don't know how to process ";
        ss << "dimension of type " << subtype;
        throw type_error(ss.str());
      }
      }
    }
  }

  lifted_reduction_arrfunc_data *self = new lifted_reduction_arrfunc_data;
  self->child_elwise_reduction = elwise_reduction_arr;
  self->child_dst_initialization = dst_initialization_arr;
  if (!reduction_identity.is_null()) {
    if (reduction_identity.is_immutable() &&
        reduction_identity.get_type() ==
            elwise_reduction_tp->get_return_type()) {
      self->reduction_identity = reduction_identity;
    } else {
      self->reduction_identity =
          nd::empty(elwise_reduction_tp->get_return_type());
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
  self->reduction_dimflags.init(reduction_ndim);
  memcpy(self->reduction_dimflags.get(), reduction_dimflags,
         sizeof(bool) * reduction_ndim);

  return nd::arrfunc(
      ndt::make_arrfunc(ndt::make_tuple(lifted_arr_type), lifted_dst_type),
      self, 0, &instantiate_lifted_reduction_arrfunc_data, NULL, NULL,
      &delete_lifted_reduction_arrfunc_data);
}
