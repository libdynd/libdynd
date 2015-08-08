//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/reduction.hpp>
#include <dynd/kernels/make_lifted_reduction_ckernel.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/kernels/make_lifted_reduction_ckernel.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/func/assignment.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/base_kernel.hpp>

using namespace std;
using namespace dynd;

namespace {

struct lifted_reduction_callable_data {
  // Pointer to the child callable
  nd::callable child_elwise_reduction;
  nd::callable child_dst_initialization;
  nd::array reduction_identity;
  // The types of the child ckernel and this one
  const ndt::type *child_data_types;
  ndt::type data_types[2];
  intptr_t reduction_ndim;
  bool associative, commutative, right_associative;
  shortvector<bool> reduction_dimflags;
};

static intptr_t instantiate_lifted_reduction_callable_data(
    char *static_data, size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &DYND_UNUSED(kwds),
    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  std::shared_ptr<lifted_reduction_callable_data> data =
      *reinterpret_cast<std::shared_ptr<lifted_reduction_callable_data> *>(
           static_data);
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

nd::callable nd::functional::reduction(
    const callable &elwise_reduction_arr, const ndt::type &lifted_arr_type,
    const callable &dst_initialization_arr, bool keepdims,
    intptr_t reduction_ndim, const bool *reduction_dimflags, bool associative,
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

  std::shared_ptr<lifted_reduction_callable_data> self =
      make_shared<lifted_reduction_callable_data>();
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
  self->reduction_dimflags.init(reduction_ndim);
  memcpy(self->reduction_dimflags.get(), reduction_dimflags,
         sizeof(bool) * reduction_ndim);

  return callable(ndt::callable_type::make(lifted_dst_type, lifted_arr_type),
                  self, 0, NULL, NULL,
                  &instantiate_lifted_reduction_callable_data);
}

/**
 * Adds a ckernel layer for processing one dimension of the reduction.
 * This is for a strided dimension which is being reduced, and is not
 * the final dimension before the accumulation operation.
 */
static size_t make_strided_initial_reduction_dimension_kernel(
    void *ckb, intptr_t ckb_offset, intptr_t src_stride, intptr_t src_size,
    kernel_request_t kernreq)
{
  nd::strided_initial_reduction_kernel_extra *e =
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
          ->alloc_ck<nd::strided_initial_reduction_kernel_extra>(ckb_offset);
  e->destructor = &nd::strided_initial_reduction_kernel_extra::destruct;
  // Get the function pointer for the first_call
  if (kernreq == kernel_request_single) {
    e->set_first_call_function(
        &nd::strided_initial_reduction_kernel_extra::single_first);
  } else if (kernreq == kernel_request_strided) {
    e->set_first_call_function(
        &nd::strided_initial_reduction_kernel_extra::strided_first);
  } else {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: unrecognized request "
       << (int)kernreq;
    throw runtime_error(ss.str());
  }
  // The function pointer for followup accumulation calls
  e->set_followup_call_function(
      &nd::strided_initial_reduction_kernel_extra::strided_followup);
  // The striding parameters
  e->src_stride = src_stride;
  e->size = src_size;
  return ckb_offset;
}

/**
 * Adds a ckernel layer for processing one dimension of the reduction.
 * This is for a strided dimension which is being broadcast, and is not
 * the final dimension before the accumulation operation.
 */
static size_t make_strided_initial_broadcast_dimension_kernel(
    void *ckb, intptr_t ckb_offset, intptr_t dst_stride, intptr_t src_stride,
    intptr_t src_size, kernel_request_t kernreq)
{
  nd::strided_initial_broadcast_kernel_extra *e =
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
          ->alloc_ck<nd::strided_initial_broadcast_kernel_extra>(ckb_offset);
  e->destructor = &nd::strided_initial_broadcast_kernel_extra::destruct;
  // Get the function pointer for the first_call
  if (kernreq == kernel_request_single) {
    e->set_first_call_function(
        &nd::strided_initial_broadcast_kernel_extra::single_first);
  } else if (kernreq == kernel_request_strided) {
    e->set_first_call_function(
        &nd::strided_initial_broadcast_kernel_extra::strided_first);
  } else {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: unrecognized request "
       << (int)kernreq;
    throw runtime_error(ss.str());
  }
  // The function pointer for followup accumulation calls
  e->set_followup_call_function(
      &nd::strided_initial_broadcast_kernel_extra::strided_followup);
  // The striding parameters
  e->dst_stride = dst_stride;
  e->src_stride = src_stride;
  e->size = src_size;
  return ckb_offset;
}

static void
check_dst_initialization(const ndt::callable_type *dst_initialization_tp,
                         const ndt::type &dst_tp, const ndt::type &src_tp)
{
  if (dst_initialization_tp->get_return_type() != dst_tp) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: dst initialization ckernel ";
    ss << "dst type is " << dst_initialization_tp->get_return_type();
    ss << ", expected " << dst_tp;
    throw type_error(ss.str());
  }
  if (dst_initialization_tp->get_pos_type(0) != src_tp) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: dst initialization ckernel ";
    ss << "src type is " << dst_initialization_tp->get_return_type();
    ss << ", expected " << src_tp;
    throw type_error(ss.str());
  }
}

/**
 * Adds a ckernel layer for processing one dimension of the reduction.
 * This is for a strided dimension which is being reduced, and is
 * the final dimension before the accumulation operation.
 *
 * If dst_initialization is NULL, an assignment kernel is used.
 */
static size_t make_strided_inner_reduction_dimension_kernel(
    const callable_type_data *elwise_reduction_const,
    const ndt::callable_type *elwise_reduction_tp,
    const callable_type_data *dst_initialization_const,
    const ndt::callable_type *dst_initialization_tp, void *ckb,
    intptr_t ckb_offset, intptr_t src_stride, intptr_t src_size,
    const ndt::type &dst_tp, const char *dst_arrmeta, const ndt::type &src_tp,
    const char *src_arrmeta, bool right_associative,
    const nd::array &reduction_identity, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  callable_type_data *elwise_reduction =
      const_cast<callable_type_data *>(elwise_reduction_const);
  callable_type_data *dst_initialization =
      const_cast<callable_type_data *>(dst_initialization_const);

  intptr_t root_ckb_offset = ckb_offset;
  nd::strided_inner_reduction_kernel_extra *e =
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
          ->alloc_ck<nd::strided_inner_reduction_kernel_extra>(ckb_offset);
  e->destructor = &nd::strided_inner_reduction_kernel_extra::destruct;
  // Cannot have both a dst_initialization kernel and a reduction identity
  if (dst_initialization != NULL && !reduction_identity.is_null()) {
    throw invalid_argument(
        "make_lifted_reduction_ckernel: cannot specify"
        " both a dst_initialization kernel and a reduction_identity");
  }
  if (reduction_identity.is_null()) {
    // Get the function pointer for the first_call, for the case with
    // no reduction identity
    if (kernreq == kernel_request_single) {
      e->set_first_call_function(
          &nd::strided_inner_reduction_kernel_extra::single_first);
    } else if (kernreq == kernel_request_strided) {
      e->set_first_call_function(
          &nd::strided_inner_reduction_kernel_extra::strided_first);
    } else {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: unrecognized request "
         << (int)kernreq;
      throw runtime_error(ss.str());
    }
  } else {
    // Get the function pointer for the first_call, for the case with
    // a reduction identity
    if (kernreq == kernel_request_single) {
      e->set_first_call_function(
          &nd::strided_inner_reduction_kernel_extra::single_first_with_ident);
    } else if (kernreq == kernel_request_strided) {
      e->set_first_call_function(
          &nd::strided_inner_reduction_kernel_extra::strided_first_with_ident);
    } else {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: unrecognized request "
         << (int)kernreq;
      throw runtime_error(ss.str());
    }
    if (reduction_identity.get_type() != dst_tp) {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: reduction identity type ";
      ss << reduction_identity.get_type() << " does not match dst type ";
      ss << dst_tp;
      throw runtime_error(ss.str());
    }
    e->ident_data = reduction_identity.get_readonly_originptr();
    e->ident_ref = reduction_identity.get_memblock().release();
  }
  // The function pointer for followup accumulation calls
  e->set_followup_call_function(
      &nd::strided_inner_reduction_kernel_extra::strided_followup);
  // The striding parameters
  e->src_stride = src_stride;
  e->size = src_size;
  // Validate that the provided callables are unary operations,
  // and have the correct types
  if (elwise_reduction_tp->get_npos() != 1 &&
      elwise_reduction_tp->get_npos() != 2) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
    ss << "funcproto must be unary or a binary expr with all equal types";
    throw runtime_error(ss.str());
  }
  if (elwise_reduction_tp->get_return_type() != dst_tp) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
    ss << "dst type is " << elwise_reduction_tp->get_return_type();
    ss << ", expected " << dst_tp;
    throw type_error(ss.str());
  }
  if (elwise_reduction_tp->get_pos_type(0) != src_tp) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
    ss << "src type is " << elwise_reduction_tp->get_return_type();
    ss << ", expected " << src_tp;
    throw type_error(ss.str());
  }
  if (dst_initialization != NULL) {
    check_dst_initialization(dst_initialization_tp, dst_tp, src_tp);
  }
  if (elwise_reduction_tp->get_npos() == 2) {
    ckb_offset = kernels::wrap_binary_as_unary_reduction_ckernel(
        ckb, ckb_offset, right_associative, kernel_request_strided);
    ndt::type src_tp_doubled[2] = {src_tp, src_tp};
    const char *src_arrmeta_doubled[2] = {src_arrmeta, src_arrmeta};
    ckb_offset = elwise_reduction->instantiate(
        elwise_reduction->static_data, 0, NULL, ckb, ckb_offset, dst_tp,
        dst_arrmeta, elwise_reduction_tp->get_npos(), src_tp_doubled,
        src_arrmeta_doubled, kernel_request_strided, ectx, nd::array(),
        std::map<std::string, ndt::type>());
  } else {
    ckb_offset = elwise_reduction->instantiate(
        elwise_reduction->static_data, 0, NULL, ckb, ckb_offset, dst_tp,
        dst_arrmeta, elwise_reduction_tp->get_npos(), &src_tp, &src_arrmeta,
        kernel_request_strided, ectx, nd::array(),
        std::map<std::string, ndt::type>());
  }
  // Make sure there's capacity for the next ckernel
  reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
      ->reserve(ckb_offset + sizeof(ckernel_prefix));
  // Need to retrieve 'e' again because it may have moved
  e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
          ->get_at<nd::strided_inner_reduction_kernel_extra>(root_ckb_offset);
  e->dst_init_kernel_offset = ckb_offset - root_ckb_offset;
  if (dst_initialization != NULL) {
    ckb_offset = dst_initialization->instantiate(
        dst_initialization->static_data, 0, NULL, ckb, ckb_offset, dst_tp,
        dst_arrmeta, elwise_reduction_tp->get_npos(), &src_tp, &src_arrmeta,
        kernel_request_single, ectx, nd::array(),
        std::map<std::string, ndt::type>());
  } else if (reduction_identity.is_null()) {
    ckb_offset =
        make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
                               src_arrmeta, kernel_request_single, ectx);
  } else {
    ckb_offset = make_assignment_kernel(
        ckb, ckb_offset, dst_tp, dst_arrmeta, reduction_identity.get_type(),
        reduction_identity.get_arrmeta(), kernel_request_single, ectx);
  }

  return ckb_offset;
}

/**
 * Adds a ckernel layer for processing one dimension of the reduction.
 * This is for a strided dimension which is being broadcast, and is
 * the final dimension before the accumulation operation.
 */
static size_t make_strided_inner_broadcast_dimension_kernel(
    const callable_type_data *elwise_reduction_const,
    const ndt::callable_type *elwise_reduction_tp,
    const callable_type_data *dst_initialization_const,
    const ndt::callable_type *dst_initialization_tp, void *ckb,
    intptr_t ckb_offset, intptr_t dst_stride, intptr_t src_stride,
    intptr_t src_size, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type &src_tp, const char *src_arrmeta, bool right_associative,
    const nd::array &reduction_identity, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  callable_type_data *elwise_reduction =
      const_cast<callable_type_data *>(elwise_reduction_const);
  callable_type_data *dst_initialization =
      const_cast<callable_type_data *>(dst_initialization_const);

  intptr_t root_ckb_offset = ckb_offset;
  nd::strided_inner_broadcast_kernel_extra *e =
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
          ->alloc_ck<nd::strided_inner_broadcast_kernel_extra>(ckb_offset);
  e->destructor = &nd::strided_inner_broadcast_kernel_extra::destruct;
  // Cannot have both a dst_initialization kernel and a reduction identity
  if (dst_initialization != NULL && !reduction_identity.is_null()) {
    throw invalid_argument(
        "make_lifted_reduction_ckernel: cannot specify"
        " both a dst_initialization kernel and a reduction_identity");
  }
  if (reduction_identity.is_null()) {
    // Get the function pointer for the first_call, for the case with
    // no reduction identity
    if (kernreq == kernel_request_single) {
      e->set_first_call_function(
          &nd::strided_inner_broadcast_kernel_extra::single_first);
    } else if (kernreq == kernel_request_strided) {
      e->set_first_call_function(
          &nd::strided_inner_broadcast_kernel_extra::strided_first);
    } else {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: unrecognized request "
         << (int)kernreq;
      throw runtime_error(ss.str());
    }
  } else {
    // Get the function pointer for the first_call, for the case with
    // a reduction identity
    if (kernreq == kernel_request_single) {
      e->set_first_call_function(
          &nd::strided_inner_broadcast_kernel_extra::single_first_with_ident);
    } else if (kernreq == kernel_request_strided) {
      e->set_first_call_function(
          &nd::strided_inner_broadcast_kernel_extra::strided_first_with_ident);
    } else {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: unrecognized request "
         << (int)kernreq;
      throw runtime_error(ss.str());
    }
    if (reduction_identity.get_type() != dst_tp) {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: reduction identity type ";
      ss << reduction_identity.get_type() << " does not match dst type ";
      ss << dst_tp;
      throw runtime_error(ss.str());
    }
    e->ident_data = reduction_identity.get_readonly_originptr();
    e->ident_ref = reduction_identity.get_memblock().release();
  }
  // The function pointer for followup accumulation calls
  e->set_followup_call_function(
      &nd::strided_inner_broadcast_kernel_extra::strided_followup);
  // The striding parameters
  e->dst_stride = dst_stride;
  e->src_stride = src_stride;
  e->size = src_size;
  // Validate that the provided callables are unary operations,
  // and have the correct types
  if (elwise_reduction_tp->get_npos() != 1 &&
      elwise_reduction_tp->get_npos() != 2) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
    ss << "funcproto must be unary or a binary expr with all equal types";
    throw runtime_error(ss.str());
  }
  if (elwise_reduction_tp->get_return_type() != dst_tp) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
    ss << "dst type is " << elwise_reduction_tp->get_return_type();
    ss << ", expected " << dst_tp;
    throw type_error(ss.str());
  }
  if (elwise_reduction_tp->get_pos_type(0) != src_tp) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
    ss << "src type is " << elwise_reduction_tp->get_return_type();
    ss << ", expected " << src_tp;
    throw type_error(ss.str());
  }
  if (dst_initialization != NULL) {
    check_dst_initialization(dst_initialization_tp, dst_tp, src_tp);
  }
  if (elwise_reduction_tp->get_npos() == 2) {
    ckb_offset = kernels::wrap_binary_as_unary_reduction_ckernel(
        ckb, ckb_offset, right_associative, kernel_request_strided);
    ndt::type src_tp_doubled[2] = {src_tp, src_tp};
    const char *src_arrmeta_doubled[2] = {src_arrmeta, src_arrmeta};
    ckb_offset = elwise_reduction->instantiate(
        elwise_reduction->static_data, 0, NULL, ckb, ckb_offset, dst_tp,
        dst_arrmeta, elwise_reduction_tp->get_npos(), src_tp_doubled,
        src_arrmeta_doubled, kernel_request_strided, ectx, nd::array(),
        std::map<std::string, ndt::type>());
  } else {
    ckb_offset = elwise_reduction->instantiate(
        elwise_reduction->static_data, 0, NULL, ckb, ckb_offset, dst_tp,
        dst_arrmeta, elwise_reduction_tp->get_npos(), &src_tp, &src_arrmeta,
        kernel_request_strided, ectx, nd::array(),
        std::map<std::string, ndt::type>());
  }
  // Make sure there's capacity for the next ckernel
  reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
      ->reserve(ckb_offset + sizeof(ckernel_prefix));
  // Need to retrieve 'e' again because it may have moved
  e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
          ->get_at<nd::strided_inner_broadcast_kernel_extra>(root_ckb_offset);
  e->dst_init_kernel_offset = ckb_offset - root_ckb_offset;
  if (dst_initialization != NULL) {
    ckb_offset = dst_initialization->instantiate(
        dst_initialization->static_data, 0, NULL, ckb, ckb_offset, dst_tp,
        dst_arrmeta, elwise_reduction_tp->get_npos(), &src_tp, &src_arrmeta,
        kernel_request_strided, ectx, nd::array(),
        std::map<std::string, ndt::type>());
  } else if (reduction_identity.is_null()) {
    ckb_offset =
        make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
                               src_arrmeta, kernel_request_strided, ectx);
  } else {
    ckb_offset = make_assignment_kernel(
        ckb, ckb_offset, dst_tp, dst_arrmeta, reduction_identity.get_type(),
        reduction_identity.get_arrmeta(), kernel_request_strided, ectx);
  }

  return ckb_offset;
}

size_t nd::make_lifted_reduction_ckernel(
    const callable_type_data *elwise_reduction_const,
    const ndt::callable_type *elwise_reduction_tp,
    const callable_type_data *dst_initialization_const,
    const ndt::callable_type *dst_initialization_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type &src_tp, const char *src_arrmeta, intptr_t reduction_ndim,
    const bool *reduction_dimflags, bool associative, bool commutative,
    bool right_associative, const nd::array &reduction_identity,
    dynd::kernel_request_t kernreq, const eval::eval_context *ectx)
{
  callable_type_data *elwise_reduction =
      const_cast<callable_type_data *>(elwise_reduction_const);
  callable_type_data *dst_initialization =
      const_cast<callable_type_data *>(dst_initialization_const);

  // Count the number of dimensions being reduced
  intptr_t reducedim_count = 0;
  for (intptr_t i = 0; i < reduction_ndim; ++i) {
    reducedim_count += reduction_dimflags[i];
  }
  if (reducedim_count == 0) {
    if (reduction_ndim == 0) {
      // If there are no dimensions to reduce, it's
      // just a dst_initialization operation, so create
      // that ckernel directly
      if (dst_initialization != NULL) {
        return dst_initialization->instantiate(
            dst_initialization->static_data, 0, NULL, ckb, ckb_offset, dst_tp,
            dst_arrmeta, elwise_reduction_tp->get_npos(), &src_tp, &src_arrmeta,
            kernreq, ectx, nd::array(), std::map<std::string, ndt::type>());
      } else if (reduction_identity.is_null()) {
        return make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta,
                                      src_tp, src_arrmeta, kernreq, ectx);
      } else {
        // Create the kernel which copies the identity and then
        // does one reduction
        return make_strided_inner_reduction_dimension_kernel(
            elwise_reduction, elwise_reduction_tp, dst_initialization,
            dst_initialization_tp, ckb, ckb_offset, 0, 1, dst_tp, dst_arrmeta,
            src_tp, src_arrmeta, right_associative, reduction_identity, kernreq,
            ectx);
      }
    }
    throw runtime_error("make_lifted_reduction_ckernel: no dimensions were "
                        "flagged for reduction");
  }

  if (!(reducedim_count == 1 || (associative && commutative))) {
    throw runtime_error(
        "make_lifted_reduction_ckernel: for reducing along multiple dimensions,"
        " the reduction function must be both associative and commutative");
  }
  if (right_associative) {
    throw runtime_error("make_lifted_reduction_ckernel: right_associative is "
                        "not yet supported");
  }

  ndt::type dst_el_tp = elwise_reduction_tp->get_return_type();
  ndt::type src_el_tp = elwise_reduction_tp->get_pos_type(0);

  // This is the number of dimensions being processed by the reduction
  if (reduction_ndim != src_tp.get_ndim() - src_el_tp.get_ndim()) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: wrong number of reduction "
          "dimensions, ";
    ss << "requested " << reduction_ndim << ", but types have ";
    ss << (src_tp.get_ndim() - src_el_tp.get_ndim());
    ss << " lifting from " << src_el_tp << " to " << src_tp;
    throw runtime_error(ss.str());
  }
  // Determine whether reduced dimensions are being kept or not
  bool keep_dims;
  if (reduction_ndim == dst_tp.get_ndim() - dst_el_tp.get_ndim()) {
    keep_dims = true;
  } else if (reduction_ndim - reducedim_count ==
             dst_tp.get_ndim() - dst_el_tp.get_ndim()) {
    keep_dims = false;
  } else {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: The number of dimensions flagged for "
          "reduction, ";
    ss << reducedim_count << ", is not consistent with the destination type ";
    ss << "reducing " << dst_tp << " with element " << dst_el_tp;
    throw runtime_error(ss.str());
  }

  ndt::type dst_i_tp = dst_tp, src_i_tp = src_tp;

  for (intptr_t i = 0; i < reduction_ndim; ++i) {
    intptr_t dst_stride, dst_size, src_stride, src_size;
    // Get the striding parameters for the source dimension
    if (!src_i_tp.get_as_strided(src_arrmeta, &src_size, &src_stride, &src_i_tp,
                                 &src_arrmeta)) {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: type " << src_i_tp
         << " not supported as source";
      throw type_error(ss.str());
    }
    if (reduction_dimflags[i]) {
      // This dimension is being reduced
      if (src_size == 0 && reduction_identity.is_null()) {
        // If the size of the src is 0, a reduction identity is required to get
        // a value
        stringstream ss;
        ss << "cannot reduce a zero-sized dimension (axis ";
        ss << i << " of " << src_i_tp << ") because the operation";
        ss << " has no identity";
        throw invalid_argument(ss.str());
      }
      if (keep_dims) {
        // If the dimensions are being kept, the output should be a
        // a strided dimension of size one
        if (dst_i_tp.get_as_strided(dst_arrmeta, &dst_size, &dst_stride,
                                    &dst_i_tp, &dst_arrmeta)) {
          if (dst_size != 1 || dst_stride != 0) {
            stringstream ss;
            ss << "make_lifted_reduction_ckernel: destination of a reduction "
                  "dimension ";
            ss << "must have size 1, not size" << dst_size << "/stride "
               << dst_stride;
            ss << " in type " << dst_i_tp;
            throw type_error(ss.str());
          }
        } else {
          stringstream ss;
          ss << "make_lifted_reduction_ckernel: type " << dst_i_tp;
          ss << " not supported the destination of a dimension being reduced";
          throw type_error(ss.str());
        }
      }
      if (i < reduction_ndim - 1) {
        // An initial dimension being reduced
        ckb_offset = make_strided_initial_reduction_dimension_kernel(
            ckb, ckb_offset, src_stride, src_size, kernreq);
        // The next request should be single, as that's the kind of
        // ckernel the 'first_call' should be in this case
        kernreq = kernel_request_single;
      } else {
        // The innermost dimension being reduced
        return make_strided_inner_reduction_dimension_kernel(
            elwise_reduction, elwise_reduction_tp, dst_initialization,
            dst_initialization_tp, ckb, ckb_offset, src_stride, src_size,
            dst_i_tp, dst_arrmeta, src_i_tp, src_arrmeta, right_associative,
            reduction_identity, kernreq, ectx);
      }
    } else {
      // This dimension is being broadcast, not reduced
      if (!dst_i_tp.get_as_strided(dst_arrmeta, &dst_size, &dst_stride,
                                   &dst_i_tp, &dst_arrmeta)) {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: type " << dst_i_tp
           << " not supported as destination";
        throw type_error(ss.str());
      }
      if (dst_size != src_size) {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: the dst dimension size "
           << dst_size;
        ss << " must equal the src dimension size " << src_size
           << " for broadcast dimensions";
        throw runtime_error(ss.str());
      }
      if (i < reduction_ndim - 1) {
        // An initial dimension being broadcast
        ckb_offset = make_strided_initial_broadcast_dimension_kernel(
            ckb, ckb_offset, dst_stride, src_stride, src_size, kernreq);
        // The next request should be strided, as that's the kind of
        // ckernel the 'first_call' should be in this case
        kernreq = kernel_request_strided;
      } else {
        // The innermost dimension being broadcast
        return make_strided_inner_broadcast_dimension_kernel(
            elwise_reduction, elwise_reduction_tp, dst_initialization,
            dst_initialization_tp, ckb, ckb_offset, dst_stride, src_stride,
            src_size, dst_i_tp, dst_arrmeta, src_i_tp, src_arrmeta,
            right_associative, reduction_identity, kernreq, ectx);
      }
    }
  }

  throw runtime_error("make_lifted_reduction_ckernel: internal error, "
                      "should have returned in the loop");
}
