//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/types/expr_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/type.hpp>

using namespace std;
using namespace dynd;

namespace {

////////////////////////////////////////////////////////////////
// Functions for the unary assignment as an arrfunc

static intptr_t instantiate_assignment_ckernel(
    const arrfunc_type_data *self, const arrfunc_type *af_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &kwds, const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  try {
    assign_error_mode errmode = *self->get_data_as<assign_error_mode>();
    if (dst_tp == af_tp->get_return_type() &&
        src_tp[0] == af_tp->get_pos_type(0)) {
      if (errmode == ectx->errmode) {
        return make_assignment_kernel(self, af_tp, ckb, ckb_offset, dst_tp, dst_arrmeta,
                                      src_tp[0], src_arrmeta[0], kernreq, ectx, kwds);
      } else {
        eval::eval_context ectx_tmp(*ectx);
        ectx_tmp.errmode = errmode;
        return make_assignment_kernel(self, af_tp, ckb, ckb_offset, dst_tp, dst_arrmeta,
                                      src_tp[0], src_arrmeta[0], kernreq,
                                      &ectx_tmp, kwds);
      }
    } else {
      stringstream ss;
      ss << "Cannot instantiate arrfunc for assigning from ";
      ss << af_tp->get_pos_type(0) << " to " << af_tp->get_return_type();
      ss << " using input type " << src_tp[0];
      ss << " and output type " << dst_tp;
      throw type_error(ss.str());
    }
  }
  catch (const std::exception &e) {
    cout << "exception: " << e.what() << endl;
    throw;
  }
}

////////////////////////////////////////////////////////////////
// Functions for property access as an arrfunc

static void delete_property_arrfunc_data(arrfunc_type_data *self_af)
{
  base_type_xdecref(*self_af->get_data_as<const base_type *>());
}

static intptr_t instantiate_property_ckernel(
    const arrfunc_type_data *self, const arrfunc_type *af_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &kwds, const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  ndt::type prop_src_tp(*self->get_data_as<const base_type *>(), true);

  if (dst_tp.value_type() == prop_src_tp.value_type()) {
    if (src_tp[0] == prop_src_tp.operand_type()) {
      return make_assignment_kernel(NULL, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta,
                                    prop_src_tp, src_arrmeta[0], kernreq, ectx, kwds);
    } else if (src_tp[0].value_type() == prop_src_tp.operand_type()) {
      return make_assignment_kernel(
          NULL, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta,
          prop_src_tp.extended<base_expr_type>()->with_replaced_storage_type(
              src_tp[0]),
          src_arrmeta[0], kernreq, ectx, kwds);
    }
  }

  stringstream ss;
  ss << "Cannot instantiate arrfunc for assigning from ";
  ss << af_tp->get_pos_type(0) << " to " << af_tp->get_return_type();
  ss << " using input type " << src_tp[0];
  ss << " and output type " << dst_tp;
  throw type_error(ss.str());
}

} // anonymous namespace

nd::arrfunc dynd::make_arrfunc_from_assignment(const ndt::type &dst_tp,
                                               const ndt::type &src_tp,
                                               assign_error_mode errmode)
{
  nd::array af = nd::empty(ndt::make_arrfunc(ndt::make_tuple(src_tp), dst_tp));
  arrfunc_type_data *out_af =
      reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
  memset(out_af, 0, sizeof(arrfunc_type_data));
  *out_af->get_data_as<assign_error_mode>() = errmode;
  out_af->free = NULL;
  out_af->instantiate = &instantiate_assignment_ckernel;
  af.flag_as_immutable();
  return af;
}

nd::arrfunc dynd::make_arrfunc_from_property(const ndt::type &tp,
                                             const std::string &propname)
{
  if (tp.get_kind() == expr_kind) {
    stringstream ss;
    ss << "Creating an arrfunc from a property requires a non-expression"
       << ", got " << tp;
    throw type_error(ss.str());
  }
  ndt::type prop_tp = ndt::make_property(tp, propname);
  nd::array af =
      nd::empty(ndt::make_arrfunc(ndt::make_tuple(tp), prop_tp.value_type()));
  arrfunc_type_data *out_af =
      reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
  out_af->free = &delete_property_arrfunc_data;
  *out_af->get_data_as<const base_type *>() = prop_tp.release();
  out_af->instantiate = &instantiate_property_ckernel;
  af.flag_as_immutable();
  return af;
}

void nd::detail::validate_kwd_types(const arrfunc_type *af_tp,
                                    std::vector<ndt::type> &kwd_tp,
                                    const std::vector<intptr_t> &available,
                                    const std::vector<intptr_t> &missing,
                                    std::map<nd::string, ndt::type> &tp_vars)
{
  for (intptr_t j : available) {
    if (j == -1) {
      continue;
    }

    ndt::type &actual_tp = kwd_tp[j];

    ndt::type expected_tp = af_tp->get_kwd_type(j);
    if (expected_tp.get_type_id() == option_type_id) {
      expected_tp = expected_tp.p("value_type").as<ndt::type>();
    }

    if (!actual_tp.value_type().matches(expected_tp, tp_vars)) {
      std::stringstream ss;
      ss << "keyword \"" << af_tp->get_kwd_name(j) << "\" does not match, ";
      ss << "arrfunc expected " << expected_tp << " but passed " << actual_tp;
      throw std::invalid_argument(ss.str());
    }

    if (j != -1 && kwd_tp[j].get_kind() == dim_kind) {
      kwd_tp[j] = ndt::make_pointer(kwd_tp[j]);
    }
  }

  for (intptr_t j : missing) {
    ndt::type &actual_tp = kwd_tp[j];
    actual_tp = ndt::substitute(af_tp->get_kwd_type(j), tp_vars, false);
    if (actual_tp.is_symbolic()) {
      actual_tp = ndt::make_option(ndt::make_type<void>());
    }
  }
}

void nd::detail::fill_missing_values(const ndt::type *tp, char *arrmeta,
                                     const uintptr_t *arrmeta_offsets,
                                     char *data, const uintptr_t *data_offsets,
                                     const std::vector<intptr_t> &missing)
{
  for (intptr_t j : missing) {
    tp[j].extended()->arrmeta_default_construct(arrmeta + arrmeta_offsets[j],
                                                true);
    assign_na(tp[j], arrmeta + arrmeta_offsets[j], data + data_offsets[j],
              &eval::default_eval_context);
  }
}

void nd::detail::check_narg(const arrfunc_type *af_tp, intptr_t npos)
{
  if (!af_tp->is_pos_variadic() && npos != af_tp->get_npos()) {
    std::stringstream ss;
    ss << "arrfunc expected " << af_tp->get_npos()
       << " positional arguments, but received " << npos;
    throw std::invalid_argument(ss.str());
  }
}

void nd::detail::check_arg(const arrfunc_type *af_tp, intptr_t i,
                           const ndt::type &actual_tp,
                           const char *actual_arrmeta,
                           std::map<nd::string, ndt::type> &tp_vars)
{
  ndt::type expected_tp = af_tp->get_pos_type(i);
  if (!actual_tp.value_type().matches(actual_arrmeta, expected_tp, NULL,
                                      tp_vars)) {
    std::stringstream ss;
    ss << "positional argument " << i << " to arrfunc does not match, ";
    ss << "expected " << expected_tp << ", received " << actual_tp;
    throw std::invalid_argument(ss.str());
  }
}

void nd::detail::check_nkwd(const arrfunc_type *af_tp,
                            const std::vector<intptr_t> &available,
                            const std::vector<intptr_t> &missing)
{
  if (intptr_t(available.size() + missing.size()) < af_tp->get_nkwd()) {
    std::stringstream ss;
    // TODO: Provide the missing keyword parameter names in this error
    //       message
    ss << "arrfunc requires keyword parameters that were not provided. "
          "arrfunc signature " << ndt::type(af_tp, true);
    throw std::invalid_argument(ss.str());
  }
}

nd::arrfunc::arrfunc(const nd::array &rhs)
{
  if (!rhs.is_null()) {
    if (rhs.get_type().get_type_id() == arrfunc_type_id) {
      if (rhs.is_immutable()) {
        const arrfunc_type_data *af =
            reinterpret_cast<const arrfunc_type_data *>(
                rhs.get_readonly_originptr());
        if (af->instantiate != NULL) {
          // It's valid: immutable, arrfunc type, contains
          // instantiate function.
          m_value = rhs;
        } else {
          throw invalid_argument("Require a non-empty arrfunc, "
                                 "provided arrfunc has NULL "
                                 "instantiate function");
        }
      } else {
        stringstream ss;
        ss << "Require an immutable arrfunc, provided arrfunc";
        rhs.get_type().extended()->print_data(ss, rhs.get_arrmeta(),
                                              rhs.get_readonly_originptr());
        ss << " is not immutable";
        throw invalid_argument(ss.str());
      }
    } else {
      stringstream ss;
      ss << "Cannot implicitly convert nd::array of type "
         << rhs.get_type().value_type() << " to  arrfunc";
      throw type_error(ss.str());
    }
  }
}