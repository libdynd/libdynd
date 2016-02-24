//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/assignment.hpp>
#include <dynd/callable.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/type.hpp>

using namespace std;
using namespace dynd;

namespace {

////////////////////////////////////////////////////////////////
// Functions for the unary assignment as an callable

struct unary_assignment_ck : nd::base_kernel<unary_assignment_ck, 1> {
  static void instantiate(char *static_data, char *DYND_UNUSED(data), nd::kernel_builder *ckb, const ndt::type &dst_tp,
                          const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                          const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                          const nd::array *DYND_UNUSED(kwds),
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    assign_error_mode errmode = *reinterpret_cast<assign_error_mode *>(static_data);
    eval::eval_context ectx_tmp;
    ectx_tmp.errmode = errmode;
    make_assignment_kernel(ckb, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq, &ectx_tmp);
  }
};

} // anonymous namespace

nd::callable dynd::make_callable_from_assignment(const ndt::type &dst_tp, const ndt::type &src_tp,
                                                 assign_error_mode errmode)
{
  return nd::callable::make<unary_assignment_ck>(ndt::callable_type::make(dst_tp, src_tp), errmode);
}

void nd::detail::check_narg(const ndt::callable_type *af_tp, intptr_t narg)
{
  if (!af_tp->is_pos_variadic() && narg != af_tp->get_npos()) {
    std::stringstream ss;
    ss << "callable expected " << af_tp->get_npos() << " positional arguments, but received " << narg;
    throw std::invalid_argument(ss.str());
  }
}

void nd::detail::check_arg(const ndt::callable_type *af_tp, intptr_t i, const ndt::type &actual_tp,
                           const char *actual_arrmeta, std::map<std::string, ndt::type> &tp_vars)
{
  if (af_tp->is_pos_variadic()) {
    return;
  }

  ndt::type expected_tp = af_tp->get_pos_type(i);
  ndt::type candidate_tp = actual_tp;
  if (actual_tp.get_id() != pointer_id) {
    candidate_tp = candidate_tp.value_type();
  }

  if (!expected_tp.match(NULL, candidate_tp, actual_arrmeta, tp_vars)) {
    std::stringstream ss;
    ss << "positional argument " << i << " to callable does not match, ";
    ss << "expected " << expected_tp << ", received " << actual_tp;
    throw std::invalid_argument(ss.str());
  }
}

nd::array nd::callable::call(size_t args_size, const array *args_values, size_t kwds_size,
                             const std::pair<const char *, array> *kwds_values)
{
  std::map<std::string, ndt::type> tp_vars;
  const ndt::callable_type *self_tp = get_type();

  if (!self_tp->is_pos_variadic() && (static_cast<intptr_t>(args_size) < self_tp->get_npos())) {
    std::stringstream ss;
    ss << "callable expected " << self_tp->get_npos() << " positional arguments, but received " << args_size;
    throw std::invalid_argument(ss.str());
  }

  std::vector<ndt::type> args_tp(args_size);
  std::vector<const char *> args_arrmeta(args_size);

  for (intptr_t i = 0; i < (self_tp->is_pos_variadic() ? static_cast<intptr_t>(args_size) : self_tp->get_npos()); ++i) {
    detail::check_arg(self_tp, i, args_values[i]->tp, args_values[i]->metadata(), tp_vars);

    args_tp[i] = args_values[i]->tp;
    args_arrmeta[i] = args_values[i]->metadata();
  }

  array dst;

  intptr_t narg = args_size;

  // ...
  intptr_t nkwd = args_size - self_tp->get_npos();
  if (!self_tp->is_kwd_variadic() && nkwd > self_tp->get_nkwd()) {
    throw std::invalid_argument("too many extra positional arguments");
  }

  std::vector<array> kwds_as_vector(nkwd + self_tp->get_nkwd());
  for (intptr_t i = 0; i < nkwd; ++i) {
    kwds_as_vector[i] = args_values[self_tp->get_npos() + i];
    --narg;
  }

  for (size_t i = 0; i < kwds_size; ++i) {
    intptr_t j = self_tp->get_kwd_index(kwds_values[i].first);
    if (j == -1) {
      if (detail::is_special_kwd(self_tp, dst, kwds_values[i].first, kwds_values[i].second)) {
      }
      else {
        std::stringstream ss;
        ss << "passed an unexpected keyword \"" << kwds_values[i].first << "\" to callable with type " << get()->tp;
        throw std::invalid_argument(ss.str());
      }
    }
    else {
      array &value = kwds_as_vector[j];
      if (!value.is_null()) {
        std::stringstream ss;
        ss << "callable passed keyword \"" << kwds_values[i].first << "\" more than once";
        throw std::invalid_argument(ss.str());
      }
      value = kwds_values[i].second;

      ndt::type expected_tp = self_tp->get_kwd_type(j);
      if (expected_tp.get_id() == option_id) {
        expected_tp = expected_tp.extended<ndt::option_type>()->get_value_type();
      }

      const ndt::type &actual_tp = value.get_type();
      if (!expected_tp.match(actual_tp.value_type(), tp_vars)) {
        std::stringstream ss;
        ss << "keyword \"" << self_tp->get_kwd_name(j) << "\" does not match, ";
        ss << "callable expected " << expected_tp << " but passed " << actual_tp;
        throw std::invalid_argument(ss.str());
      }
      ++nkwd;
    }
  }

  // Validate the destination type, if it was provided
  if (!dst.is_null()) {
    if (!self_tp->get_return_type().match(NULL, dst.get_type(), dst.get()->metadata(), tp_vars)) {
      std::stringstream ss;
      ss << "provided \"dst\" type " << dst.get_type() << " does not match callable return type "
         << self_tp->get_return_type();
      throw std::invalid_argument(ss.str());
    }
  }

  for (intptr_t j : self_tp->get_option_kwd_indices()) {
    if (kwds_as_vector[j].is_null()) {
      ndt::type actual_tp = ndt::substitute(self_tp->get_kwd_type(j), tp_vars, false);
      if (actual_tp.is_symbolic()) {
        actual_tp = ndt::make_type<ndt::option_type>(ndt::make_type<void>());
      }
      kwds_as_vector[j] = empty(actual_tp);
      kwds_as_vector[j].assign_na();
      ++nkwd;
    }
  }

  if (nkwd < self_tp->get_nkwd()) {
    std::stringstream ss;
    // TODO: Provide the missing keyword parameter names in this error
    //       message
    ss << "callable requires keyword parameters that were not provided. "
          "callable signature " << get()->tp;
    throw std::invalid_argument(ss.str());
  }

  ndt::type dst_tp;
  if (dst.is_null()) {
    dst_tp = self_tp->get_return_type();
    return get()->call(dst_tp, narg, args_tp.data(), args_arrmeta.data(), args_values, nkwd, kwds_as_vector.data(),
                       tp_vars);
  }

  dst_tp = dst.get_type();
  get()->call(dst_tp, dst->metadata(), &dst, narg, args_tp.data(), args_arrmeta.data(), args_values, nkwd,
              kwds_as_vector.data(), tp_vars);
  return dst;
}
