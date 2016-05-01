//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/arithmetic.hpp>
#include <dynd/assignment.hpp>
#include <dynd/comparison.hpp>
#include <dynd/index.hpp>
#include <dynd/io.hpp>
#include <dynd/math.hpp>
#include <dynd/option.hpp>
#include <dynd/pointer.hpp>
#include <dynd/random.hpp>
#include <dynd/range.hpp>
#include <dynd/statistics.hpp>

using namespace std;
using namespace dynd;

namespace {

////////////////////////////////////////////////////////////////
// Functions for the unary assignment as an callable

class unary_assignment_callable : public nd::base_callable {
  assign_error_mode errmode;

public:
  unary_assignment_callable(const ndt::type &tp, assign_error_mode error_mode)
      : base_callable(tp), errmode(error_mode) {}

  ndt::type resolve(nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), nd::call_graph &cg,
                    const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                    size_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                    const std::map<std::string, ndt::type> &tp_vars) {
    nd::array error_mode = errmode;
    return nd::assign->resolve(this, nullptr, cg, dst_tp, 1, src_tp, 1, &error_mode, tp_vars);
  }
};

} // anonymous namespace

std::map<std::string, nd::callable> &nd::detail::get_regfunctions() {
  static map<std::string, callable> registry{{"add", add},
                                             {"assign", assign},
                                             {"assign_na", assign_na},
                                             {"bitwise_and", bitwise_and},
                                             {"bitwise_not", bitwise_not},
                                             {"bitwise_or", bitwise_or},
                                             {"bitwise_xor", bitwise_xor},
                                             {"cbrt", cbrt},
                                             {"compound_add", compound_add},
                                             {"compound_div", compound_div},
                                             {"conj", conj},
                                             {"cos", cos},
                                             {"dereference", dereference},
                                             {"divide", divide},
                                             {"equal", equal},
                                             {"exp", exp},
                                             {"greater", greater},
                                             {"greater_equal", greater_equal},
                                             {"imag", imag},
                                             {"is_na", is_na},
                                             {"left_shift", left_shift},
                                             {"less", less},
                                             {"less_equal", less_equal},
                                             {"logical_and", logical_and},
                                             {"logical_not", logical_not},
                                             {"logical_or", logical_or},
                                             {"logical_xor", logical_xor},
                                             {"max", max},
                                             {"min", min},
                                             {"minus", minus},
                                             {"mod", mod},
                                             {"multiply", multiply},
                                             {"not_equal", not_equal},
                                             {"plus", plus},
                                             {"pow", pow},
                                             {"range", range},
                                             {"real", real},
                                             {"right_shift", right_shift},
                                             {"serialize", serialize},
                                             {"sin", sin},
                                             {"sqrt", sqrt},
                                             {"subtract", subtract},
                                             {"sum", sum},
                                             {"take", take},
                                             {"tan", tan},
                                             {"total_order", total_order},
                                             {"uniform", random::uniform}};

  return registry;
}

nd::callable dynd::make_callable_from_assignment(const ndt::type &dst_tp, const ndt::type &src_tp,
                                                 assign_error_mode errmode) {
  return nd::make_callable<unary_assignment_callable>(ndt::make_type<ndt::callable_type>(dst_tp, {src_tp}), errmode);
}

void nd::detail::check_narg(const base_callable *self, size_t narg) {
  if (!self->is_arg_variadic() && narg != self->get_narg()) {
    std::stringstream ss;
    ss << "callable expected " << self->get_narg() << " positional arguments, but received " << narg;
    throw std::invalid_argument(ss.str());
  }
}

void nd::detail::check_arg(const base_callable *self, intptr_t i, const ndt::type &actual_tp,
                           const char *DYND_UNUSED(actual_arrmeta), std::map<std::string, ndt::type> &tp_vars) {
  if (self->is_arg_variadic()) {
    return;
  }

  ndt::type expected_tp = self->get_arg_types()[i];
  ndt::type candidate_tp = actual_tp;

  if (!expected_tp.match(candidate_tp, tp_vars)) {
    std::stringstream ss;
    ss << "positional argument " << i << " to callable does not match, ";
    ss << "expected " << expected_tp << ", received " << actual_tp;
    throw std::invalid_argument(ss.str());
  }
}

nd::array nd::callable::call(size_t narg, const array *args, size_t nkwd,
                             const pair<const char *, array> *unordered_kwds) const {
  std::map<std::string, ndt::type> tp_vars;

  if (!m_ptr->is_arg_variadic() && (narg < m_ptr->get_narg())) {
    std::stringstream ss;
    ss << "callable expected " << m_ptr->get_narg() << " positional arguments, but received " << narg;
    throw std::invalid_argument(ss.str());
  }

  unique_ptr<ndt::type[]> args_tp(new ndt::type[narg]);
  unique_ptr<const char *[]> args_arrmeta(new const char *[narg]);
  unique_ptr<array[]> kwds(new array[narg + m_ptr->get_nkwd()]);

  size_t j = 0;
  if (m_ptr->is_arg_variadic()) {
    for (size_t i = 0; i < narg; ++i) {
      detail::check_arg(m_ptr, i, args[i]->tp, args[i]->metadata(), tp_vars);

      args_tp[i] = args[i]->tp;
      args_arrmeta[i] = args[i]->metadata();
    }
  } else {
    size_t i = 0;
    for (; i < m_ptr->get_narg(); ++i) {
      detail::check_arg(m_ptr, i, args[i]->tp, args[i]->metadata(), tp_vars);

      args_tp[i] = args[i]->tp;
      args_arrmeta[i] = args[i]->metadata();
    }

    // ...
    if (!m_ptr->is_kwd_variadic() && (narg - m_ptr->get_narg()) > m_ptr->get_nkwd()) {
      throw std::invalid_argument("too many extra positional arguments");
    }

    for (; narg > m_ptr->get_narg(); ++i, --narg, ++j, ++nkwd) {
      kwds[j] = args[i];
    }
  }

  array dst;

  const std::vector<std::pair<ndt::type, std::string>> kwd_tp = m_ptr->get_kwd_types();
  for (; j < nkwd; ++j, ++unordered_kwds) {
    intptr_t k = m_ptr->get_kwd_index(unordered_kwds->first);

    if (k == -1) {
      if (detail::is_special_kwd(dst, unordered_kwds->first, unordered_kwds->second)) {
      } else {
        std::stringstream ss;
        ss << "passed an unexpected keyword \"" << unordered_kwds->first << "\" to callable with type "
           << m_ptr->get_type();
        throw std::invalid_argument(ss.str());
      }
    } else {
      array &value = kwds[k];
      if (!value.is_null()) {
        std::stringstream ss;
        ss << "callable passed keyword \"" << unordered_kwds->first << "\" more than once";
        throw std::invalid_argument(ss.str());
      }
      value = unordered_kwds->second;

      ndt::type expected_tp = kwd_tp[k].first;
      if (expected_tp.get_id() == option_id) {
        expected_tp = expected_tp.extended<ndt::option_type>()->get_value_type();
      }

      const ndt::type &actual_tp = value.get_type();
      if (!expected_tp.match(actual_tp.value_type(), tp_vars)) {
        std::stringstream ss;
        ss << "keyword \"" << kwd_tp[k].second << "\" does not match, ";
        ss << "callable expected " << expected_tp << " but passed " << actual_tp;
        throw std::invalid_argument(ss.str());
      }
    }
  }

  // Validate the destination type, if it was provided
  if (!dst.is_null()) {
    if (!m_ptr->get_ret_type().match(dst.get_type(), tp_vars)) {
      std::stringstream ss;
      ss << "provided \"dst\" type " << dst.get_type() << " does not match callable return type "
         << m_ptr->get_ret_type();
      throw std::invalid_argument(ss.str());
    }
  }

  for (intptr_t j : m_ptr->get_option_kwd_indices()) {
    if (kwds[j].is_null()) {
      ndt::type actual_tp = ndt::substitute(kwd_tp[j].first, tp_vars, false);
      if (actual_tp.is_symbolic()) {
        actual_tp = ndt::make_type<ndt::option_type>(ndt::make_type<void>());
      }
      kwds[j] = assign_na({{"dst_tp", actual_tp}});
      ++nkwd;
    }
  }

  if (nkwd < m_ptr->get_nkwd()) {
    std::stringstream ss;
    // TODO: Provide the missing keyword parameter names in this error
    //       message
    ss << "callable requires keyword parameters that were not provided. "
          "callable signature "
       << m_ptr->get_type();
    throw std::invalid_argument(ss.str());
  }

  ndt::type dst_tp;
  if (dst.is_null()) {
    dst_tp = m_ptr->get_ret_type();
    return m_ptr->call(dst_tp, narg, args_tp.get(), args_arrmeta.get(), args, nkwd, kwds.get(), tp_vars);
  }

  dst_tp = dst.get_type();
  m_ptr->call(dst_tp, dst->metadata(), &dst, narg, args_tp.get(), args_arrmeta.get(), args, nkwd, kwds.get(), tp_vars);
  return dst;
}

std::map<std::string, nd::callable> &nd::callables() { return detail::get_regfunctions(); }

nd::callable &nd::reg(const std::string &name) {
  std::map<std::string, callable> &registry = detail::get_regfunctions();

  auto it = registry.find(name);
  if (it != registry.end()) {
    return it->second;
  }

  stringstream ss;
  ss << "No dynd function ";
  print_escaped_utf8_string(ss, name);
  ss << " has been registered";
  throw invalid_argument(ss.str());
}

void nd::reg(const std::string &name, const nd::callable &f) {
  std::map<std::string, callable> &registry = detail::get_regfunctions();
  registry[name] = f;
}
