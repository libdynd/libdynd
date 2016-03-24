//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/compose.hpp>
#include <dynd/callables/compose_callable.hpp>
#include <dynd/types/substitute_typevars.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::functional::compose(const nd::callable &first, const nd::callable &second, const ndt::type &buf_tp)
{
  if (first.get_type()->get_npos() != 1) {
    throw runtime_error("Multi-parameter callable chaining is not implemented");
  }

  if (second.get_type()->get_npos() != 1) {
    stringstream ss;
    ss << "Cannot chain functions " << first << " and " << second << ", because the second function is not unary";
    throw invalid_argument(ss.str());
  }

  if (buf_tp.get_id() == uninitialized_id) {
    throw runtime_error("Chaining functions without a provided intermediate "
                        "type is not implemented");
  }

  /* // TODO: Something like this should work
  map<nd::string, ndt::type> tp_vars;
  second.get_type()->get_pos_type(0).match(first.get_type()->get_return_type(),
                                           tp_vars);
  ndt::type return_tp =
      ndt::substitute(second.get_type()->get_return_type(), tp_vars, false);
  */

  return make_callable<compose_callable>(
      ndt::callable_type::make(second.get_type()->get_return_type(), first.get_type()->get_pos_tuple()), first, second,
      buf_tp);
}
