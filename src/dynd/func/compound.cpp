//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/compound_callable.hpp>
#include <dynd/func/compound.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::functional::left_compound(const callable &child)
{
  vector<ndt::type> pos_types = child.get_type()->get_pos_types();
  pos_types.resize(1);

  return make_callable<left_compound_callable>(
      ndt::callable_type::make(child.get_type()->get_return_type(), pos_types), // head element or empty
      child);
}

nd::callable nd::functional::right_compound(const callable &child)
{
  vector<ndt::type> pos_types = child.get_type()->get_pos_types();
  pos_types.erase(pos_types.begin());

  return make_callable<right_compound_callable>(ndt::callable_type::make(child.get_type()->get_return_type(),
                                                                         pos_types), // tail elements or empty
                                                child);
}
