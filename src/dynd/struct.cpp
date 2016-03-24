//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/field_access_kernel.hpp>
#include <dynd/callables/field_access_callable.hpp>
#include <dynd/struct.hpp>
#include <dynd/types/callable_type.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::field_access::make() { return make_callable<field_access_callable>(); }

DYND_DEFAULT_DECLFUNC_GET(nd::field_access)

DYND_API struct nd::field_access nd::field_access;

// Temporary solution until nd::field_access() handles views.
DYND_API nd::callable nd::make_field_access_kernel(const ndt::type &dt, const std::string &name)
{
  if (dt.get_id() != struct_id) {
    throw invalid_argument(std::string("no property named '") + name + "'");
  }

  intptr_t i = dt.extended<ndt::struct_type>()->get_field_index(name);
  if (i < 0) {
    throw std::invalid_argument("no field named '" + name + "'");
  }

  nd::callable c = make_callable<get_array_field_callable>(i);

  return c;
}
