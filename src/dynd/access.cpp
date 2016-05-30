//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/access.hpp>
#include <dynd/array.hpp>
#include <dynd/callable.hpp>
#include <dynd/callables/field_access_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/field_access_kernel.hpp>
#include <dynd/types/callable_type.hpp>

using namespace std;
using namespace dynd;

namespace {

class access_dispatch_callable : public nd::base_callable {
public:
  access_dispatch_callable()
      : base_callable(ndt::make_type<ndt::callable_type>(ndt::type("Any"), {ndt::type("Any")},
                                                         {{ndt::make_type<std::string>(), "name"}})) {}

  ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), nd::call_graph &cg,
                    const ndt::type &res_tp, size_t narg, const ndt::type *arg_tp, size_t nkwd, const nd::array *kwds,
                    const std::map<std::string, ndt::type> &tp_vars) {
    static nd::callable array_field_access = nd::make_callable<nd::get_array_field_callable>();

    return array_field_access->resolve(this, nullptr, cg, res_tp, narg, arg_tp, nkwd, kwds, tp_vars);
  }
};

} // unnamed namespace

DYND_API nd::callable nd::access = nd::make_callable<access_dispatch_callable>();

DYND_API nd::callable nd::field_access = nd::make_callable<nd::field_access_callable>();
