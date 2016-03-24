//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_instantiable_callable.hpp>
#include <dynd/kernels/equal_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID, type_id_t Arg1ID>
  class equal_callable : public base_instantiable_callable<equal_kernel<Arg0ID, Arg1ID>> {
  public:
    equal_callable()
        : base_instantiable_callable<equal_kernel<Arg0ID, Arg1ID>>(
              ndt::callable_type::make(ndt::make_type<bool1>(), {ndt::type(Arg0ID), ndt::type(Arg1ID)}))
    {
    }
  };

  template <>
  class equal_callable<tuple_id, tuple_id> : public base_callable {
  public:
    equal_callable()
        : base_callable(ndt::callable_type::make(ndt::make_type<bool1>(), {ndt::type(tuple_id), ndt::type(tuple_id)}))
    {
    }

    void instantiate(char *static_data, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                     const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars);
  };

} // namespace dynd::nd
} // namespace dynd
