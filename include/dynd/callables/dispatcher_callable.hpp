//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>

namespace dynd {
namespace nd {

  template <typename...>
  struct dispatch_callable;

  template <typename SpecializerType>
  struct dispatch_callable<SpecializerType> : base_callable {
    SpecializerType specializer;

    dispatch_callable(const ndt::type &tp, kernel_targets_t targets, const volatile char *ir, callable_alloc_t alloc,
                      callable_data_init_t data_init, callable_resolve_dst_type_t resolve_dst_type,
                      callable_instantiate_t instantiate, const SpecializerType &specializer)
        : base_callable(tp, targets, ir, alloc, data_init, resolve_dst_type, instantiate), specializer(specializer)
    {
    }

    const callable &specialize(const ndt::type &ret_tp, intptr_t narg, const ndt::type *arg_tp)
    {
      return specializer(ret_tp, narg, arg_tp);
    }
  };

  template <typename OverloaderType, typename SpecializerType>
  struct dispatch_callable<OverloaderType, SpecializerType> : base_callable {
    SpecializerType specializer;
    OverloaderType overloader;

    dispatch_callable(const ndt::type &tp, kernel_targets_t targets, const volatile char *ir, callable_alloc_t alloc,
                      callable_data_init_t data_init, callable_resolve_dst_type_t resolve_dst_type,
                      callable_instantiate_t instantiate, const OverloaderType &overloader,
                      const SpecializerType &specializer)
        : base_callable(tp, targets, ir, alloc, data_init, resolve_dst_type, instantiate), specializer(specializer),
          overloader(overloader)
    {
    }

    void overload(const ndt::type &ret_tp, intptr_t narg, const ndt::type *arg_tp, const callable &value)
    {
      overloader(ret_tp, narg, arg_tp, value);
    }

    const callable &specialize(const ndt::type &ret_tp, intptr_t narg, const ndt::type *arg_tp)
    {
      return specializer(ret_tp, narg, arg_tp);
    }
  };

} // namespace dynd::nd
} // namespace dynd
