//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>

namespace dynd {
namespace nd {

  template <typename T>
  struct static_data_callable : base_callable {
    typedef typename std::remove_reference<T>::type static_data_type;
    static_data_type static_data;

    static_data_callable(const ndt::type &tp, kernel_request_t kernreq, kernel_targets_t targets,
                         const volatile char *ir, callable_data_init_t data_init,
                         callable_resolve_dst_type_t resolve_dst_type, callable_instantiate_t instantiate,
                         T &&static_data)
        : base_callable(tp, kernreq, targets, ir, data_init, resolve_dst_type, instantiate),
          static_data(std::forward<T>(static_data))
    {
      static_assert(scalar_align_of<static_data_type>::value <= scalar_align_of<std::uint64_t>::value,
                    "static data requires stronger alignment");
    }

    virtual ~static_data_callable() {}
  };

} // namespace dynd::nd
} // namespace dynd
