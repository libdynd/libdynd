//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// String count kernel

#pragma once

#include <dynd/string.hpp>
#include <dynd/callable.hpp>
#include <dynd/kernels/kernel_builder.hpp>
#include <dynd/types/string_type.hpp>

namespace dynd {
  namespace nd {

    struct string_count_kernel
      : base_kernel<string_count_kernel, 2> {

      void single(char *dst, char *const *src) {
        intptr_t *d = reinterpret_cast<intptr_t *>(dst);
        const dynd::string *const *s = reinterpret_cast<const dynd::string *const *>(src);

        *d = dynd::string_count(s[0], s[1]);
      }
    };

  } // namespace nd

  namespace ndt {

    template<>
    struct traits<dynd::nd::string_count_kernel> {
      /* TODO: Rather than int64_id, I really want the id that corresponds to
               intptr_t... */
      static type equivalent() { return callable_type::make(type(int64_id), {type(string_id), type(string_id)}); }
    };

  } // namespace ndt

} // namespace dynd
