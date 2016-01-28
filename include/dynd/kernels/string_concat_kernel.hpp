//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// Implement a string concatenation kernel

#pragma once

#include <dynd/callable.hpp>
#include <dynd/kernels/kernel_builder.hpp>
#include <dynd/types/string_type.hpp>

namespace dynd {
  namespace nd {

    namespace detail {
      inline void concat_one_string(size_t nop, dynd::string *d, const dynd::string *const *s)
      {
        // Get the size of the concatenated string
        size_t size = 0;
        for (size_t i = 0; i != nop; ++i) {
          size += (s[i]->end() - s[i]->begin());
        }

        // Allocate the output
        d->resize(size);
        // Copy the string data
        char *dst = d->begin();
        for (size_t i = 0; i != nop; ++i) {
          size_t op_size = (s[i]->end() - s[i]->begin());
          DYND_MEMCPY(dst, s[i]->begin(), op_size);
          dst += op_size;
        }
      }
    }

    struct string_concatenation_kernel
      : base_kernel<string_concatenation_kernel, 2> {

      void single(char *dst, char *const *src) {
        dynd::string *d = reinterpret_cast<dynd::string *>(dst);
        const dynd::string *const *s = reinterpret_cast<const dynd::string *const *>(src);

        detail::concat_one_string(2, d, s);
      }
    };

  } // namespace nd

  namespace ndt {

    template<>
    struct traits<dynd::nd::string_concatenation_kernel> {
      static type equivalent() { return callable_type::make(type(string_id), {type(string_id), type(string_id)}); }
    };

  } // namespace ndt

} // namespace dynd
