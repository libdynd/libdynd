//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/kernel_prefix.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    struct DYND_API reduction_kernel_prefix : kernel_prefix {
      // This function pointer is for all the calls of the function
      // on a given destination data address after the "first call".
      kernel_strided_t followup_call_function;

      template <typename T>
      T get_first_call_function() const
      {
        return get_function<T>();
      }

      template <typename T>
      void set_first_call_function(T fnptr)
      {
        function = reinterpret_cast<void *>(fnptr);
      }

      kernel_strided_t get_followup_call_function() const { return followup_call_function; }

      void set_followup_call_function(kernel_strided_t fnptr) { followup_call_function = fnptr; }

      void single_first(char *dst, char *const *src) { (*reinterpret_cast<kernel_single_t>(function))(this, dst, src); }

      void strided_first(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        (*reinterpret_cast<kernel_strided_t>(function))(this, dst, dst_stride, src, src_stride, count);
      }

      void strided_followup(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        (*reinterpret_cast<kernel_strided_t>(followup_call_function))(this, dst, dst_stride, src, src_stride, count);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
