//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dynd/config.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/kernel_builder.hpp>

namespace dynd {

typedef void (*kernel_call_t)(nd::kernel_prefix *self, const nd::array *dst, const nd::array *src);
typedef void (*kernel_single_t)(nd::kernel_prefix *self, char *dst, char *const *src);
typedef void (*kernel_strided_t)(nd::kernel_prefix *self, char *dst, intptr_t dst_stride, char *const *src,
                                 const intptr_t *src_stride, size_t count);

/**
 * Definition for kernel request parameters.
 */
enum {
  /** Kernel function in host memory */
  kernel_request_host = 0x00000000,

  kernel_request_call = 0x00000000,
  kernel_request_single = 0x00000001,
  kernel_request_strided = 0x00000003,

  kernel_request_data_only = 0x00000001
};
typedef uint32_t kernel_request_t;

namespace nd {

  /**
   * This is the struct which begins the memory layout
   * of all ckernels. First comes the function pointer,
   * which has a context-specific prototype, such as
   * `kernel_single_t`, and then comes the
   * destructor.
   *
   * The ckernel is defined in terms of a C ABI definition,
   * and must satisfy alignments, movability, and
   * concurrency requirements to be valid. See the
   * document
   */
  struct DYND_API kernel_prefix {
    typedef void (*destructor_fn_t)(kernel_prefix *);

    void (*destructor)(kernel_prefix *self);
    void *function;

    /**
     * Call to get the kernel function pointer, whose type
     * must be known by the context.
     *
     *      kdp->get_function<kernel_single_t>()
     */
    template <typename T>
    T get_function() const
    {
      return reinterpret_cast<T>(function);
    }

    /**
     * Calls the destructor of the ckernel if it is
     * non-NULL.
     */
    void destroy()
    {
      if (destructor != NULL) {
        destructor(this);
      }
    }

    void single(char *dst, char *const *src) { (*reinterpret_cast<kernel_single_t>(function))(this, dst, src); }

    void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
    {
      (*reinterpret_cast<kernel_strided_t>(function))(this, dst, dst_stride, src, src_stride, count);
    }

    /**
     * Returns the pointer to a child ckernel at the provided
     * offset.
     */
    kernel_prefix *get_child(intptr_t offset)
    {
      return reinterpret_cast<kernel_prefix *>(reinterpret_cast<char *>(this) + kernel_builder::aligned_size(offset));
    }

    /**
     * Returns a pointer to the list of child offsets.
     */
    size_t *get_offsets() { return reinterpret_cast<size_t *>(this + 1); }

    static kernel_prefix *init(kernel_prefix *self, void *func)
    {
      self->function = func;
      self->destructor = NULL;
      return self;
    }
  };

} // namespace dynd::nd
} // namespace dynd
