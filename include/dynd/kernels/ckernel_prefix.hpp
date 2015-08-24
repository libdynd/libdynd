//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dynd/config.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/typed_data_assign.hpp>

namespace dynd {

struct ckernel_prefix;

typedef void (*expr_single_t)(ckernel_prefix *self, char *dst,
                              char *const *src);
typedef void (*expr_strided_t)(ckernel_prefix *self, char *dst,
                               intptr_t dst_stride, char *const *src,
                               const intptr_t *src_stride, size_t count);

/**
 * Definition for kernel request parameters.
 */
enum {
  /** Kernel function in host memory */
  kernel_request_host = 0x00000000,
  /** Kernel function in CUDA device memory */
  kernel_request_cuda_device = 0x00000001,
/** Kernel function in both host memory and CUDA device memory */
#ifdef DYND_CUDA
  kernel_request_cuda_host_device = 0x00000002,
#else
  kernel_request_cuda_host_device = kernel_request_host,
#endif

  /** Kernel function expr_single_t, "(T1, T2, ...) -> R" */
  kernel_request_single = 0x00000008,
  /** Kernel function expr_strided_t, "(T1, T2, ...) -> R" */
  kernel_request_strided = 0x00000010,

  /** ... */
  kernel_request_memory = 0x00000007,
};
typedef uint32_t kernel_request_t;

inline kernel_request_t
kernel_request_without_function(kernel_request_t kernreq)
{
  return kernreq & 0x00000007;
}

/*
kernel_request_t without_memory(kernel_request_t kernreq) {

}
*/

/**
 * This is the struct which begins the memory layout
 * of all ckernels. First comes the function pointer,
 * which has a context-specific prototype, such as
 * `expr_single_t`, and then comes the
 * destructor.
 *
 * The ckernel is defined in terms of a C ABI definition,
 * and must satisfy alignments, movability, and
 * concurrency requirements to be valid. See the
 * document
 */
struct ckernel_prefix {
  typedef void (*destructor_fn_t)(ckernel_prefix *);

  void *function;
  destructor_fn_t destructor;

  /**
   * Aligns an offset as required by ckernels.
   */
  DYND_CUDA_HOST_DEVICE static size_t align_offset(size_t offset)
  {
    return (offset + size_t(7)) & ~size_t(7);
  }

  /**
   * Call to get the kernel function pointer, whose type
   * must be known by the context.
   *
   *      kdp->get_function<expr_single_t>()
   */
  template <typename T>
  DYND_CUDA_HOST_DEVICE T get_function() const
  {
    return reinterpret_cast<T>(function);
  }

  template <typename T>
  DYND_CUDA_HOST_DEVICE void set_function(T fnptr)
  {
    function = reinterpret_cast<void *>(fnptr);
  }

  DYND_CUDA_HOST_DEVICE void set_expr_function(kernel_request_t kernreq,
                                               expr_single_t single,
                                               expr_strided_t strided)
  {
    if (kernreq == kernel_request_single) {
      function = reinterpret_cast<void *>(single);
    } else if (kernreq == kernel_request_strided) {
      function = reinterpret_cast<void *>(strided);
    } else {
      DYND_HOST_THROW(std::runtime_error, "unrecognized dynd kernel request " +
                                              std::to_string(kernreq));
    }
  }

  template <class T>
  void set_expr_function(kernel_request_t kernreq)
  {
    set_expr_function(kernreq, &T::single, &T::strided);
  }

  /**
   * Calls the destructor of the ckernel if it is
   * non-NULL.
   */
  DYND_CUDA_HOST_DEVICE void destroy()
  {
    if (destructor != NULL) {
      destructor(this);
    }
  }

  /**
   * Returns the pointer to a child ckernel at the provided
   * offset.
   */
  DYND_CUDA_HOST_DEVICE ckernel_prefix *get_child_ckernel(intptr_t offset)
  {
    return reinterpret_cast<ckernel_prefix *>(
        reinterpret_cast<char *>(this) + ckernel_prefix::align_offset(offset));
  }

  static ckernel_prefix *init(ckernel_prefix *self,
                              kernel_request_t DYND_UNUSED(kernreq), void *func)
  {
    self->function = func;
    self->destructor = NULL;
    return self;
  }

  template <typename CKBT>
  static ckernel_prefix *make(CKBT *ckb, kernel_request_t kernreq,
                              intptr_t &inout_ckb_offset, void *func);

  static intptr_t
  instantiate(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
              char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
              const ndt::type &DYND_UNUSED(dst_tp),
              const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
              const ndt::type *DYND_UNUSED(src_tp),
              const char *const *DYND_UNUSED(src_arrmeta),
              kernel_request_t kernreq,
              const eval::eval_context *DYND_UNUSED(ectx),
              intptr_t DYND_UNUSED(nkwds), const nd::array *DYND_UNUSED(kwds),
              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars));
};

template <template <kernel_request_t, typename...> class F,
          kernel_request_t kernreq, typename T, bool flatten = false>
struct ex_for_each;

template <template <kernel_request_t, typename...> class F,
          kernel_request_t kernreq, typename T0>
struct ex_for_each<F, kernreq, type_sequence<T0>, false> {
  typedef type_sequence<F<kernreq, T0>> type;
};

template <template <kernel_request_t, typename...> class F,
          kernel_request_t kernreq, typename T0, typename... T>
struct ex_for_each<F, kernreq, type_sequence<T0, T...>, false> {
  typedef type_sequence<F<kernreq, T0>, F<kernreq, T>...> type;
};

template <template <kernel_request_t, typename...> class F,
          kernel_request_t kernreq, typename... T0>
struct ex_for_each<F, kernreq, type_sequence<type_sequence<T0...>>, true> {
  typedef type_sequence<F<kernreq, T0...>> type;
};

template <template <kernel_request_t, typename...> class F,
          kernel_request_t kernreq, typename... T0, typename... T>
struct ex_for_each<F, kernreq, type_sequence<type_sequence<T0...>, T...>,
                   true> {
  typedef typename join<type_sequence<F<kernreq, T0...>>,
                        typename ex_for_each<F, kernreq, type_sequence<T...>,
                                             true>::type>::type type;
};

} // namespace dynd
