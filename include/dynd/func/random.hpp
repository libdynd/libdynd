#pragma once

#include <chrono>
#include <memory>
#include <random>

#include <dynd/func/callable.hpp>

namespace dynd {
namespace nd {
  namespace random {

    extern DYND_API struct uniform : declfunc<uniform> {
      static DYND_API callable children[DYND_TYPE_ID_MAX + 1];

      static DYND_API callable make();
    } uniform;

  } // namespace dynd::nd::random

  inline array rand(const ndt::type &tp)
  {
    return random::uniform(kwds("dst_tp", tp));
  }

  inline array rand(intptr_t dim0, const ndt::type &tp)
  {
    return rand(ndt::make_fixed_dim(dim0, tp));
  }

  inline array rand(intptr_t dim0, intptr_t dim1, const ndt::type &tp)
  {
    return rand(ndt::make_fixed_dim(dim0, ndt::make_fixed_dim(dim1, tp)));
  }

  inline array rand(intptr_t dim0, intptr_t dim1, intptr_t dim2,
                    const ndt::type &tp)
  {
    return rand(ndt::make_fixed_dim(
        dim0, ndt::make_fixed_dim(dim1, ndt::make_fixed_dim(dim2, tp))));
  }

} // namespace dynd::nd
} // namespace dynd

/*

#ifdef DYND_CUDA
      template <kernel_request_t kernreq>
      static typename std::enable_if<kernreq == kernel_request_cuda_device,
                                     callable>::type
      make();
#endif

#ifdef __CUDACC__

  template <typename S>
  struct uniform_complex_ck<kernel_request_cuda_device, S, complex<double>>
      : base_kernel<
            uniform_complex_ck<kernel_request_cuda_device, S, complex<double>>,
            kernel_request_cuda_device, 0> {
    typedef uniform_complex_ck self_type;

    S *s;

    __device__ uniform_complex_ck(S *s) : s(s) {}

    __device__ void single(char *dst, char *const *DYND_UNUSED(src))
    {
      *reinterpret_cast<complex<double> *>(dst) =
          complex<double>(curand_uniform_double(s), curand_uniform_double(s));
    }

    static ndt::type make_type()
    {
      typedef dynd::complex<double> R;

      std::map<nd::string, ndt::type> tp_vars;
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(a: ?R, b: ?R) -> cuda_device[R]"),
                             tp_vars, true);
    }

    static intptr_t instantiate(
        const callable_type_data *self,
        const ndt::callable_type *DYND_UNUSED(self_tp),
        const char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      if ((kernreq & kernel_request_memory) == kernel_request_host) {
        typedef cuda_launch_ck<0> self_type;
        self_type *self = self_type::make(ckb, kernreq, ckb_offset, 1, 1);
        ckb = &self->ckb;
        kernreq |= kernel_request_cuda_device;
        ckb_offset = 0;
      }

      self_type::make(ckb, kernreq, ckb_offset, *self->get_data_as<S *>());
      return ckb_offset;
    }
  };

#endif


#ifdef __CUDACC__

  template <typename S>
  struct uniform_real_ck<kernel_request_cuda_device, S, double>
      : base_kernel<uniform_real_ck<kernel_request_cuda_device, S, double>,
                    kernel_request_cuda_device, 0> {
    typedef uniform_real_ck self_type;

    S *s;

    __device__ uniform_real_ck(S *s) : s(s) {}

    __device__ void single(char *dst, char *const *DYND_UNUSED(src))
    {
      *reinterpret_cast<double *>(dst) = curand_uniform_double(s);
    }

    __device__ void strided(char *dst, intptr_t dst_stride,
                            char *const *DYND_UNUSED(src),
                            const intptr_t *DYND_UNUSED(src_stride),
                            size_t count)
    {
      dst += DYND_THREAD_ID(0) * dst_stride;

      for (size_t i = DYND_THREAD_ID(0); i < count; i += DYND_THREAD_COUNT(0)) {
        *reinterpret_cast<double *>(dst) =
            curand_uniform_double(s + DYND_THREAD_ID(0));
        dst += DYND_THREAD_COUNT(0) * dst_stride;
      }
    }

    static ndt::type make_type()
    {
      typedef double R;

      std::map<nd::string, ndt::type> tp_vars;
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(a: ?R, b: ?R) -> cuda_device[R]"),
                             tp_vars, true);
    }

    static intptr_t instantiate(
        const callable_type_data *self,
        const ndt::callable_type *DYND_UNUSED(self_tp),
        const char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      if ((kernreq & kernel_request_memory) == kernel_request_host) {
        typedef cuda_launch_ck<0> self_type;
        self_type *self = self_type::make(ckb, kernreq, ckb_offset, 1, 1);
        ckb = &self->ckb;
        kernreq |= kernel_request_cuda_device;
        ckb_offset = 0;
      }

      self_type::make(ckb, kernreq, ckb_offset, *self->get_data_as<S *>());
      return ckb_offset;
    }
  };

#endif

*/
