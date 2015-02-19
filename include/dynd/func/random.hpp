#pragma once

#include <chrono>
#include <memory>
#include <random>

#ifdef DYND_CUDA
#include <curand_kernel.h>
#endif

#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/kernels/cuda_launch.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {

typedef type_sequence<int32_t, int64_t, uint32_t, uint64_t> integral_types;
typedef type_sequence<float, double> real_types;
typedef type_sequence<complex<float>, complex<double>> complex_types;

typedef join<typename join<integral_types, real_types>::type,
             complex_types>::type numeric_types;

namespace kernels {

  template <kernel_request_t kernreq, typename... T>
  struct uniform_int_ck;

  template <typename G, typename R>
  struct uniform_int_ck<kernel_request_host, G, R>
      : expr_ck<uniform_int_ck<kernel_request_host, G, R>, kernel_request_host,
                0> {
    typedef uniform_int_ck self_type;

    G &g;
    std::uniform_int_distribution<R> d;

    uniform_int_ck(G *g, R a, R b) : g(*g), d(a, b) {}

    void single(char *dst, char *const *DYND_UNUSED(src))
    {
      *reinterpret_cast<R *>(dst) = d(g);
    }

    static ndt::type make_type()
    {
      std::map<nd::string, ndt::type> tp_vars;
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(a: ?R, b: ?R) -> R"), tp_vars, true);
    }

    static void resolve_option_values(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp), nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      nd::array a = kwds.p("a");
      if (a.is_missing()) {
        a.val_assign(0);
      }

      nd::array b = kwds.p("b");
      if (b.is_missing()) {
        b.val_assign(std::numeric_limits<R>::max());
      }
    }

    static intptr_t instantiate(
        const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      std::shared_ptr<G> g = *self->get_data_as<std::shared_ptr<G>>();
      R a = kwds.p("a").as<R>();
      R b = kwds.p("b").as<R>();

      self_type::create(ckb, kernreq, ckb_offset, g.get(), a, b);
      return ckb_offset;
    }
  };

  template <kernel_request_t kernreq, typename... T>
  struct uniform_real_ck;

  template <typename G, typename R>
  struct uniform_real_ck<kernel_request_host, G, R>
      : expr_ck<uniform_real_ck<kernel_request_host, G, R>, kernel_request_host,
                0> {
    typedef uniform_real_ck self_type;

    G &g;
    std::uniform_real_distribution<R> d;

    uniform_real_ck(G *g, R a, R b) : g(*g), d(a, b) {}

    void single(char *dst, char *const *DYND_UNUSED(src))
    {
      *reinterpret_cast<R *>(dst) = d(g);
    }

    static ndt::type make_type()
    {
      std::map<nd::string, ndt::type> tp_vars;
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(a: ?R, b: ?R) -> R"), tp_vars, true);
    }

    static void resolve_option_values(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp), nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      nd::array a = kwds.p("a");
      if (a.is_missing()) {
        a.val_assign(0);
      }

      nd::array b = kwds.p("b");
      if (b.is_missing()) {
        b.val_assign(1);
      }
    }

    static intptr_t instantiate(
        const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      std::shared_ptr<G> g = *self->get_data_as<std::shared_ptr<G>>();
      R a = kwds.p("a").as<R>();
      R b = kwds.p("b").as<R>();

      self_type::create(ckb, kernreq, ckb_offset, g.get(), a, b);
      return ckb_offset;
    }
  };

#ifdef __CUDACC__

  template <typename S>
  struct uniform_real_ck<kernel_request_cuda_device, S, double>
      : expr_ck<uniform_real_ck<kernel_request_cuda_device, S, double>,
                kernel_request_cuda_device, 0> {
    typedef uniform_real_ck self_type;

    S *s;

    __device__ uniform_real_ck(S *s) : s(s) {}

    __device__ void single(char *dst, char *const *DYND_UNUSED(src))
    {
      *reinterpret_cast<double *>(dst) = curand_uniform_double(s);
    }

    __device__ void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                            const intptr_t *DYND_UNUSED(src_stride), size_t count)
    {
      dst += DYND_THREAD_ID(0) * dst_stride;

      for (size_t i = DYND_THREAD_ID(0); i < count; i += DYND_THREAD_COUNT(0)) {
        *reinterpret_cast<double *>(dst) = curand_uniform_double(s + DYND_THREAD_ID(0));
        dst += DYND_THREAD_COUNT(0) * dst_stride;
      }
    }

    static ndt::type make_type()
    {
      typedef double R;

      std::map<nd::string, ndt::type> tp_vars;
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(a: ?R, b: ?R) -> cuda_device[R]"), tp_vars, true);
    }

    static intptr_t instantiate(
        const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      if ((kernreq & kernel_request_memory) == kernel_request_host) {
        typedef cuda_launch_ck<0> self_type;
        self_type *self = self_type::create(ckb, kernreq, ckb_offset, 1, 1);
        ckb = &self->ckb;
        kernreq |= kernel_request_cuda_device;
        ckb_offset = 0;
      }

      self_type::create(ckb, kernreq, ckb_offset,
                        *self->get_data_as<S *>());
      return ckb_offset;
    }
  };

#endif

  template <kernel_request_t kernreq, typename... T>
  struct uniform_complex_ck;

  template <typename G, typename R>
  struct uniform_complex_ck<kernel_request_host, G, R>
      : expr_ck<uniform_complex_ck<kernel_request_host, G, R>,
                kernel_request_host, 0> {
    typedef uniform_complex_ck self_type;

    G &g;
    std::uniform_real_distribution<typename R::value_type> d_real;
    std::uniform_real_distribution<typename R::value_type> d_imag;

    uniform_complex_ck(G *g, R a, R b)
        : g(*g), d_real(a.real(), b.real()), d_imag(a.imag(), b.imag())
    {
    }

    void single(char *dst, char *const *DYND_UNUSED(src))
    {
      *reinterpret_cast<R *>(dst) = R(d_real(g), d_imag(g));
    }

    static ndt::type make_type()
    {
      std::map<nd::string, ndt::type> tp_vars;
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(a: ?R, b: ?R) -> R"), tp_vars, true);
    }

    static void resolve_option_values(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp), nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      nd::array a = kwds.p("a");
      if (a.is_missing()) {
        a.val_assign(R(0, 0));
      }

      nd::array b = kwds.p("b");
      if (b.is_missing()) {
        b.val_assign(R(1, 1));
      }
    }

    static intptr_t instantiate(
        const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      std::shared_ptr<G> g = *self->get_data_as<std::shared_ptr<G>>();
      R a = kwds.p("a").as<R>();
      R b = kwds.p("b").as<R>();

      self_type::create(ckb, kernreq, ckb_offset, g.get(), a, b);
      return ckb_offset;
    }
  };

#ifdef __CUDACC__

  template <typename S>
  struct uniform_complex_ck<kernel_request_cuda_device, S, complex<double>>
      : expr_ck<uniform_complex_ck<kernel_request_cuda_device, S, complex<double>>,
                kernel_request_cuda_device, 0> {
    typedef uniform_complex_ck self_type;

    S *s;

    __device__ uniform_complex_ck(S *s) : s(s) {}

    __device__ void single(char *dst, char *const *DYND_UNUSED(src))
    {
      *reinterpret_cast<complex<double> *>(dst) = complex<double>(curand_uniform_double(s), curand_uniform_double(s));
    }

    static ndt::type make_type()
    {
      typedef dynd::complex<double> R;

      std::map<nd::string, ndt::type> tp_vars;
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(a: ?R, b: ?R) -> cuda_device[R]"), tp_vars, true);
    }

    static intptr_t instantiate(
        const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      if ((kernreq & kernel_request_memory) == kernel_request_host) {
        typedef cuda_launch_ck<0> self_type;
        self_type *self = self_type::create(ckb, kernreq, ckb_offset, 1, 1);
        ckb = &self->ckb;
        kernreq |= kernel_request_cuda_device;
        ckb_offset = 0;
      }

      self_type::create(ckb, kernreq, ckb_offset,
                        *self->get_data_as<S *>());
      return ckb_offset;
    }
  };

#endif

  template <kernel_request_t kernreq, typename... T>
  struct uniform_ck;

  template <kernel_request_t kernreq, typename G>
  struct uniform_ck<kernreq, G, int8_t> : uniform_int_ck<kernreq, G, int8_t> {
  };

  template <kernel_request_t kernreq, typename G>
  struct uniform_ck<kernreq, G, int16_t> : uniform_int_ck<kernreq, G, int16_t> {
  };

  template <kernel_request_t kernreq, typename G>
  struct uniform_ck<kernreq, G, int32_t> : uniform_int_ck<kernreq, G, int32_t> {
  };

  template <kernel_request_t kernreq, typename G>
  struct uniform_ck<kernreq, G, int64_t> : uniform_int_ck<kernreq, G, int64_t> {
  };

  template <kernel_request_t kernreq, typename G>
  struct uniform_ck<kernreq, G, uint8_t> : uniform_int_ck<kernreq, G, uint8_t> {
  };

  template <kernel_request_t kernreq, typename G>
  struct uniform_ck<kernreq, G, uint16_t>
      : uniform_int_ck<kernreq, G, uint16_t> {
  };

  template <kernel_request_t kernreq, typename G>
  struct uniform_ck<kernreq, G, uint32_t>
      : uniform_int_ck<kernreq, G, uint32_t> {
  };

  template <kernel_request_t kernreq, typename G>
  struct uniform_ck<kernreq, G, uint64_t>
      : uniform_int_ck<kernreq, G, uint64_t> {
  };

  template <kernel_request_t kernreq, typename G>
  struct uniform_ck<kernreq, G, float> : uniform_real_ck<kernreq, G, float> {
  };

  template <kernel_request_t kernreq, typename G>
  struct uniform_ck<kernreq, G, double> : uniform_real_ck<kernreq, G, double> {
  };

  template <kernel_request_t kernreq, typename G>
  struct uniform_ck<kernreq, G, complex<float>>
      : uniform_complex_ck<kernreq, G, complex<float>> {
  };

  template <kernel_request_t kernreq, typename G>
  struct uniform_ck<kernreq, G, complex<double>>
      : uniform_complex_ck<kernreq, G, complex<double>> {
  };

} // namespace dynd::kernels

namespace nd {
  namespace decl {
    namespace random {

      struct uniform : arrfunc<uniform> {
        template <kernel_request_t kernreq>
        static typename std::enable_if<kernreq == kernel_request_host,
                                       nd::arrfunc>::type
        as_arrfunc();

#ifdef DYND_CUDA
        template <kernel_request_t kernreq>
        static typename std::enable_if<kernreq == kernel_request_cuda_device,
                                       nd::arrfunc>::type
        as_arrfunc();
#endif

        static nd::arrfunc as_arrfunc();
      };

    } // namespace dynd::nd::decl::random
  }   // namespace dynd::nd::decl

  namespace random {

    extern decl::random::uniform uniform;

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