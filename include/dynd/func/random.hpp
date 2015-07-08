#pragma once

#include <chrono>
#include <memory>
#include <random>

#ifdef DYND_CUDA
#include <curand_kernel.h>
#endif

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/cuda_launch.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {

typedef type_sequence<int32_t, int64_t, uint32_t, uint64_t> integral_types;
typedef type_sequence<float, double> real_types;
typedef type_sequence<complex<float>, complex<double>> complex_types;

typedef join<typename join<integral_types, real_types>::type,
             complex_types>::type numeric_types;

inline std::shared_ptr<std::default_random_engine> &get_random_device()
{
  static std::random_device random_device;
  static std::shared_ptr<std::default_random_engine> g(
      new std::default_random_engine(random_device()));

  return g;
}

namespace nd {

  template <typename... T>
  struct uniform_int_ck;

  template <typename G, typename R>
  struct uniform_int_ck<G, R>
      : base_kernel<uniform_int_ck<G, R>, kernel_request_host, 0> {
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

    /*
        static void
        data_init(const arrfunc_type_data *DYND_UNUSED(self),
                  const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                  const char *DYND_UNUSED(static_data),
                  size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
                  intptr_t DYND_UNUSED(nsrc), const ndt::type
       *DYND_UNUSED(src_tp),
                  nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type>
       &DYND_UNUSED(tp_vars))
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
    */

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const ndt::arrfunc_type *DYND_UNUSED(self_tp),
        const char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      std::shared_ptr<G> g = get_random_device();

      R a;
      if (kwds.p("a").is_missing()) {
        a = 0;
      } else {
        a = kwds.p("a").as<R>();
      }

      R b;
      if (kwds.p("b").is_missing()) {
        b = std::numeric_limits<R>::max();
      } else {
        b = kwds.p("b").as<R>();
      }

      self_type::make(ckb, kernreq, ckb_offset, g.get(), a, b);
      return ckb_offset;
    }
  };

  template <typename... T>
  struct uniform_real_ck;

  template <typename G, typename R>
  struct uniform_real_ck<G, R>
      : base_kernel<uniform_real_ck<G, R>, kernel_request_host, 0> {
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

    /*
        static void
        data_init(const arrfunc_type_data *DYND_UNUSED(self),
                  const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                  const char *DYND_UNUSED(static_data),
                  size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
                  intptr_t DYND_UNUSED(nsrc), const ndt::type
       *DYND_UNUSED(src_tp),
                  nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type>
       &DYND_UNUSED(tp_vars))
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
    */

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const ndt::arrfunc_type *DYND_UNUSED(self_tp),
        const char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      std::shared_ptr<G> g = get_random_device();

      R a;
      if (kwds.p("a").is_missing()) {
        a = 0;
      } else {
        a = kwds.p("a").as<R>();
      }

      R b;
      if (kwds.p("b").is_missing()) {
        b = 1;
      } else {
        b = kwds.p("b").as<R>();
      }

      self_type::make(ckb, kernreq, ckb_offset, g.get(), a, b);
      return ckb_offset;
    }
  };

  template <typename... T>
  struct uniform_complex_ck;

  template <typename G, typename R>
  struct uniform_complex_ck<G, R>
      : base_kernel<uniform_complex_ck<G, R>, kernel_request_host, 0> {
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

    /*
        static void
        data_init(const arrfunc_type_data *DYND_UNUSED(self),
                  const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                  const char *DYND_UNUSED(static_data),
                  size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
                  intptr_t DYND_UNUSED(nsrc), const ndt::type
       *DYND_UNUSED(src_tp),
                  nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type>
       &DYND_UNUSED(tp_vars))
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
    */

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const ndt::arrfunc_type *DYND_UNUSED(self_tp),
        const char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      std::shared_ptr<G> g = get_random_device();

      R a;
      if (kwds.p("a").is_missing()) {
        a = R(0, 0);
      } else {
        a = kwds.p("a").as<R>();
      }

      R b;
      if (kwds.p("b").is_missing()) {
        b = R(1, 1);
      } else {
        b = kwds.p("b").as<R>();
      }

      self_type::make(ckb, kernreq, ckb_offset, g.get(), a, b);
      return ckb_offset;
    }
  };

  template <typename G, type_id_t DstTypeID>
  struct uniform_ck;

  template <typename G>
  struct uniform_ck<G, int8_type_id> : uniform_int_ck<G, int8_t> {
  };

  template <typename G>
  struct uniform_ck<G, int16_type_id> : uniform_int_ck<G, int16_t> {
  };

  template <typename G>
  struct uniform_ck<G, int32_type_id> : uniform_int_ck<G, int32_t> {
  };

  template <typename G>
  struct uniform_ck<G, int64_type_id> : uniform_int_ck<G, int64_t> {
  };

  template <typename G>
  struct uniform_ck<G, uint8_type_id> : uniform_int_ck<G, uint8_t> {
  };

  template <typename G>
  struct uniform_ck<G, uint16_type_id> : uniform_int_ck<G, uint16_t> {
  };

  template <typename G>
  struct uniform_ck<G, uint32_type_id> : uniform_int_ck<G, uint32_t> {
  };

  template <typename G>
  struct uniform_ck<G, uint64_type_id> : uniform_int_ck<G, uint64_t> {
  };

  template <typename G>
  struct uniform_ck<G, float32_type_id> : uniform_real_ck<G, float> {
  };

  template <typename G>
  struct uniform_ck<G, float64_type_id> : uniform_real_ck<G, double> {
  };

  template <typename G>
  struct uniform_ck<G, complex_float32_type_id>
      : uniform_complex_ck<G, complex<float>> {
  };

  template <typename G>
  struct uniform_ck<G, complex_float64_type_id>
      : uniform_complex_ck<G, complex<double>> {
  };

  namespace random {

    extern struct uniform : declfunc<uniform> {
      template <kernel_request_t kernreq>
      static typename std::enable_if<kernreq == kernel_request_host,
                                     arrfunc>::type
      make();

      static nd::arrfunc make();
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
                                     arrfunc>::type
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
        const arrfunc_type_data *self,
        const ndt::arrfunc_type *DYND_UNUSED(self_tp),
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
        const arrfunc_type_data *self,
        const ndt::arrfunc_type *DYND_UNUSED(self_tp),
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