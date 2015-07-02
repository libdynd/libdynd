//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/array_range.hpp>
#include <dynd/types/tuple_type.hpp>
#include <map>
#include <dynd/func/call_callable.hpp>

#ifdef DYND_CUDA
#include <cufft.h>
#endif

#ifdef DYND_FFTW
#include <fftw3.h>

namespace dynd {

namespace nd {

  namespace detail {

    template <typename dst_type, typename src_type>
    struct fftw_plan {
      typedef ::fftw_plan type;
    };

    template <>
    struct fftw_plan<fftw_complex, fftw_complex> {
      typedef ::fftw_plan type;
    };

    template <>
    struct fftw_plan<double, fftw_complex> {
      typedef ::fftw_plan type;
    };

    template <>
    struct fftw_plan<fftw_complex, double> {
      typedef ::fftw_plan type;
    };

    template <>
    struct fftw_plan<fftwf_complex, fftwf_complex> {
      typedef ::fftwf_plan type;
    };

    fftwf_plan fftw_plan_guru_dft(int rank, const fftw_iodim *dims,
                                  int howmany_rank,
                                  const fftw_iodim *howmany_dims,
                                  fftwf_complex *in, fftwf_complex *out,
                                  int sign, unsigned flags);

    ::fftw_plan fftw_plan_guru_dft(int rank, const fftw_iodim *dims,
                                   int howmany_rank,
                                   const fftw_iodim *howmany_dims,
                                   fftw_complex *in, fftw_complex *out,
                                   int sign, unsigned flags);

    inline ::fftw_plan fftw_plan_guru_dft(int rank, const fftw_iodim *dims,
                                          int howmany_rank,
                                          const fftw_iodim *howmany_dims,
                                          double *in, fftw_complex *out,
                                          int sign, unsigned flags)
    {
      if (sign != 0) {
      }

      return ::fftw_plan_guru_dft_r2c(rank, dims, howmany_rank, howmany_dims,
                                      in, out, flags);
    }

    inline ::fftw_plan fftw_plan_guru_dft(int rank, const fftw_iodim *dims,
                                          int howmany_rank,
                                          const fftw_iodim *howmany_dims,
                                          fftw_complex *in, double *out,
                                          int sign, unsigned flags)
    {
      if (sign != 0) {
      }

      return ::fftw_plan_guru_dft_c2r(rank, dims, howmany_rank, howmany_dims,
                                      in, out, flags);
    }

    inline void fftw_execute_dft(const ::fftwf_plan plan, ::fftwf_complex *in,
                                 ::fftwf_complex *out)
    {
      ::fftwf_execute_dft(plan, in, out);
    }

    inline void fftw_execute_dft(const ::fftw_plan plan, ::fftw_complex *in,
                                 ::fftw_complex *out)
    {
      ::fftw_execute_dft(plan, in, out);
    }

    inline void fftw_execute_dft(const ::fftw_plan plan, double *in,
                                 ::fftw_complex *out)
    {
      ::fftw_execute_dft_r2c(plan, in, out);
    }

    inline void fftw_execute_dft(const ::fftw_plan plan, ::fftw_complex *in,
                                 double *out)
    {
      ::fftw_execute_dft_c2r(plan, in, out);
    }

    inline void fftw_destroy_plan(::fftwf_plan plan)
    {
      ::fftwf_destroy_plan(plan);
    }

    inline void fftw_destroy_plan(::fftw_plan plan)
    {
      ::fftw_destroy_plan(plan);
    }
  }

  template <typename T>
  struct is_double_precision {
    static const bool value = std::is_same<T, double>::value;
  };

  template <typename T, int N>
  struct is_double_precision<T[N]> {
    static const bool value = is_double_precision<T>::value;
  };

  template <typename fftw_dst_type, typename fftw_src_type, int sign = 0>
  struct fftw_ck : base_kernel<fftw_ck<fftw_dst_type, fftw_src_type, sign>,
                               kernel_request_host, 1> {
    typedef typename std::conditional<
        std::is_same<fftw_dst_type, fftw_complex>::value, complex<double>,
        typename std::conditional<
            std::is_same<fftw_dst_type, fftwf_complex>::value, complex<float>,
            fftw_dst_type>::type>::type dst_type;
    typedef typename std::conditional<
        std::is_same<fftw_src_type, fftw_complex>::value, complex<double>,
        typename std::conditional<
            std::is_same<fftw_src_type, fftwf_complex>::value, complex<float>,
            fftw_src_type>::type>::type src_type;

    typedef typename detail::fftw_plan<fftw_dst_type, fftw_src_type>::type
        plan_type;
    typedef fftw_ck self_type;

    plan_type plan;

    fftw_ck(const plan_type &plan) : plan(plan) {}

    ~fftw_ck() { detail::fftw_destroy_plan(plan); }

    void single(char *dst, char *const *src)
    {
      detail::fftw_execute_dft(plan,
                               *reinterpret_cast<fftw_src_type *const *>(src),
                               reinterpret_cast<fftw_dst_type *>(dst));
    }

    static ndt::type make_type()
    {
      return ndt::type("(Fixed**N * complex[float64], shape: ?N * int64, axes: "
                       "?Fixed * int64, flags: ?int32) -> Fixed**N * "
                       "complex[float64]");
    }

    static void
    data_init(const arrfunc_type_data *DYND_UNUSED(self),
              const ndt::arrfunc_type *DYND_UNUSED(self_tp),
              const char *DYND_UNUSED(static_data),
              size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
              intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
              nd::array &kwds,
              const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      if (kwds.p("flags").is_missing()) {
        kwds.p("flags").vals() = FFTW_ESTIMATE;
      }
    }

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const ndt::arrfunc_type *DYND_UNUSED(self_tp),
        const char *DYND_UNUSED(static_data),
        size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data), void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
        const char *const *src_arrmeta, kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      nd::array shape = kwds.p("shape");
      if (!shape.is_missing()) {
        if (shape.get_type().get_type_id() == pointer_type_id) {
          shape = shape.f("dereference");
        }
      }

      nd::array axes;
      if (!kwds.p("axes").is_missing()) {
        axes = kwds.p("axes");
        if (axes.get_type().get_type_id() == pointer_type_id) {
          axes = axes.f("dereference");
        }
      } else {
        axes = nd::range(src_tp[0].get_ndim());
      }

      const size_stride_t *src_size_stride =
          reinterpret_cast<const size_stride_t *>(src_arrmeta[0]);
      const size_stride_t *dst_size_stride =
          reinterpret_cast<const size_stride_t *>(dst_arrmeta);

      int rank = axes.get_dim_size();
      shortvector<fftw_iodim> dims(rank);
      for (intptr_t i = 0; i < rank; ++i) {
        intptr_t j = axes(i).as<intptr_t>();
        dims[i].n = shape.is_missing() ? src_size_stride[j].dim_size
                                       : shape(j).as<intptr_t>();
        dims[i].is = src_size_stride[j].stride / sizeof(fftw_src_type);
        dims[i].os = dst_size_stride[j].stride / sizeof(fftw_dst_type);
      }

      int howmany_rank = src_tp[0].get_ndim() - rank;
      shortvector<fftw_iodim> howmany_dims(howmany_rank);
      for (intptr_t i = 0, j = 0, k = 0; i < howmany_rank; ++i, ++j) {
        for (; k < rank && j == axes(k).as<intptr_t>(); ++j, ++k) {
        }
        howmany_dims[i].n = shape.is_missing() ? src_size_stride[j].dim_size
                                               : shape(j).as<intptr_t>();
        howmany_dims[i].is = src_size_stride[j].stride / sizeof(fftw_src_type);
        howmany_dims[i].os = dst_size_stride[j].stride / sizeof(fftw_dst_type);
      }

      nd::array src = nd::empty(src_tp[0]);
      nd::array dst = nd::empty(dst_tp);

      fftw_ck::make(
          ckb, kernreq, ckb_offset,
          detail::fftw_plan_guru_dft(
              rank, dims.get(), howmany_rank, howmany_dims.get(),
              reinterpret_cast<fftw_src_type *>(src.get_readwrite_originptr()),
              reinterpret_cast<fftw_dst_type *>(dst.get_readwrite_originptr()),
              sign, kwds.p("flags").as<int>()));

      return ckb_offset;
    }

    template <bool real_to_complex>
    static typename std::enable_if<real_to_complex, void>::type
    resolve_dst_type(
        const arrfunc_type_data *DYND_UNUSED(self),
        const ndt::arrfunc_type *DYND_UNUSED(self_tp),
        const char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      nd::array shape;
      try {
        shape = kwds.p("shape");
      }
      catch (...) {
      }

      intptr_t ndim = src_tp[0].get_ndim();
      dimvector src_shape(ndim);
      src_tp[0].extended()->get_shape(ndim, 0, src_shape.get(), NULL, NULL);
      src_shape[ndim - 1] = (shape.is_null() ? src_shape[ndim - 1]
                                             : shape(ndim - 1).as<intptr_t>()) /
                                2 +
                            1;
      dst_tp = ndt::make_fixed_dim(ndim, src_shape.get(),
                                   ndt::make_type<complex<double>>());
    }

    template <bool real_to_complex>
    static typename std::enable_if<!real_to_complex, void>::type
    resolve_dst_type(
        const arrfunc_type_data *DYND_UNUSED(self),
        const ndt::arrfunc_type *DYND_UNUSED(self_tp),
        const char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      nd::array shape = kwds.p("shape");
      if (shape.is_missing()) {
        dst_tp = src_tp[0];
      } else {
        if (shape.get_type().get_type_id() == pointer_type_id) {
          shape = shape.f("dereference");
        }
        dst_tp = ndt::make_fixed_dim(
            shape.get_dim_size(),
            reinterpret_cast<const intptr_t *>(shape.get_readonly_originptr()),
            ndt::make_type<complex<double>>());
      }
    }

    static void resolve_dst_type(
        const arrfunc_type_data *self, const ndt::arrfunc_type *self_tp,
        const char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t nsrc,
        const ndt::type *src_tp, const nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      resolve_dst_type<std::is_same<fftw_src_type, double>::value>(
          self, self_tp, NULL, 0, NULL, dst_tp, nsrc, src_tp, kwds, tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd

#endif