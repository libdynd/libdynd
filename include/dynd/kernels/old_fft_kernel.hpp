//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_strided_kernel.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/array_range.hpp>
#include <dynd/types/tuple_type.hpp>
#include <map>

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

    fftwf_plan fftw_plan_guru_dft(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims,
                                  fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags);

    ::fftw_plan fftw_plan_guru_dft(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims,
                                   fftw_complex *in, fftw_complex *out, int sign, unsigned flags);

    inline ::fftw_plan fftw_plan_guru_dft(int rank, const fftw_iodim *dims, int howmany_rank,
                                          const fftw_iodim *howmany_dims, double *in, fftw_complex *out, int sign,
                                          unsigned flags)
    {
      if (sign != 0) {
      }

      return ::fftw_plan_guru_dft_r2c(rank, dims, howmany_rank, howmany_dims, in, out, flags);
    }

    inline ::fftw_plan fftw_plan_guru_dft(int rank, const fftw_iodim *dims, int howmany_rank,
                                          const fftw_iodim *howmany_dims, fftw_complex *in, double *out, int sign,
                                          unsigned flags)
    {
      if (sign != 0) {
      }

      return ::fftw_plan_guru_dft_c2r(rank, dims, howmany_rank, howmany_dims, in, out, flags);
    }

    inline void fftw_execute_dft(const ::fftwf_plan plan, ::fftwf_complex *in, ::fftwf_complex *out)
    {
      ::fftwf_execute_dft(plan, in, out);
    }

    inline void fftw_execute_dft(const ::fftw_plan plan, ::fftw_complex *in, ::fftw_complex *out)
    {
      ::fftw_execute_dft(plan, in, out);
    }

    inline void fftw_execute_dft(const ::fftw_plan plan, double *in, ::fftw_complex *out)
    {
      ::fftw_execute_dft_r2c(plan, in, out);
    }

    inline void fftw_execute_dft(const ::fftw_plan plan, ::fftw_complex *in, double *out)
    {
      ::fftw_execute_dft_c2r(plan, in, out);
    }

    inline void fftw_destroy_plan(::fftwf_plan plan) { ::fftwf_destroy_plan(plan); }

    inline void fftw_destroy_plan(::fftw_plan plan) { ::fftw_destroy_plan(plan); }
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
  struct fftw_ck : base_strided_kernel<fftw_ck<fftw_dst_type, fftw_src_type, sign>, 1> {
    typedef typename std::conditional<std::is_same<fftw_dst_type, fftw_complex>::value, complex<double>,
                                      typename std::conditional<std::is_same<fftw_dst_type, fftwf_complex>::value,
                                                                complex<float>, fftw_dst_type>::type>::type dst_type;
    typedef typename std::conditional<std::is_same<fftw_src_type, fftw_complex>::value, complex<double>,
                                      typename std::conditional<std::is_same<fftw_src_type, fftwf_complex>::value,
                                                                complex<float>, fftw_src_type>::type>::type src_type;

    typedef typename detail::fftw_plan<fftw_dst_type, fftw_src_type>::type plan_type;
    typedef fftw_ck self_type;

    plan_type plan;

    fftw_ck(const plan_type &plan) : plan(plan) {}

    ~fftw_ck() { detail::fftw_destroy_plan(plan); }

    void single(char *dst, char *const *src)
    {
      detail::fftw_execute_dft(plan, *reinterpret_cast<fftw_src_type *const *>(src),
                               reinterpret_cast<fftw_dst_type *>(dst));
    }
  };

} // namespace dynd::nd
} // namespace dynd

#endif
