//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/fft_kernel.hpp>
#include <dynd/func/callable.hpp>

namespace dynd {
namespace nd {

#ifdef DYND_CUDA

  template <typename dst_type, typename src_type, int sign>
  struct cufft_ck
      : expr_ck<cufft_ck<dst_type, src_type, sign>, kernel_request_host, 1> {
    typedef cufft_ck self_type;

    cufftHandle plan;

    void single(char *dst, char *const *src)
    {
      cufftExecZ2Z(plan, reinterpret_cast<src_type *>(src[0]),
                   reinterpret_cast<dst_type *>(dst), sign);
    }

    static ndt::type make_type()
    {
      return ndt::type(
          "(cuda_device[Fixed**N * complex[float64]], shape: ?N * int64, axes: "
          "?Fixed * int64, flags: ?int32) -> cuda_device[Fixed**N * "
          "complex[float64]]");
    }

    static int resolve_dst_type(
        const callable_type_data *DYND_UNUSED(self),
        const callable_type *DYND_UNUSED(self_tp), char *DYND_UNUSED(data),
        intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
        int DYND_UNUSED(throw_on_error), ndt::type &dst_tp,
        const nd::array &kwds,
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      nd::array shape = kwds.p("shape");
      //      if (shape.is_missing()) {
      dst_tp = src_tp[0];
      //    }

      return 1;
    }

    static intptr_t instantiate(
        const callable_type_data *DYND_UNUSED(self),
        const callable_type *DYND_UNUSED(self_tp), void *ckb,
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        const nd::array &kwds,
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      const size_stride_t *dst_size_stride =
          reinterpret_cast<const size_stride_t *>(dst_arrmeta);
      const size_stride_t *src_size_stride =
          reinterpret_cast<const size_stride_t *>(src_arrmeta[0]);

      array axes = kwds.p("axes");
      array shape = kwds.p("shape");

      int ndim = static_cast<int>(src_tp[0].get_ndim());

      int rank = static_cast<int>(axes.is_missing() ? ndim : (ndim - 1));
      int istride =
          static_cast<int>(src_size_stride[ndim - 1].stride / sizeof(src_type));
      int idist =
          static_cast<int>(src_size_stride[0].stride / sizeof(src_type));
      int ostride =
          static_cast<int>(dst_size_stride[ndim - 1].stride / sizeof(dst_type));
      int odist =
          static_cast<int>(dst_size_stride[0].stride / sizeof(dst_type));

      std::vector<int> n(rank), inembed(rank), onembed(rank);
      for (int i = 0, j = axes.is_missing() ? 0 : 1; j < ndim; ++i, ++j) {
        n[i] = static_cast<int>(src_size_stride[j].dim_size);
        inembed[i] = static_cast<int>(src_size_stride[j].dim_size);
        onembed[i] = static_cast<int>(dst_size_stride[j].dim_size);
      }

      int batch =
          static_cast<int>(axes.is_missing() ? 1 : src_size_stride[0].dim_size);

      self_type *self = self_type::create(ckb, kernreq, ckb_offset);
      cufftPlanMany(&self->plan, rank, n.data(), inembed.data(), istride, idist,
                    onembed.data(), ostride, odist, CUFFT_Z2Z, batch);

      return ckb_offset;
    }
  };

#endif

  extern DYND_API struct fft : declfunc<fft> {
    static DYND_API callable make();
  } fft;

  extern DYND_API struct ifft : declfunc<ifft> {
    static DYND_API callable make();
  } ifft;

  extern DYND_API struct rfft : declfunc<rfft> {
    static DYND_API callable make();
  } rfft;

  extern DYND_API struct irfft : declfunc<irfft> {
    static DYND_API callable make();
  } irfft;

  /**
   * Shifts the zero-frequency element to the center of an array.
   */
  DYND_API array fftshift(const nd::array &x);

  /**
   * Inverts fftshift.
   */
  DYND_API array ifftshift(const nd::array &x);

  /**
   * Returns the sample frequencies of a discrete Fourier transform, with units
   * of cycles per seconds.
   */
  DYND_API array fftspace(intptr_t count, double step = 1.0);

} // namespace dynd::nd
} // namespace dynd
