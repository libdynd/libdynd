//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/fft_kernel.hpp>

namespace dynd {
namespace nd {

#ifdef DYND_FFTW

  template <typename fftw_dst_type, typename fftw_src_type, int sign = 0>
  class fftw_callable : public base_callable {
  public:
    fftw_callable()
        : base_callable(ndt::type("(Fixed**N * complex[float64], shape: ?N * int64, axes: "
                                  "?Fixed * int64, flags: ?int32) -> Fixed**N * "
                                  "complex[float64]"))
    {
    }

    template <bool real_to_complex>
    typename std::enable_if<real_to_complex, void>::type
    resolve_dst_type_(char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      nd::array shape = kwds[0];

      intptr_t ndim = src_tp[0].get_ndim();
      dimvector src_shape(ndim);
      src_tp[0].extended()->get_shape(ndim, 0, src_shape.get(), NULL, NULL);
      src_shape[ndim - 1] = (shape.is_null() ? src_shape[ndim - 1] : shape(ndim - 1).as<intptr_t>()) / 2 + 1;
      dst_tp = ndt::make_fixed_dim(ndim, src_shape.get(), ndt::make_type<complex<double>>());
    }

    template <bool real_to_complex>
    typename std::enable_if<!real_to_complex, void>::type
    resolve_dst_type_(char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      nd::array shape = kwds[0];
      if (shape.is_na()) {
        dst_tp = src_tp[0];
      }
      else {
        if (shape.get_type().get_id() == pointer_id) {
          shape = shape.f("dereference");
        }
        dst_tp = ndt::make_fixed_dim(shape.get_dim_size(), reinterpret_cast<const intptr_t *>(shape.data()),
                                     ndt::make_type<complex<double>>());
      }
    }

    void resolve_dst_type(char *data, ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                          const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      resolve_dst_type_<std::is_same<fftw_src_type, double>::value>(data, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
    }

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      int flags;
      if (kwds[2].is_na()) {
        flags = FFTW_ESTIMATE;
      }
      else {
        flags = kwds[2].as<int>();
      }

      nd::array shape = kwds[0];
      if (!shape.is_na()) {
        if (shape.get_type().get_id() == pointer_id) {
          shape = shape;
        }
      }

      nd::array axes;
      if (!kwds[1].is_na()) {
        axes = kwds[1];
        if (axes.get_type().get_id() == pointer_id) {
          axes = axes;
        }
      }
      else {
        axes = nd::range(src_tp[0].get_ndim());
      }

      const size_stride_t *src_size_stride = reinterpret_cast<const size_stride_t *>(src_arrmeta[0]);
      const size_stride_t *dst_size_stride = reinterpret_cast<const size_stride_t *>(dst_arrmeta);

      int rank = axes.get_dim_size();
      shortvector<fftw_iodim> dims(rank);
      for (intptr_t i = 0; i < rank; ++i) {
        intptr_t j = axes(i).as<intptr_t>();
        dims[i].n = shape.is_na() ? src_size_stride[j].dim_size : shape(j).as<intptr_t>();
        dims[i].is = src_size_stride[j].stride / sizeof(fftw_src_type);
        dims[i].os = dst_size_stride[j].stride / sizeof(fftw_dst_type);
      }

      int howmany_rank = src_tp[0].get_ndim() - rank;
      shortvector<fftw_iodim> howmany_dims(howmany_rank);
      for (intptr_t i = 0, j = 0, k = 0; i < howmany_rank; ++i, ++j) {
        for (; k < rank && j == axes(k).as<intptr_t>(); ++j, ++k) {
        }
        howmany_dims[i].n = shape.is_na() ? src_size_stride[j].dim_size : shape(j).as<intptr_t>();
        howmany_dims[i].is = src_size_stride[j].stride / sizeof(fftw_src_type);
        howmany_dims[i].os = dst_size_stride[j].stride / sizeof(fftw_dst_type);
      }

      nd::array src = nd::empty(src_tp[0]);
      nd::array dst = nd::empty(dst_tp);

      ckb->emplace_back<fftw_ck<fftw_dst_type, fftw_src_type, sign>>(
          kernreq, detail::fftw_plan_guru_dft(rank, dims.get(), howmany_rank, howmany_dims.get(),
                                              reinterpret_cast<fftw_src_type *>(src.data()),
                                              reinterpret_cast<fftw_dst_type *>(dst.data()), sign, flags));
    }
  };

#endif

} // namespace dynd::nd
} // namespace dynd
