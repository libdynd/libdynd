//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/kernels/expr_kernels.hpp>

#ifdef __CUDACC__

namespace dynd {

namespace kernels {
  template <int N>
  __global__ void cuda_launch_single(char *dst, detail::array_wrapper<char *, N> src,
                                     ckernel_prefix *self)
  {
    expr_single_t func = self->get_function<expr_single_t>();
    func(dst, src, self);
  }

  template <int N>
  __global__ void cuda_launch_strided(char *dst, intptr_t dst_stride,
                                      detail::array_wrapper<char *, N> src,
                                      detail::array_wrapper<intptr_t, N> src_stride,
                                      size_t count, ckernel_prefix *self)
  {
    expr_strided_t func = self->get_function<expr_strided_t>();
    func(dst, dst_stride, src, src_stride, count, self);
  }

  template <int Nsrc>
  struct cuda_launch_ck
      : expr_ck<cuda_launch_ck<Nsrc>, kernel_request_host, Nsrc> {
    typedef cuda_launch_ck<Nsrc> self_type;

    ckernel_builder<kernel_request_cuda_device> ckb;
    dim3 blocks;
    dim3 threads;

    cuda_launch_ck(dim3 blocks, dim3 threads) : blocks(blocks), threads(threads)
    {
    }

    void single(char *dst, char *const *src)
    {
      kernels::cuda_launch_single << <blocks, threads>>>
          (dst, detail::make_array_wrapper<Nsrc>(src), ckb.get());
      // check for CUDA errors here
    }

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count)
    {
      cuda_launch_strided << <blocks, threads>>>
          (dst, dst_stride, detail::make_array_wrapper<Nsrc>(src),
           detail::make_array_wrapper<Nsrc>(src_stride), count, ckb.get());
      // check for CUDA errors here
    }

    static intptr_t
    instantiate(const arrfunc_type_data *DYND_UNUSED(self),
                const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,
                intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                const char *DYND_UNUSED(dst_arrmeta),
                intptr_t DYND_UNUSED(nsrc),
                const ndt::type *DYND_UNUSED(src_tp),
                const char *const *DYND_UNUSED(src_arrmeta),
                kernel_request_t kernreq,
                const eval::eval_context *DYND_UNUSED(ectx),
                const nd::array &DYND_UNUSED(kwds),
                const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      self_type::create(ckb, kernreq, ckb_offset, 1, 1);
      return ckb_offset;
    }

    template <arrfunc_instantiate_t instantiate>
    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                void *ckb, intptr_t ckb_offset, const ndt::type dst_tp,
                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                const char *const *src_arrmeta, kernel_request_t kernreq,
                const eval::eval_context *ectx, const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars)
    {
      bool cuda_device_readable = dst_tp.is_cuda_device_readable();
      for (intptr_t i = 0; i < Nsrc; ++i) {
        cuda_device_readable &= src_tp[i].is_cuda_device_readable();
      }

      intptr_t res_ckb_offset = ckb_offset;
      if (kernel_request_without_function(kernreq) == kernel_request_host && cuda_device_readable) {
        res_ckb_offset = self_type::instantiate(
            self, self_tp, ckb, res_ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
            src_arrmeta, kernreq, ectx, kwds, tp_vars);
        self_type *self =
            reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                ->get_at<self_type>(ckb_offset);
        ckb = &self->ckb;
        kernreq |= kernel_request_cuda_device;
        ckb_offset = 0;
      }
      instantiate(self, self_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
                  src_arrmeta, kernreq, ectx, kwds, tp_vars);

      return res_ckb_offset;
    }
  };
}
}

#endif // __CUDACC__