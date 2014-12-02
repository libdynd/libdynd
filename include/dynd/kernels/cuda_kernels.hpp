//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/kernels/expr_kernels.hpp>

#ifdef __CUDACC__

namespace dynd {
namespace kernels {
  namespace detail {

    __global__ void cuda_parallel_single(ckernel_prefix *self, char *dst)
    {
      expr_single_t func = self->get_function<expr_single_t>();
      func(dst, NULL, self);
    }

    template <typename... T>
    __global__ void
    cuda_parallel_single(ckernel_prefix *DYND_CONDITIONAL_UNUSED(self),
                         char *DYND_CONDITIONAL_UNUSED(dst),
                         T... DYND_CONDITIONAL_UNUSED(xsrc))
    {
      char *src[sizeof...(T)] = {xsrc...};

      expr_single_t func = self->get_function<expr_single_t>();
      func(dst, src, self);
    }

    template <typename T>
    class cuda_parallel_ck;

    template <size_t... I>
    class cuda_parallel_ck<index_sequence<I...>>
        : public expr_ck<cuda_parallel_ck<index_sequence<I...>>,
                         kernel_request_host, sizeof...(I)> {
      typedef cuda_parallel_ck<index_sequence<I...>> self_type;

      ckernel_builder<kernel_request_cuda_device> m_ckb;
      dim3 m_blocks;
      dim3 m_threads;

    public:
      cuda_parallel_ck(dim3 blocks, dim3 threads)
          : m_blocks(blocks), m_threads(threads)
      {
      }

      ckernel_builder<kernel_request_cuda_device> *get_ckb() { return &m_ckb; }

      void single(char *dst, char **DYND_CONDITIONAL_UNUSED(src))
      {
        detail::cuda_parallel_single << <m_blocks, m_threads>>>
            (m_ckb.get(), dst, src[I]...);
        throw_if_not_cuda_success(cudaDeviceSynchronize());
      }

      static intptr_t instantiate(const arrfunc_type_data *DYND_UNUSED(af_self),
                                  const arrfunc_type *DYND_UNUSED(af_tp),
                                  void *ckb, intptr_t ckb_offset,
                                  const ndt::type &DYND_UNUSED(dst_tp),
                                  const char *DYND_UNUSED(dst_arrmeta),
                                  const ndt::type *DYND_UNUSED(src_tp),
                                  const char *const *DYND_UNUSED(src_arrmeta),
                                  kernel_request_t kernreq,
                                  const eval::eval_context *DYND_UNUSED(ectx),
                                  const nd::array &DYND_UNUSED(args),
                                  const nd::array &kwds)
      {
        self_type::create(ckb, kernreq, ckb_offset,
                          kwds.p("grids").as<intptr_t>(),
                          kwds.p("blocks").as<intptr_t>());
        return ckb_offset;
      }
    };
  }

  template <int N>
  using cuda_parallel_ck =
      detail::cuda_parallel_ck<typename make_index_sequence<0, N>::type>;
}
}

#endif // __CUDACC__