//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/kernels/cuda_kernels.hpp>

namespace dynd {
namespace kernels {

  template <int N, bool with_dst = true>
  struct permute_ck;

  template <int N>
  struct permute_ck<N, true>
      : expr_ck<permute_ck<N, true>, kernel_request_host, N> {
    typedef permute_ck self_type;

    intptr_t perm[N];

    permute_ck(const intptr_t *perm)
    {
      memcpy(this->perm, perm, sizeof(this->perm));
    }

    void single(char *dst, char *const *src)
    {
      char *src_inv_perm[N];
      inv_permute(src_inv_perm, dst, src, perm);

      ckernel_prefix *child = this->get_child_ckernel();
      expr_single_t single = child->get_function<expr_single_t>();
      single(NULL, src_inv_perm, child);
    }

    static intptr_t
    instantiate(const arrfunc_type_data *self,
                const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,
                intptr_t ckb_offset, const ndt::type &dst_tp,
                const char *dst_arrmeta, const ndt::type *src_tp,
                const char *const *src_arrmeta, kernel_request_t kernreq,
                const eval::eval_context *ectx, const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars)
    {
      const std::pair<nd::arrfunc, std::vector<intptr_t>> *data =
          self->get_data_as<std::pair<nd::arrfunc, std::vector<intptr_t>>>();

      const arrfunc_type_data *child = data->first.get();
      const arrfunc_type *child_tp = data->first.get_type();

      const intptr_t *perm = data->second.data();

      ndt::type src_tp_inv_perm[N];
      inv_permute(src_tp_inv_perm, dst_tp, src_tp, perm);

      const char *src_arrmeta_inv_perm[N];
      inv_permute(src_arrmeta_inv_perm, dst_arrmeta, src_arrmeta, perm);

      self_type::create(ckb, kernreq, ckb_offset,
                        array_wrapper<intptr_t, N>(perm));
      return child->instantiate(
          child, child_tp, ckb, ckb_offset, ndt::make_type<void>(), NULL,
          src_tp_inv_perm, src_arrmeta_inv_perm, kernreq, ectx, kwds, tp_vars);
    }

  private:
    template <typename T>
    static void inv_permute(T *src_copy, const T &dst, const T *src,
                            const intptr_t *perm)
    {
      for (size_t i = 0; i < N; ++i) {
        intptr_t j = perm[i];
        if (j == -1) {
          src_copy[i] = dst;
        } else {
          src_copy[i] = src[j];
        }
      }
    }
  };

  template <int N>
  struct permute_ck<N, false>
      : expr_ck<permute_ck<N, false>, kernel_request_host, N> {
    typedef permute_ck self_type;

    intptr_t perm[N];

    permute_ck(const intptr_t *perm)
    {
      memcpy(this->perm, perm, sizeof(this->perm));
    }

    void single(char *dst, char *const *src)
    {
      char *src_inv_perm[N];
      inv_permute(src_inv_perm, src, perm);

      ckernel_prefix *child = this->get_child_ckernel();
      expr_single_t single = child->get_function<expr_single_t>();
      single(dst, src_inv_perm, child);
    }

    static intptr_t
    instantiate(const arrfunc_type_data *self,
                const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,
                intptr_t ckb_offset, const ndt::type &dst_tp,
                const char *dst_arrmeta, const ndt::type *src_tp,
                const char *const *src_arrmeta, kernel_request_t kernreq,
                const eval::eval_context *ectx, const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars)
    {
      const std::pair<nd::arrfunc, std::vector<intptr_t>> *data =
          self->get_data_as<std::pair<nd::arrfunc, std::vector<intptr_t>>>();

      const arrfunc_type_data *child = data->first.get();
      const arrfunc_type *child_tp = data->first.get_type();

      const intptr_t *perm = data->second.data();

      ndt::type src_tp_inv_perm[N];
      inv_permute(src_tp_inv_perm, src_tp, perm);

      const char *src_arrmeta_inv_perm[N];
      inv_permute(src_arrmeta_inv_perm, src_arrmeta, perm);

      self_type::create(ckb, kernreq, ckb_offset,
                        array_wrapper<intptr_t, N>(perm));
      return child->instantiate(
          child, child_tp, ckb, ckb_offset, dst_tp, dst_arrmeta,
          src_tp_inv_perm, src_arrmeta_inv_perm, kernreq, ectx, kwds, tp_vars);
    }

  private:
    template <typename T>
    static void inv_permute(T *src_copy, const T *src, const intptr_t *perm)
    {
      for (size_t i = 0; i < N; ++i) {
        intptr_t j = perm[i];
        src_copy[i] = src[j];
      }
    }
  };

} // dynd::kernels

namespace nd {
  namespace functional {

    arrfunc permute(const arrfunc &child, const std::vector<intptr_t> &perm);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd