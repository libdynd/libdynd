//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/callables/base_elwise_callable.hpp>
#include <dynd/kernels/elwise_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    struct no_traits {
      no_traits(char *DYND_UNUSED(data)) {}

      size_t begin() { return 0; }

      static char *child_data(char *data) { return data; }
    };

    struct state_traits {
      size_t &it;

      state_traits(char *data) : it(*reinterpret_cast<size_t *>(data)) {}

      size_t &begin() {
        it = 0;
        return it;
      }

      static char *child_data(char *data) { return reinterpret_cast<char *>(reinterpret_cast<size_t *>(data) + 1); }
    };

    /**
     * This defines the type and keyword argument resolution for
     * an elwise callable.
     */
    template <type_id_t DstTypeID, type_id_t SrcTypeID, typename TraitsType, size_t N>
    class elwise_callable;

    template <typename TraitsType, size_t N>
    class elwise_callable<fixed_dim_id, fixed_dim_id, TraitsType, N> : public base_elwise_callable<N> {
      typedef typename base_elwise_callable<N>::data_type data_type;

    public:
      void resolve(call_graph &cg, const char *data) {
        // outermost -- stores index, doesn't increment it
        // otherwise -- stores index, increments it
        // normal -- nothing

        const std::array<bool, N> &arg_broadcast = reinterpret_cast<const data_type *>(data)->arg_broadcast;
        cg.emplace_back([arg_broadcast](kernel_builder &kb, kernel_request_t kernreq, char *data,
                                        const char *dst_arrmeta, size_t DYND_UNUSED(nsrc),
                                        const char *const *src_arrmeta) {
          intptr_t size = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->dim_size;
          intptr_t dst_stride = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->stride;

          std::array<const char *, N> child_src_arrmeta;
          std::array<intptr_t, N> src_stride;
          for (size_t i = 0; i < N; ++i) {
            if (arg_broadcast[i]) {
              src_stride[i] = 0;
              child_src_arrmeta[i] = src_arrmeta[i];
            } else {
              src_stride[i] = reinterpret_cast<const size_stride_t *>(src_arrmeta[i])->stride;
              child_src_arrmeta[i] = src_arrmeta[i] + sizeof(size_stride_t);
            }
          }

          kb.emplace_back<elwise_kernel<fixed_dim_id, fixed_dim_id, TraitsType, N>>(kernreq, data, size, dst_stride,
                                                                                    src_stride.data());

          kb(kernel_request_strided, TraitsType::child_data(data), dst_arrmeta + sizeof(size_stride_t), N,
             child_src_arrmeta.data());
        });
      }

      ndt::type with_return_type(intptr_t ret_size, const ndt::type &ret_element_tp) {
        return ndt::make_type<ndt::fixed_dim_type>(ret_size, ret_element_tp);
      }
    };

    // src is either fixed or var
    template <typename TraitsType, size_t N>
    class elwise_callable<fixed_dim_id, var_dim_id, TraitsType, N> : public base_elwise_callable<N> {
      typedef typename base_elwise_callable<N>::data_type data_type;

    public:
      void resolve(call_graph &cg, const char *data) {
        std::array<bool, N> arg_broadcast = reinterpret_cast<const data_type *>(data)->arg_broadcast;
        std::array<bool, N> arg_var = reinterpret_cast<const data_type *>(data)->arg_var;

        cg.emplace_back([arg_broadcast, arg_var](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
                                                 const char *dst_arrmeta, size_t DYND_UNUSED(nsrc),
                                                 const char *const *src_arrmeta) {
          intptr_t dst_size = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->dim_size;
          intptr_t dst_stride = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->stride;

          std::array<const char *, N> child_src_arrmeta;
          std::array<intptr_t, N> src_stride, src_offset;
          for (size_t i = 0; i < N; ++i) {
            if (arg_var[i]) {
              const ndt::var_dim_type::metadata_type *src_md =
                  reinterpret_cast<const ndt::var_dim_type::metadata_type *>(src_arrmeta[i]);
              src_stride[i] = src_md->stride;
              src_offset[i] = src_md->offset;
              child_src_arrmeta[i] = src_arrmeta[i] + sizeof(ndt::var_dim_type::metadata_type);
            } else {
              if (arg_broadcast[i]) {
                src_offset[i] = 0;
                src_stride[i] = 0;
                child_src_arrmeta[i] = src_arrmeta[i];
              } else {
                src_offset[i] = 0;
                src_stride[i] = reinterpret_cast<const size_stride_t *>(src_arrmeta[i])->stride;
                child_src_arrmeta[i] = src_arrmeta[i] + sizeof(ndt::var_dim_type::metadata_type);
              }
            }
          }

          kb.emplace_back<elwise_kernel<fixed_dim_id, var_dim_id, TraitsType, N>>(
              kernreq, dst_size, dst_stride, src_stride.data(), src_offset.data(), arg_var.data());

          kb(kernel_request_strided, nullptr, dst_arrmeta + sizeof(size_stride_t), N, child_src_arrmeta.data());
        });
      }

      ndt::type with_return_type(intptr_t ret_size, const ndt::type &ret_element_tp) {
        if (ret_size == 1) {
          return ndt::make_type<ndt::var_dim_type>(ret_element_tp);
        }

        return ndt::make_type<ndt::fixed_dim_type>(ret_size, ret_element_tp);
      }
    };

    template <typename TraitsType, size_t N>
    class elwise_callable<var_dim_id, fixed_dim_id, TraitsType, N> : public base_elwise_callable<N> {
      typedef typename base_elwise_callable<N>::data_type node_type;

    public:
      ndt::type with_return_type(intptr_t DYND_UNUSED(ret_size), const ndt::type &ret_element_tp) {
        return ndt::make_type<ndt::var_dim_type>(ret_element_tp);
      }

      void resolve(call_graph &cg, const char *data) {
        std::array<bool, N> arg_broadcast = reinterpret_cast<const node_type *>(data)->arg_broadcast;
        std::array<bool, N> arg_var = reinterpret_cast<const node_type *>(data)->arg_var;
        intptr_t res_alignment = reinterpret_cast<const node_type *>(data)->res_alignment;

        cg.emplace_back([arg_broadcast, arg_var, res_alignment](
            kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data), const char *dst_arrmeta,
            size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {

          const ndt::var_dim_type::metadata_type *dst_md =
              reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta);

          std::array<const char *, N> child_src_arrmeta;
          std::array<intptr_t, N> src_stride;
          std::array<intptr_t, N> src_offset;
          std::array<intptr_t, N> src_size;
          for (size_t i = 0; i < N; ++i) {
            if (arg_var[i]) {
              const ndt::var_dim_type::metadata_type *src_md =
                  reinterpret_cast<const ndt::var_dim_type::metadata_type *>(src_arrmeta[i]);
              src_stride[i] = src_md->stride;
              src_offset[i] = src_md->offset;
              child_src_arrmeta[i] = src_arrmeta[i] + sizeof(ndt::var_dim_type::metadata_type);
              //            src_size[i] = -1;
            } else {
              if (arg_broadcast[i]) {
                src_stride[i] = 0;
                src_offset[i] = 0;
                child_src_arrmeta[i] = src_arrmeta[i];
                src_size[i] = 1;
              } else {
                src_offset[i] = 0;
                src_stride[i] = reinterpret_cast<const size_stride_t *>(src_arrmeta[i])->stride;
                src_size[i] = reinterpret_cast<const size_stride_t *>(src_arrmeta[i])->dim_size;
              }
            }
          }

          kb.emplace_back<elwise_kernel<var_dim_id, fixed_dim_id, TraitsType, N>>(
              kernreq, dst_md->blockref.get(), res_alignment, dst_md->stride, dst_md->offset, src_stride.data(),
              src_offset.data(), src_size.data(), arg_var.data());

          kb(kernel_request_strided, nullptr, dst_arrmeta + sizeof(ndt::var_dim_type::metadata_type), N,
             child_src_arrmeta.data());
        });
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
