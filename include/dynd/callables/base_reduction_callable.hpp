//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <array>

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/reduction_kernel.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    class base_reduction_callable : public base_callable {
    public:
      struct data_type {
        callable identity;
        callable child;
        bool keepdims;
        size_t naxis;
        const int *axes;
        int axis;
        intptr_t ndim;
      };

      struct node_type {
        bool inner;
        bool broadcast;
        bool keepdim;
      };

      base_reduction_callable() : base_callable(ndt::type()) {}

      virtual void resolve(call_graph &cg, char *data) = 0;

      ndt::type resolve(base_callable *caller, char *data, call_graph &cg, const ndt::type &res_tp, size_t nsrc,
                        const ndt::type *src_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        node_type node;

        callable &child = reinterpret_cast<data_type *>(data)->child;
        const ndt::type &child_ret_tp = child->get_ret_type();

        bool reduce = reinterpret_cast<data_type *>(data)->axes == NULL;
        for (size_t i = 0; i < reinterpret_cast<data_type *>(data)->naxis && !reduce; ++i) {
          if (reinterpret_cast<data_type *>(data)->axes[i] == reinterpret_cast<data_type *>(data)->axis) {
            reduce = true;
          }
        }
        node.broadcast = !reduce;
        node.keepdim = reinterpret_cast<data_type *>(data)->keepdims;

        std::vector<ndt::type> arg_element_tp(2);
        for (size_t i = 0; i < nsrc; ++i) {
          if (reduce) {
            arg_element_tp[i] = src_tp[i].extended<ndt::base_dim_type>()->get_element_type();
          } else {
            arg_element_tp[i] = src_tp[i].extended<ndt::base_dim_type>()->get_element_type();
          }
        }
        ++reinterpret_cast<data_type *>(data)->axis;

        ndt::type ret_element_tp;
        if (reinterpret_cast<data_type *>(data)->axis == reinterpret_cast<data_type *>(data)->ndim) {
          node.inner = true;
          resolve(cg, reinterpret_cast<char *>(&node));

          ret_element_tp =
              child->resolve(this, nullptr, cg, child_ret_tp, nsrc, arg_element_tp.data(), nkwd - 2, kwds + 2, tp_vars);

          nd::callable constant = reinterpret_cast<data_type *>(data)->identity;
          constant->resolve(this, nullptr, cg, ret_element_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
        } else {
          node.inner = false;
          resolve(cg, reinterpret_cast<char *>(&node));

          ret_element_tp = caller->resolve(this, data, cg, res_tp, nsrc, arg_element_tp.data(), nkwd, kwds, tp_vars);
        }

        if (reduce) {
          if (reinterpret_cast<data_type *>(data)->keepdims) {
            return ndt::make_type<ndt::fixed_dim_type>(1, ret_element_tp);
          }

          return ret_element_tp;
        }

        return src_tp[0].extended<ndt::base_dim_type>()->with_element_type(ret_element_tp);
      }
    };

    template <type_id_t Arg0ID, size_t NArg>
    class reduction_callable;

    template <size_t NArg>
    class reduction_callable<fixed_dim_id, NArg> : public base_reduction_callable {
      void resolve(call_graph &cg, char *data) {
        bool inner = reinterpret_cast<node_type *>(data)->inner;
        bool broadcast = reinterpret_cast<node_type *>(data)->broadcast;
        bool keepdim = reinterpret_cast<node_type *>(data)->keepdim;

        cg.emplace_back([inner, broadcast, keepdim](kernel_builder &kb, kernel_request_t kernreq,
                                                    char *DYND_UNUSED(data), const char *dst_arrmeta, size_t nsrc,
                                                    const char *const *src_arrmeta) {
          if (inner) {
            if (!broadcast) {
              intptr_t src_size = reinterpret_cast<const size_stride_t *>(src_arrmeta[0])->dim_size;

              typedef reduction_kernel<ndt::fixed_dim_type, false, true, NArg> self_type;
              intptr_t root_ckb_offset = kb.size();
              kb.emplace_back<self_type>(kernreq);
              self_type *e = kb.get_at<self_type>(root_ckb_offset);
              for (size_t i = 0; i < NArg; ++i) {
                e->src_stride[i] = reinterpret_cast<const size_stride_t *>(src_arrmeta[i])->stride;
              }
              e->_size = src_size;

              e->size_first = e->_size;
              for (size_t i = 0; i < NArg; ++i) {
                e->src_stride_first[i] = 0;
              }

              const char *src_element_arrmeta[NArg];
              for (size_t i = 0; i < NArg; ++i) {
                src_element_arrmeta[i] = src_arrmeta[i] + sizeof(size_stride_t);
              }

              kb(kernel_request_strided, nullptr, dst_arrmeta + sizeof(size_stride_t), nsrc, src_element_arrmeta);

              intptr_t init_offset = kb.size();
              kb(kernel_request_single, nullptr, dst_arrmeta + sizeof(size_stride_t), nsrc, src_element_arrmeta);

              e = kb.get_at<self_type>(root_ckb_offset);
              e->init_offset = init_offset - root_ckb_offset;
            } else {
              const char *src_element_arrmeta[NArg];
              for (size_t j = 0; j < NArg; ++j) {
                src_element_arrmeta[j] = src_arrmeta[j] + sizeof(size_stride_t);
              }

              intptr_t src_size = reinterpret_cast<const size_stride_t *>(src_arrmeta[0])->dim_size;
              intptr_t dst_stride = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->stride;

              const char *dst_element_arrmeta = dst_arrmeta + sizeof(size_stride_t);

              typedef reduction_kernel<ndt::fixed_dim_type, true, true, NArg> self_type;
              intptr_t root_ckb_offset = kb.size();
              kb.emplace_back<self_type>(kernreq, dst_stride, src_arrmeta);

              self_type *self_k = kb.get_at<self_type>(root_ckb_offset);

              // The striding parameters
              self_k->_size = src_size;
              // Need to retrieve 'e' again because it may have moved
              self_k->size_first = self_k->_size;
              self_k->dst_stride_first = 0;
              for (size_t i = 0; i < NArg; ++i) {
                self_k->src_stride_first[i] = 0;
              }

              kb(kernel_request_strided, nullptr, dst_element_arrmeta, nsrc, src_element_arrmeta);

              intptr_t init_offset = kb.size();
              kb(kernel_request_strided, nullptr, dst_element_arrmeta, nsrc, src_element_arrmeta);

              self_k = kb.get_at<self_type>(root_ckb_offset);
              self_k->dst_init_kernel_offset = init_offset - root_ckb_offset;
            }
          } else {
            const char *src_element_arrmeta[NArg];
            for (size_t j = 0; j < NArg; ++j) {
              src_element_arrmeta[j] = src_arrmeta[j] + sizeof(size_stride_t);
            }

            intptr_t src_size = reinterpret_cast<const size_stride_t *>(src_arrmeta[0])->dim_size;

            if (broadcast) {
              kb.emplace_back<reduction_kernel<ndt::fixed_dim_type, true, false, NArg>>(kernreq, src_size, dst_arrmeta,
                                                                                        src_arrmeta);
              kernreq = kernel_request_strided;
            } else {
              kb.emplace_back<reduction_kernel<ndt::fixed_dim_type, false, false, NArg>>(kernreq, src_size,
                                                                                         src_arrmeta);
              kernreq = kernel_request_single;
            }

            kb(kernreq, nullptr, keepdim ? (dst_arrmeta + sizeof(size_stride_t)) : dst_arrmeta, nsrc,
               src_element_arrmeta);
          }
        });
      }
    };

    template <size_t NArg>
    class reduction_callable<var_dim_id, NArg> : public base_reduction_callable {
      void resolve(call_graph &cg, char *DYND_UNUSED(data)) {
        cg.emplace_back([](kernel_builder &kb, kernel_request_t kernreq,
                           char *DYND_UNUSED(data), const char *dst_arrmeta, size_t nsrc,
                           const char *const *src_arrmeta) {
          typedef reduction_kernel<ndt::var_dim_type, false, true, NArg> self_type;
          intptr_t root_ckb_offset = kb.size();
          kb.emplace_back<self_type>(
              kernreq, reinterpret_cast<const ndt::var_dim_type::metadata_type *>(src_arrmeta[0])->stride);

          self_type *e = kb.get_at<self_type>(root_ckb_offset);
          const char *src_element_arrmeta[NArg];
          for (size_t j = 0; j < NArg; ++j) {
            src_element_arrmeta[j] = src_arrmeta[j] + sizeof(ndt::var_dim_type::metadata_type);
          }

          kb(kernel_request_strided, nullptr, dst_arrmeta, nsrc, src_element_arrmeta);

          intptr_t init_offset = kb.size();
          kb(kernel_request_single, nullptr, dst_arrmeta, nsrc, src_element_arrmeta);

          e = kb.get_at<self_type>(root_ckb_offset);
          e->init_offset = init_offset - root_ckb_offset;
        });
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
