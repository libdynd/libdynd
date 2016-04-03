//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <array>

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/reduction_kernel.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    class base_reduction_callable : public base_callable {
    public:
      struct data_type {
        callable child;
        bool keepdims;
        size_t naxis;
        const int *axes;
        int axis;
        intptr_t ndim;
      };

      struct node_type : call_node {
        bool inner;

        node_type(base_callable *callee) : call_node(callee) {}
      };

      base_reduction_callable() : base_callable(ndt::type(), sizeof(node_type)) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *data, call_graph &cg, const ndt::type &res_tp,
                        size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        std::cout << "base_reduction_callable::resolve" << std::endl;

        node_type *node = cg.emplace_back<node_type>(this);

        callable &child = reinterpret_cast<data_type *>(data)->child;
        const ndt::type &child_ret_tp = child.get_ret_type();

        bool reduce = reinterpret_cast<data_type *>(data)->axes == NULL;
        for (size_t i = 0; i < reinterpret_cast<data_type *>(data)->naxis && !reduce; ++i) {
          if (reinterpret_cast<data_type *>(data)->axes[i] == reinterpret_cast<data_type *>(data)->axis) {
            reduce = true;
          }
        }

        ndt::type arg_element_tp[1];
        if (reduce) {
          arg_element_tp[0] = src_tp[0].extended<ndt::base_dim_type>()->get_element_type();
        } else {
          arg_element_tp[0] = src_tp[0];
        }
        ++reinterpret_cast<data_type *>(data)->axis;

        ndt::type ret_element_tp;
        if (reinterpret_cast<data_type *>(data)->axis == reinterpret_cast<data_type *>(data)->ndim) {
          node->inner = true;
          ret_element_tp =
              child->resolve(this, nullptr, cg, child_ret_tp, nsrc, arg_element_tp, nkwd - 3, kwds + 3, tp_vars);

          nd::array error_mode = eval::default_eval_context.errmode;
          assign->resolve(this, nullptr, cg, ret_element_tp, 1, arg_element_tp, 1, &error_mode, tp_vars);
        } else {
          node->inner = false;
          ret_element_tp = this->resolve(this, data, cg, res_tp, nsrc, arg_element_tp, nkwd, kwds, tp_vars);
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

    template <type_id_t Arg0ID>
    class reduction_callable;

    template <>
    class reduction_callable<fixed_dim_id> : public base_reduction_callable {
      void instantiate(call_node *&node, char *DYND_UNUSED(data), kernel_builder *ckb,
                       const ndt::type &DYND_UNUSED(dst_tp), const char *dst_arrmeta, intptr_t nsrc,
                       const ndt::type *DYND_UNUSED(src_tp), const char *const *src_arrmeta, kernel_request_t kernreq,
                       intptr_t nkwd, const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
        std::cout << "reduction_callable<fixed_dim_id>::instantiate" << std::endl;

        bool inner = reinterpret_cast<node_type *>(node)->inner;
        if (inner) {
          intptr_t src_size = reinterpret_cast<const size_stride_t *>(src_arrmeta[0])->dim_size;
          intptr_t src_stride = reinterpret_cast<const size_stride_t *>(src_arrmeta[0])->stride;

          typedef reduction_kernel<fixed_dim_id, false, true> self_type;
          intptr_t root_ckb_offset = ckb->size();
          ckb->emplace_back<self_type>(kernreq);
          node = next(node);
          self_type *e = ckb->get_at<self_type>(root_ckb_offset);
          e->src_stride = src_stride;
          e->_size = src_size;

          if (true) { // identity is null
            e->size_first = e->_size - 1;
            e->src_stride_first = e->src_stride;
          } else {
            e->size_first = e->_size;
            e->src_stride_first = 0;
          }

          const char *src0_element_arrmeta = src_arrmeta[0] + sizeof(size_stride_t);

          node->callee->instantiate(node, nullptr, ckb, ndt::type(), dst_arrmeta + sizeof(size_stride_t), nsrc, nullptr,
                                    &src0_element_arrmeta, kernel_request_strided, nkwd - 3, kwds + 3, tp_vars);

          intptr_t init_offset = ckb->size();
          node->callee->instantiate(node, nullptr, ckb, ndt::type(), dst_arrmeta + sizeof(size_stride_t), nsrc, nullptr,
                                    &src0_element_arrmeta, kernel_request_single, 0, nullptr, tp_vars);

          e = ckb->get_at<self_type>(root_ckb_offset);
          e->init_offset = init_offset - root_ckb_offset;
        } else {
          const char *src0_element_arrmeta = src_arrmeta[0] + sizeof(size_stride_t);

          intptr_t src_size = reinterpret_cast<const size_stride_t *>(src_arrmeta[0])->dim_size;
          intptr_t src_stride = reinterpret_cast<const size_stride_t *>(src_arrmeta[0])->stride;

          ckb->emplace_back<reduction_kernel<fixed_dim_id, false, false>>(kernreq, src_size, src_stride);
          node = next(node);

          node->callee->instantiate(node, nullptr, ckb, ndt::type(), dst_arrmeta + sizeof(size_stride_t), nsrc, nullptr,
                                    &src0_element_arrmeta, kernel_request_single, 0, nullptr, tp_vars);
        }
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
