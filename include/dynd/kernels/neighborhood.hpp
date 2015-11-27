//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <algorithm>

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/substitute_shape.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <int N>
    struct neighborhood_kernel : base_kernel<neighborhood_kernel<N>, N> {
      struct static_data_type {
        callable child;
        callable boundary_child;

        static_data_type(const callable &child, const callable &boundary_child)
            : child(child), boundary_child(boundary_child)
        {
        }
      };

      struct data_type {
        ndt::type child_src_tp;
        size_stride_t child_src_arrmeta[5];

        const ndt::type *src_tp;
        const char *src_arrmeta;

        intptr_t ndim;
        intptr_t *shape;
        int *offset;
        std::shared_ptr<bool> out_of_bounds;

        data_type(const ndt::type *src_tp, intptr_t ndim, int *shape, int *offset)
            : src_tp(src_tp), src_arrmeta(NULL), ndim(ndim), offset(offset), out_of_bounds(std::make_shared<bool>())
        {
          this->shape = new intptr_t[ndim];
          for (int i = 0; i < ndim; ++i) {
            this->shape[i] = shape[i];
          }

          /*
                    child_src_arrmeta = new char[ndim * sizeof(size_stride_t)];
                    for (int i = 0; i < ndim; ++i) {
                      (reinterpret_cast<size_stride_t *>(child_src_arrmeta) + i)->dim_size = shape[i];
                      (reinterpret_cast<size_stride_t *>(child_src_arrmeta) + i)->stride = sizeof(int);
                    }
          */

          if (ndim == 1) {
            child_src_arrmeta[0].dim_size = shape[0];
            child_src_arrmeta[0].stride = sizeof(int);
          }
          else if (ndim == 2) {
            child_src_arrmeta[0].dim_size = shape[0];
            child_src_arrmeta[0].stride = 4 * sizeof(int);
            child_src_arrmeta[1].dim_size = shape[1];
            child_src_arrmeta[1].stride = sizeof(int);
          }
          else if (ndim == 3) {
            child_src_arrmeta[0].dim_size = shape[0];
            child_src_arrmeta[0].stride = 4 * 4 * sizeof(int);
            child_src_arrmeta[1].dim_size = shape[1];
            child_src_arrmeta[1].stride = 4 * sizeof(int);
            child_src_arrmeta[2].dim_size = shape[2];
            child_src_arrmeta[2].stride = sizeof(int);
          }
        }

        ~data_type()
        {
          //          delete[] child_src_arrmeta;
        }
      };

      intptr_t dst_stride;
      intptr_t src0_offset;
      intptr_t src0_stride;
      intptr_t offset;
      intptr_t counts[3];
      std::shared_ptr<bool> out_of_bounds;
      intptr_t boundary_child_offset;

      neighborhood_kernel(intptr_t dst_stride, intptr_t src0_size, intptr_t src0_stride, intptr_t size, intptr_t offset,
                          const std::shared_ptr<bool> &out_of_bounds)
          : dst_stride(dst_stride), src0_offset(offset * src0_stride), src0_stride(src0_stride), offset(offset),
            out_of_bounds(out_of_bounds)
      {
        counts[0] = std::min((intptr_t)0, src0_size + offset);
        counts[1] = std::min(src0_size + offset, src0_size - size + 1);
        counts[2] = src0_size + offset;

        *out_of_bounds = false;
      }

      void single(char *dst, char *const *src)
      {
        ckernel_prefix *child = this->get_child();
        ckernel_prefix *boundary_child = this->get_child(boundary_child_offset);

        char *src0 = src[0] + src0_offset;

        intptr_t i = offset;
        bool old_out_of_bounds = *out_of_bounds;

        *out_of_bounds = true;
        while (i < counts[0]) {
          boundary_child->single(dst, &src0);
          ++i;
          dst += dst_stride;
          src0 += src0_stride;
        };

        *out_of_bounds = old_out_of_bounds;
        while (i < counts[1]) {
          if (*out_of_bounds) {
            boundary_child->single(dst, &src0);
          }
          else {
            child->single(dst, &src0);
          }
          ++i;
          dst += dst_stride;
          src0 += src0_stride;
        }

        *out_of_bounds = true;
        while (i < counts[2]) {
          boundary_child->single(dst, &src0);
          ++i;
          dst += dst_stride;
          src0 += src0_stride;
        }

        *out_of_bounds = old_out_of_bounds;
      }

      static char *data_init(char *static_data, const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                             const ndt::type *src_tp, intptr_t DYND_UNUSED(nkwd), const array *kwds,
                             const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        char *data = reinterpret_cast<char *>(
            new data_type(src_tp, kwds[0].get_dim_size(), reinterpret_cast<int *>(kwds[0].data()),
                          kwds[1].is_missing() ? NULL : reinterpret_cast<int *>(kwds[1].data())));

        reinterpret_cast<data_type *>(data)->child_src_tp = ndt::substitute_shape(
            reinterpret_cast<callable *>(static_data)->get_arg_type(0), reinterpret_cast<data_type *>(data)->ndim,
            reinterpret_cast<data_type *>(data)->shape);

        return data;
      }

      static void resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                                   intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, intptr_t DYND_UNUSED(nkwd),
                                   const array *DYND_UNUSED(kwds),
                                   const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        // swap in the input dimension values for the Fixed**N
        intptr_t ndim = src_tp[0].get_ndim();
        dimvector shape(ndim);
        src_tp[0].extended()->get_shape(ndim, 0, shape.get(), NULL, NULL);
        dst_tp = ndt::substitute_shape(dst_tp, ndim, shape.get());
      }

      static intptr_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars)
      {
        intptr_t neighborhood_offset = ckb_offset;
        neighborhood_kernel::make(
            ckb, kernreq, ckb_offset, reinterpret_cast<const fixed_dim_type_arrmeta *>(dst_arrmeta)->stride,
            reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->dim_size,
            reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->stride,
            reinterpret_cast<data_type *>(data)->shape[0],
            (reinterpret_cast<data_type *>(data)->offset == NULL) ? 0 : reinterpret_cast<data_type *>(data)->offset[0],
            reinterpret_cast<data_type *>(data)->out_of_bounds);

        const ndt::type &child_dst_tp = dst_tp.extended<ndt::fixed_dim_type>()->get_element_type();
        const char *child_dst_arrmeta = dst_arrmeta + sizeof(fixed_dim_type_arrmeta);

        reinterpret_cast<data_type *>(data)->ndim -= 1;
        reinterpret_cast<data_type *>(data)->shape += 1;
        if (reinterpret_cast<data_type *>(data)->offset != NULL) {
          reinterpret_cast<data_type *>(data)->offset += 1;
        }

        if (reinterpret_cast<data_type *>(data)->ndim == 0) {
          const callable &child = *reinterpret_cast<callable *>(static_data);
          const callable &boundary_child = reinterpret_cast<static_data_type *>(static_data)->boundary_child;

          const char *child_src_arrmeta =
              reinterpret_cast<char *>(reinterpret_cast<data_type *>(data)->child_src_arrmeta);
          ckb_offset =
              child.get()->instantiate(child.get()->static_data(), NULL, ckb, ckb_offset, child_dst_tp,
                                       child_dst_arrmeta, nsrc, &reinterpret_cast<data_type *>(data)->child_src_tp,
                                       &child_src_arrmeta, kernel_request_single, ectx, nkwd - 3, kwds + 3, tp_vars);
          neighborhood_kernel::get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
                                        neighborhood_offset)
              ->boundary_child_offset = ckb_offset - neighborhood_offset;

          ckb_offset = boundary_child.get()->instantiate(boundary_child.get()->static_data(), NULL, ckb, ckb_offset,
                                                         child_dst_tp, child_dst_arrmeta, 0, NULL, NULL,
                                                         kernel_request_single, ectx, nkwd - 3, kwds + 3, tp_vars);

          delete reinterpret_cast<data_type *>(data);
          return ckb_offset;
        }

        ndt::type child_src_tp[N];
        const char *child_src_arrmeta[N];
        for (int i = 0; i < N; ++i) {
          child_src_tp[i] = src_tp[i].extended<ndt::fixed_dim_type>()->get_element_type();
          child_src_arrmeta[i] = src_arrmeta[i] + sizeof(fixed_dim_type_arrmeta);
        }

        neighborhood_kernel::get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
                                      neighborhood_offset)
            ->boundary_child_offset = sizeof(neighborhood_kernel);

        return instantiate(static_data, data, ckb, ckb_offset, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp,
                           child_src_arrmeta, kernel_request_single, ectx, nkwd, kwds, tp_vars);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
