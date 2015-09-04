//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/substitute_shape.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    struct DYND_API neighborhood_data {
      callable op;
      start_stop_t *start_stop;

      neighborhood_data(const callable &neighborhood_op, intptr_t ndim)
      {
        op = neighborhood_op;
        start_stop = static_cast<start_stop_t *>(
            std::malloc(ndim * sizeof(start_stop_t)));
      }

      ~neighborhood_data()
      {
        std::free(start_stop);
      }
    };

    template <int N>
    struct neighborhood_ck : base_kernel<neighborhood_ck<N>, N> {
      typedef neighborhood_ck<N> self_type;

      intptr_t dst_stride;
      intptr_t src_offset[N];
      intptr_t src_stride[N];
      intptr_t count[3];
      intptr_t nh_size;
      start_stop_t *nh_start_stop;

      // local index of first in of bounds element in the neighborhood
      // local index of first out of bounds element in the neighborhood

      void single(char *dst, char *const *src)
      {
        ckernel_prefix *child = self_type::get_child();
        expr_single_t child_fn = child->get_function<expr_single_t>();

        char *src_copy[N];
        memcpy(src_copy, src, sizeof(src_copy));
        for (intptr_t j = 0; j < N; ++j) {
          src_copy[j] += src_offset[j];
        }

        nh_start_stop->start = count[0];
        nh_start_stop->stop = nh_size; // min(nh_size, dst_size)
        for (intptr_t i = 0; i < count[0]; ++i) {
          child_fn(child, dst, src_copy);
          --(nh_start_stop->start);
          dst += dst_stride;
          for (intptr_t j = 0; j < N; ++j) {
            src_copy[j] += src_stride[j];
          }
        }
        //  *nh_start = 0;
        //    *nh_stop = nh_size;
        for (intptr_t i = 0; i < count[1]; ++i) {
          child_fn(child, dst, src_copy);
          dst += dst_stride;
          for (intptr_t j = 0; j < N; ++j) {
            src_copy[j] += src_stride[j];
          }
        }
        //      *nh_start = 0;
        //        *nh_stop = count[2]; // 0 if count[2] >
        for (intptr_t i = 0; i < count[2]; ++i) {
          --(nh_start_stop->stop);
          child_fn(child, dst, src_copy);
          dst += dst_stride;
          for (intptr_t j = 0; j < N; ++j) {
            src_copy[j] += src_stride[j];
          }
        }
      }

      static intptr_t
      instantiate(char *static_data, size_t DYND_UNUSED(data_size),
                  char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                  const ndt::type &dst_tp, const char *dst_arrmeta,
                  intptr_t nsrc, const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *ectx, intptr_t nkwd,
                  const nd::array *kwds,
                  const std::map<std::string, ndt::type> &tp_vars)
      {
        std::shared_ptr<neighborhood_data> nh =
            *reinterpret_cast<std::shared_ptr<neighborhood_data> *>(
                 static_data);
        nd::callable nh_op = nh->op;

        nd::array shape = kwds[0];
        if (shape.is_missing()) {
          const nd::array &mask = kwds[2];
          shape = nd::array(mask.get_shape());
        }
        intptr_t ndim = shape.get_dim_size();

        nd::array offset;
        if (!kwds[1].is_missing()) {
          offset = kwds[1];
        }

        // Process the dst array striding/types
        const size_stride_t *dst_shape;
        ndt::type nh_dst_tp;
        const char *nh_dst_arrmeta;
        if (!dst_tp.get_as_strided(dst_arrmeta, ndim, &dst_shape, &nh_dst_tp,
                                   &nh_dst_arrmeta)) {
          std::stringstream ss;
          ss << "neighborhood callable dst must be a strided array, not "
             << dst_tp;
          throw std::invalid_argument(ss.str());
        }

        // Process the src[0] array striding/type
        const size_stride_t *src0_shape;
        ndt::type src0_el_tp;
        const char *src0_el_arrmeta;
        if (!src_tp[0].get_as_strided(src_arrmeta[0], ndim, &src0_shape,
                                      &src0_el_tp, &src0_el_arrmeta)) {
          std::stringstream ss;
          ss << "neighborhood callable argument 1 must be a 2D strided array, "
                "not " << src_tp[0];
          throw std::invalid_argument(ss.str());
        }

        // Synthesize the arrmeta for the src[0] passed to the neighborhood op
        ndt::type nh_src_tp[1];
        nh_src_tp[0] = ndt::make_fixed_dim_kind(src0_el_tp, ndim);
        arrmeta_holder nh_arrmeta;
        arrmeta_holder(nh_src_tp[0]).swap(nh_arrmeta);
        size_stride_t *nh_src0_arrmeta =
            reinterpret_cast<size_stride_t *>(nh_arrmeta.get());
        for (intptr_t i = 0; i < ndim; ++i) {
          nh_src0_arrmeta[i].dim_size = shape(i).as<intptr_t>();
          nh_src0_arrmeta[i].stride = src0_shape[i].stride;
        }
        const char *nh_src_arrmeta[1] = {nh_arrmeta.get()};

        for (intptr_t i = 0; i < ndim; ++i) {
          typedef dynd::nd::functional::neighborhood_ck<N> self_type;
          self_type *self = self_type::make(ckb, kernreq, ckb_offset);

          self->dst_stride = dst_shape[i].stride;
          for (intptr_t j = 0; j < N; ++j) {
            self->src_offset[j] =
                offset.is_null()
                    ? 0
                    : (offset(i).as<intptr_t>() * src0_shape[i].stride);
            self->src_stride[j] = src0_shape[i].stride;
          }

          self->count[0] = offset.is_null() ? 0 : -offset(i).as<intptr_t>();
          if (self->count[0] < 0) {
            self->count[0] = 0;
          } else if (self->count[0] > dst_shape[i].dim_size) {
            self->count[0] = dst_shape[i].dim_size;
          }
          self->count[2] = shape(i).as<intptr_t>() +
                           (offset.is_null() ? 0 : offset(i).as<intptr_t>()) -
                           1;
          if (self->count[2] < 0) {
            self->count[2] = 0;
          } else if (self->count[2] >
                     (dst_shape[i].dim_size - self->count[0])) {
            self->count[2] = dst_shape[i].dim_size - self->count[0];
          }
          self->count[1] =
              dst_shape[i].dim_size - self->count[0] - self->count[2];

          self->nh_size = shape(i).as<intptr_t>();
          self->nh_start_stop = nh->start_stop + i;
        }

        std::vector<array> new_kwds(nkwd + 1);
        for (int i = 0; i < nkwd; ++i) {
          new_kwds[i] = kwds[i];
        }
        new_kwds[nkwd] = reinterpret_cast<intptr_t>(nh->start_stop);

        ckb_offset = nh_op.get()->instantiate(
            nh_op.get()->static_data, 0, NULL, ckb, ckb_offset, nh_dst_tp,
            nh_dst_arrmeta, nsrc, nh_src_tp, nh_src_arrmeta,
            kernel_request_single, ectx, nkwd + 1, new_kwds.data(), tp_vars);

        return ckb_offset;
      }

      static void resolve_dst_type(
          char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
          char *DYND_UNUSED(data), ndt::type &dst_tp,
          intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
          intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        // swap in the input dimension values for the Fixed**N
        intptr_t ndim = src_tp[0].get_ndim();
        dimvector shape(ndim);
        src_tp[0].extended()->get_shape(ndim, 0, shape.get(), NULL, NULL);
        dst_tp = ndt::substitute_shape(dst_tp, ndim, shape.get());
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
