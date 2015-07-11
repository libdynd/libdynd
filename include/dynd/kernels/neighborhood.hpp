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

    struct neighborhood_data {
      arrfunc op;
      start_stop_t *start_stop;

      neighborhood_data(const arrfunc &neighborhood_op, intptr_t ndim)
      {
        op = neighborhood_op;
        start_stop = static_cast<start_stop_t *>(
            std::malloc(ndim * sizeof(start_stop_t)));
      }

      ~neighborhood_data() { std::free(start_stop); }
    };

    template <int N>
    struct neighborhood_ck
        : base_kernel<neighborhood_ck<N>, kernel_request_host, N> {
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
        ckernel_prefix *child = self_type::get_child_ckernel();
        expr_single_t child_fn = child->get_function<expr_single_t>();

        char *src_copy[N];
        memcpy(src_copy, src, sizeof(src_copy));
        for (intptr_t j = 0; j < N; ++j) {
          src_copy[j] += src_offset[j];
        }

        nh_start_stop->start = count[0];
        nh_start_stop->stop = nh_size; // min(nh_size, dst_size)
        for (intptr_t i = 0; i < count[0]; ++i) {
          child_fn(dst, src_copy, child);
          --(nh_start_stop->start);
          dst += dst_stride;
          for (intptr_t j = 0; j < N; ++j) {
            src_copy[j] += src_stride[j];
          }
        }
        //  *nh_start = 0;
        //    *nh_stop = nh_size;
        for (intptr_t i = 0; i < count[1]; ++i) {
          child_fn(dst, src_copy, child);
          dst += dst_stride;
          for (intptr_t j = 0; j < N; ++j) {
            src_copy[j] += src_stride[j];
          }
        }
        //      *nh_start = 0;
        //        *nh_stop = count[2]; // 0 if count[2] >
        for (intptr_t i = 0; i < count[2]; ++i) {
          --(nh_start_stop->stop);
          child_fn(dst, src_copy, child);
          dst += dst_stride;
          for (intptr_t j = 0; j < N; ++j) {
            src_copy[j] += src_stride[j];
          }
        }
      }

      static intptr_t
      instantiate(const arrfunc_type_data *af_self,
                  const ndt::arrfunc_type *DYND_UNUSED(af_tp),
                  const char *DYND_UNUSED(static_data),
                  size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
                  void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                  const char *dst_arrmeta, intptr_t nsrc,
                  const ndt::type *src_tp, const char *const *src_arrmeta,
                  kernel_request_t kernreq, const eval::eval_context *ectx,
                  const nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        std::shared_ptr<neighborhood_data> nh =
            *af_self->get_data_as<std::shared_ptr<neighborhood_data>>();
        nd::arrfunc nh_op = nh->op;

        nd::array shape;
        // TODO: Eliminate all try/catch(...)
        try {
          shape = kwds.p("shape").f("dereference");
        }
        catch (...) {
          const nd::array &mask = kwds.p("mask").f("dereference");
          shape = nd::array(mask.get_shape());
        }
        intptr_t ndim = shape.get_dim_size();

        nd::array offset;
        if (!kwds.p("offset").is_missing()) {
          offset = kwds.p("offset").f("dereference");
        }

        // Process the dst array striding/types
        const size_stride_t *dst_shape;
        ndt::type nh_dst_tp;
        const char *nh_dst_arrmeta;
        if (!dst_tp.get_as_strided(dst_arrmeta, ndim, &dst_shape, &nh_dst_tp,
                                   &nh_dst_arrmeta)) {
          std::stringstream ss;
          ss << "neighborhood arrfunc dst must be a strided array, not "
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
          ss << "neighborhood arrfunc argument 1 must be a 2D strided array, "
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

        ckb_offset = nh_op.get()->instantiate(
            nh_op.get(), nh_op.get_type(), NULL, 0, NULL, ckb, ckb_offset,
            nh_dst_tp, nh_dst_arrmeta, nsrc, nh_src_tp, nh_src_arrmeta,
            kernel_request_single, ectx,
            struct_concat(kwds, pack("start_stop", reinterpret_cast<intptr_t>(
                                                       nh->start_stop))),
            tp_vars);

        return ckb_offset;
      }

      static void resolve_dst_type(
          const ndt::arrfunc_type *self_tp, char *DYND_UNUSED(static_data),
          size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
          ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
          const ndt::type *src_tp, const nd::array &DYND_UNUSED(kwds),
          const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        // TODO: Should be able to express the match/subsitution without special
        // code

        // This is basically resolve() from arrfunc.hpp
        /*
          if (nsrc != af_tp->get_npos()) {
            std::stringstream ss;
            ss << "arrfunc expected " << af_tp->get_npos()
               << " parameters, but received " << nsrc;
            throw std::invalid_argument(ss.str());
          }
          const ndt::type *param_types = af_tp->get_pos_types_raw();
          std::map<nd::string, ndt::type> typevars;
          for (intptr_t i = 0; i != nsrc; ++i) {
            if (!ndt::pattern_match(src_tp[i].value_type(), param_types[i],
          typevars))
          {
              std::stringstream ss;
              ss << "parameter " << (i + 1) << " to arrfunc does not match, ";
              ss << "expected " << param_types[i] << ", received " << src_tp[i];
              throw std::invalid_argument(ss.str());
            }
          }
        */
        //  out_dst_tp = ndt::substitute(af_tp->get_return_type(), typevars,
        //  false);

        dst_tp = self_tp->get_return_type();

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