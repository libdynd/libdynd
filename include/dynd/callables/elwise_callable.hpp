//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <array>

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/elwise_kernel.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/dim_fragment_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * This defines the type and keyword argument resolution for
     * an elwise callable.
     */
    template <type_id_t DstTypeID, type_id_t SrcTypeID, size_t N>
    class elwise_callable;

    template <size_t N>
    class elwise_callable<fixed_dim_id, fixed_dim_id, N> : public base_callable {
      struct data_type {
        callable &child;
      };

      struct elwise_call_frame : call_frame {
        bool broadcast_dst;
        std::array<bool, N> broadcast_src;
      };

    public:
      elwise_callable() : base_callable(ndt::type()) {}

      callable &get_child(base_callable *parent);

      ndt::type resolve(base_callable *caller, char *data, call_graph &cg, const ndt::type &res_tp,
                        size_t DYND_UNUSED(narg), const ndt::type *arg_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        cg.emplace_back(this);

        callable &child = reinterpret_cast<data_type *>(data)->child;
        const ndt::type &child_ret_tp = child.get_ret_type();
        const std::vector<ndt::type> &child_arg_tp = child.get_arg_types();

        std::array<intptr_t, N> arg_size;
        std::array<intptr_t, N> arg_ndim;
        intptr_t max_ndim = 0;
        for (size_t i = 0; i < N; ++i) {
          arg_ndim[i] = arg_tp[i].get_ndim() - child_arg_tp[i].get_ndim();
          if (arg_ndim[i] == 0) {
            arg_size[i] = 1;
          } else {
            arg_size[i] = arg_tp[i].extended<ndt::fixed_dim_type>()->get_fixed_dim_size();
            if (arg_ndim[i] > max_ndim) {
              max_ndim = arg_ndim[i];
            }
          }
        }

        bool res_variadic = res_tp.is_variadic();
        intptr_t res_size;
        ndt::type res_element_tp;
        intptr_t ret_ndim = res_tp.get_ndim() - child_ret_tp.get_ndim();
        if (res_variadic) {
          res_size = 1;
          for (size_t i = 0; i < N && res_size == 1; ++i) {
            if (arg_ndim[i] == max_ndim) {
              res_size = arg_size[i];
            }
          }
          res_element_tp = res_tp;
        } else {
          if (ret_ndim > max_ndim) {
            max_ndim = ret_ndim;
          }
          res_size = res_tp.extended<ndt::fixed_dim_type>()->get_fixed_dim_size();
          res_element_tp = res_tp.extended<ndt::fixed_dim_type>()->get_element_type();
        }

        std::array<bool, N> arg_broadcast;
        std::array<ndt::type, N> arg_element_tp;
        bool callback = ret_ndim > 1;
        for (size_t i = 0; i < N; ++i) {
          if (arg_ndim[i] == max_ndim) {
            arg_broadcast[i] = false;
            if (res_size != arg_size[i] && arg_size[i] != 1) {
              throw std::runtime_error("broadcast error");
            }
            arg_element_tp[i] = arg_tp[i].extended<ndt::fixed_dim_type>()->get_element_type();
          } else {
            arg_broadcast[i] = true;
            arg_element_tp[i] = arg_tp[i];
          }
          if (arg_element_tp[i].get_ndim() != child_arg_tp[i].get_ndim()) {
            callback = true;
          }
        }

        if (callback) {
          return ndt::make_type<ndt::fixed_dim_type>(
              res_size,
              caller->resolve(this, nullptr, cg, res_element_tp, N, arg_element_tp.data(), nkwd, kwds, tp_vars));
        }

        return ndt::make_type<ndt::fixed_dim_type>(
            res_size, child->resolve(this, nullptr, cg, res_variadic ? child.get_ret_type() : res_element_tp, N,
                                     arg_element_tp.data(), nkwd, kwds, tp_vars));
      }

      void new_resolve(base_callable *parent, call_graph &cg, ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                       size_t nkwd, const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
        elwise_call_frame *data = reinterpret_cast<elwise_call_frame *>(cg.back());

        callable &child = get_child(parent);
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic() ||
            child_tp->get_return_type().get_id() == typevar_constructed_id) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        ndt::type child_dst_tp = dst_tp.extended<ndt::fixed_dim_type>()->get_element_type();
        std::array<ndt::type, N> child_src_tp;

        intptr_t size;
        size = dst_tp.extended<ndt::fixed_dim_type>()->get_fixed_dim_size();

        bool finished = dst_ndim == 1;
        for (size_t i = 0; i < N; ++i) {
          intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          intptr_t src_size = src_tp[i].extended<ndt::fixed_dim_type>()->get_fixed_dim_size();
          if (src_ndim < dst_ndim) {
            // This src value is getting broadcasted
            data->broadcast_src[i] = true;
            //            child_src_arrmeta[i] = src_arrmeta[i];
            child_src_tp[i] = src_tp[i];
            finished &= src_ndim == 0;
          } else {
            data->broadcast_src[i] = false;
            src_size = src_tp[i].extended<ndt::fixed_dim_type>()->get_fixed_dim_size();
            child_src_tp[i] = src_tp[i].extended<ndt::fixed_dim_type>()->get_element_type();
            if (src_size != 1 && size != src_size) {
              throw std::runtime_error("broadcast error");
            }

            finished &= src_ndim == 1;
          }
        }

        if (!finished) {
          if (!parent->is_abstract()) {
            cg.emplace_back(parent);
          }

          parent->new_resolve(this, cg, child_dst_tp, nsrc, child_src_tp.data(), nkwd, kwds, tp_vars);
        } else {
          if (!child->is_abstract()) {
            cg.emplace_back(child.get());
          }

          child->new_resolve(this, cg, child_dst_tp, nsrc, child_src_tp.data(), nkwd, kwds, tp_vars);
        }
      }

      void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                           const char *const *src_arrmeta, size_t nkwd, const array *kwds) {
        elwise_call_frame *data = reinterpret_cast<elwise_call_frame *>(frame);

        intptr_t size = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->dim_size;
        intptr_t dst_stride = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->stride;

        std::array<const char *, N> child_src_arrmeta;
        std::array<intptr_t, N> src_stride;
        for (size_t i = 0; i < N; ++i) {
          if (data->broadcast_src[i]) {
            src_stride[i] = 0;
            child_src_arrmeta[i] = src_arrmeta[i];
          } else {
            src_stride[i] = reinterpret_cast<const size_stride_t *>(src_arrmeta[i])->stride;
            child_src_arrmeta[i] = src_arrmeta[i] + sizeof(size_stride_t);
          }
        }

        ckb.emplace_back<elwise_kernel<fixed_dim_id, fixed_dim_id, N>>(kernreq, size, dst_stride, src_stride.data());

        frame = frame->next();
        frame->callee->new_instantiate(frame, ckb, kernel_request_strided, dst_arrmeta + sizeof(size_stride_t),
                                       child_src_arrmeta.data(), nkwd, kwds);
      }

      static void elwise_instantiate(callable &self, callable &child, char *data, kernel_builder *ckb,
                                     const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                     intptr_t nkwd, const nd::array *kwds,
                                     const std::map<std::string, ndt::type> &tp_vars) {
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic() ||
            child_tp->get_return_type().get_id() == typevar_constructed_id) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        std::array<const char *, N> child_src_arrmeta;
        ndt::type child_dst_tp;
        std::array<ndt::type, N> child_src_tp;

        intptr_t size, dst_stride;
        std::array<intptr_t, N> src_stride;
        if (!dst_tp.get_as_strided(dst_arrmeta, &size, &dst_stride, &child_dst_tp, &child_dst_arrmeta)) {
          std::stringstream ss;
          ss << "make_elwise_strided_dimension_expr_kernel: error processing "
                "type "
             << dst_tp << " as strided";
          throw type_error(ss.str());
        }

        bool finished = dst_ndim == 1;
        for (size_t i = 0; i < N; ++i) {
          intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          intptr_t src_size;
          if (src_ndim < dst_ndim) {
            // This src value is getting broadcasted
            src_stride[i] = 0;
            child_src_arrmeta[i] = src_arrmeta[i];
            child_src_tp[i] = src_tp[i];
            finished &= src_ndim == 0;
          } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size, &src_stride[i], &child_src_tp[i],
                                              &child_src_arrmeta[i])) {
            // Check for a broadcasting error
            if (src_size != 1 && size != src_size) {
              throw broadcast_error(dst_tp, dst_arrmeta, src_tp[i], src_arrmeta[i]);
            }
            finished &= src_ndim == 1;
          } else {
            std::stringstream ss;
            ss << "make_elwise_strided_dimension_expr_kernel: expected strided "
                  "or fixed dim, got "
               << src_tp[i];
            throw std::runtime_error(ss.str());
          }
        }

        ckb->emplace_back<elwise_kernel<fixed_dim_id, fixed_dim_id, N>>(kernreq, size, dst_stride, src_stride.data());

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return self->instantiate(data, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                   child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
        }

        // Instantiate the elementwise handler
        return child->instantiate(NULL, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                  child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
      }

      virtual void instantiate(char *DYND_UNUSED(data), kernel_builder *DYND_UNUSED(ckb),
                               const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                               intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                               const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t DYND_UNUSED(kernreq),
                               intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                               const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {}
    };

    // src is either fixed or var
    template <size_t N>
    class elwise_callable<fixed_dim_id, var_dim_id, N> : public base_callable {
    public:
      elwise_callable() : base_callable(ndt::type()) {}

      callable &get_child(base_callable *parent);

      ndt::type resolve(base_callable *caller, char *DYND_UNUSED(data), call_graph &cg, const ndt::type &res_tp,
                        size_t DYND_UNUSED(narg), const ndt::type *arg_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        cg.emplace_back(this);

        callable &child = get_child(caller);
        const ndt::type &child_ret_tp = child.get_ret_type();
        const std::vector<ndt::type> &child_arg_tp = child.get_arg_types();

        std::array<intptr_t, N> arg_size;
        std::array<intptr_t, N> arg_ndim;
        intptr_t max_ndim = 0;
        for (size_t i = 0; i < N; ++i) {
          arg_ndim[i] = arg_tp[i].get_ndim() - child_arg_tp[i].get_ndim();
          if (arg_ndim[i] == 0) {
            arg_size[i] = 1;
          } else {
            arg_size[i] = arg_tp[i].extended<ndt::base_dim_type>()->get_dim_size();
            if (arg_ndim[i] > max_ndim) {
              max_ndim = arg_ndim[i];
            }
          }
        }

        bool res_variadic = res_tp.is_variadic();
        intptr_t res_size;
        ndt::type res_element_tp;
        if (res_variadic) {
          res_size = 1;
          for (size_t i = 0; i < N && res_size == 1; ++i) {
            if (arg_ndim[i] == max_ndim && arg_size[i] != -1) {
              res_size = arg_size[i];
            }
          }
          res_element_tp = res_tp;
        } else {
          if (res_tp.get_ndim() - child_ret_tp.get_ndim() > max_ndim) {
            max_ndim = res_tp.get_ndim() - child_ret_tp.get_ndim();
          } else if (res_tp.get_ndim() - child_ret_tp.get_ndim() < max_ndim) {
            throw std::runtime_error("broadcast error");
          }
          res_size = res_tp.extended<ndt::base_dim_type>()->get_dim_size();
          res_element_tp = res_tp.extended<ndt::base_dim_type>()->get_element_type();
        }

        std::array<bool, N> arg_broadcast;
        std::array<ndt::type, N> arg_element_tp;
        bool callback = true;
        for (size_t i = 0; i < N; ++i) {
          if (arg_ndim[i] == max_ndim) {
            arg_broadcast[i] = false;
            if (arg_size[i] != -1 && res_size != -1 && res_size != arg_size[i] && arg_size[i] != 1) {
              throw std::runtime_error("broadcast error");
            }
            arg_element_tp[i] = arg_tp[i].extended<ndt::base_dim_type>()->get_element_type();
          } else {
            arg_broadcast[i] = true;
            arg_element_tp[i] = arg_tp[i];
          }
          if (arg_element_tp[i].get_ndim() != child_arg_tp[i].get_ndim()) {
            callback = true;
          }
        }

        if (callback) {
          if (res_size == 1) {
            return ndt::make_type<ndt::var_dim_type>(
                caller->resolve(this, nullptr, cg, res_element_tp, N, arg_element_tp.data(), nkwd, kwds, tp_vars));
          } else {
            return ndt::make_type<ndt::fixed_dim_type>(
                res_size,
                caller->resolve(this, nullptr, cg, res_element_tp, N, arg_element_tp.data(), nkwd, kwds, tp_vars));
          }
        }

        if (res_size == 1) {
          return ndt::make_type<ndt::var_dim_type>(child->resolve(this, nullptr, cg,
                                                                  res_variadic ? child.get_ret_type() : res_element_tp,
                                                                  N, arg_element_tp.data(), nkwd, kwds, tp_vars));
        } else {
          return ndt::make_type<ndt::fixed_dim_type>(
              res_size, child->resolve(this, nullptr, cg, res_variadic ? child.get_ret_type() : res_element_tp, N,
                                       arg_element_tp.data(), nkwd, kwds, tp_vars));
        }
      }

      static void elwise_instantiate(callable &self, callable &child, char *data, kernel_builder *ckb,
                                     const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                     intptr_t nkwd, const nd::array *kwds,
                                     const std::map<std::string, ndt::type> &tp_vars) {
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        std::array<const char *, N> child_src_arrmeta;
        ndt::type child_dst_tp;
        std::array<ndt::type, N> child_src_tp;

        intptr_t size, dst_stride;
        if (!dst_tp.get_as_strided(dst_arrmeta, &size, &dst_stride, &child_dst_tp, &child_dst_arrmeta)) {
          std::stringstream ss;
          ss << "make_elwise_strided_dimension_expr_kernel: error processing "
                "type "
             << dst_tp << " as strided";
          throw type_error(ss.str());
        }

        std::array<intptr_t, N> src_stride, src_offset;
        std::array<bool, N> is_src_var;
        bool finished = dst_ndim == 1;
        for (size_t i = 0; i < N; ++i) {
          intptr_t src_size;
          intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          // The src[i] strided parameters
          if (src_ndim < dst_ndim) {
            // This src value is getting broadcasted
            src_stride[i] = 0;
            src_offset[i] = 0;
            is_src_var[i] = false;
            child_src_arrmeta[i] = src_arrmeta[i];
            child_src_tp[i] = src_tp[i];
            finished &= src_ndim == 0;
          } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size, &src_stride[i], &child_src_tp[i],
                                              &child_src_arrmeta[i])) {
            // Check for a broadcasting error
            if (src_size != 1 && size != src_size) {
              throw broadcast_error(dst_tp, dst_arrmeta, src_tp[i], src_arrmeta[i]);
            }
            src_offset[i] = 0;
            is_src_var[i] = false;
            finished &= src_ndim == 1;
          } else {
            const ndt::var_dim_type *vdd = static_cast<const ndt::var_dim_type *>(src_tp[i].extended());
            const ndt::var_dim_type::metadata_type *src_md =
                reinterpret_cast<const ndt::var_dim_type::metadata_type *>(src_arrmeta[i]);
            src_stride[i] = src_md->stride;
            src_offset[i] = src_md->offset;
            is_src_var[i] = true;
            child_src_arrmeta[i] = src_arrmeta[i] + sizeof(ndt::var_dim_type::metadata_type);
            child_src_tp[i] = vdd->get_element_type();
            finished &= src_ndim == 1;
          }
        }

        ckb->emplace_back<elwise_kernel<fixed_dim_id, var_dim_id, N>>(kernreq, size, dst_stride, src_stride.data(),
                                                                      src_offset.data(), is_src_var.data());

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return self->instantiate(data, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                   child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
        }
        // Instantiate the elementwise handler
        return child->instantiate(NULL, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                  child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
      }

      virtual void instantiate(char *DYND_UNUSED(data), kernel_builder *DYND_UNUSED(ckb),
                               const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                               intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                               const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t DYND_UNUSED(kernreq),
                               intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                               const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {}
    };

    template <size_t N>
    class elwise_callable<var_dim_id, fixed_dim_id, N> : public base_callable {
      callable m_child;

    public:
      struct data_type {
        bool broadcast_dst;
        std::array<bool, N> broadcast_src;
        std::array<bool, N> is_src_var;
      };

      elwise_callable() : base_callable(ndt::type()) {}

      callable &get_child(base_callable *parent);

      ndt::type resolve(base_callable *caller, char *DYND_UNUSED(data), call_graph &cg, const ndt::type &res_tp,
                        size_t DYND_UNUSED(narg), const ndt::type *arg_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        cg.emplace_back(this);

        callable &child = get_child(caller);
        const ndt::type &child_ret_tp = child.get_ret_type();
        const std::vector<ndt::type> &child_arg_tp = child.get_arg_types();

        std::array<intptr_t, N> arg_size;
        std::array<intptr_t, N> arg_ndim;
        intptr_t max_ndim = 0;
        for (size_t i = 0; i < N; ++i) {
          arg_ndim[i] = arg_tp[i].get_ndim() - child_arg_tp[i].get_ndim();
          if (arg_ndim[i] == 0) {
            arg_size[i] = 1;
          } else {
            arg_size[i] = arg_tp[i].extended<ndt::base_dim_type>()->get_dim_size();
            if (arg_ndim[i] > max_ndim) {
              max_ndim = arg_ndim[i];
            }
          }
        }

        bool res_variadic = res_tp.is_variadic();
        intptr_t res_size;
        ndt::type res_element_tp;
        if (res_variadic) {
          res_size = 1;
          for (size_t i = 0; i < N && res_size == 1; ++i) {
            if (arg_ndim[i] == max_ndim && arg_size[i] != -1) {
              res_size = arg_size[i];
            }
          }
          res_element_tp = res_tp;
        } else {
          if (res_tp.get_ndim() - child_ret_tp.get_ndim() > max_ndim) {
            max_ndim = res_tp.get_ndim() - child_ret_tp.get_ndim();
          }
          res_size = res_tp.extended<ndt::base_dim_type>()->get_dim_size();
          res_element_tp = res_tp.extended<ndt::base_dim_type>()->get_element_type();
        }

        std::array<bool, N> arg_broadcast;
        std::array<ndt::type, N> arg_element_tp;
        bool callback = true;
        for (size_t i = 0; i < N; ++i) {
          if (arg_ndim[i] == max_ndim) {
            arg_broadcast[i] = false;
            if (arg_size[i] != -1 && res_size != -1 && res_size != arg_size[i] && arg_size[i] != 1) {
              throw std::runtime_error("broadcast error");
            }
            arg_element_tp[i] = arg_tp[i].extended<ndt::base_dim_type>()->get_element_type();
          } else {
            arg_broadcast[i] = true;
            arg_element_tp[i] = arg_tp[i];
          }
          if (arg_element_tp[i].get_ndim() != child_arg_tp[i].get_ndim()) {
            callback = true;
          }
        }

        if (callback) {
          return ndt::make_type<ndt::var_dim_type>(
              caller->resolve(this, nullptr, cg, res_element_tp, N, arg_element_tp.data(), nkwd, kwds, tp_vars));
        }

        return ndt::make_type<ndt::var_dim_type>(child->resolve(this, nullptr, cg,
                                                                res_variadic ? child.get_ret_type() : res_element_tp, N,
                                                                arg_element_tp.data(), nkwd, kwds, tp_vars));
      }

      elwise_callable(const callable &child) : base_callable(ndt::type()), m_child(child) {}

      /*
            void new_resolve(callable_graph &g, call_stack &stack, size_t nkwd, const array *kwds,
                             const std::map<std::string, ndt::type> &tp_vars)
            {
              data_type data;

              const ndt::type &dst_tp = stack.res_type();
              const ndt::type *src_tp = stack.arg_types();

              callable &child = m_child;

              const ndt::callable_type *child_tp = child.get_type();

              intptr_t dst_ndim = dst_tp.get_ndim();
              if (!child_tp->get_return_type().is_symbolic()) {
                dst_ndim -= child_tp->get_return_type().get_ndim();
              }

              ndt::type child_dst_tp;
              std::array<ndt::type, N> child_src_tp;

              // The dst var parameters
              const ndt::var_dim_type *dst_vdd = dst_tp.extended<ndt::var_dim_type>();

              child_dst_tp = dst_vdd->get_element_type();

              std::array<intptr_t, N> src_size;

              bool finished = dst_ndim == 1;
              for (size_t i = 0; i < N; ++i) {
                // The src[i] strided parameters
                intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
                if (src_ndim < dst_ndim) {
                  // This src value is getting broadcasted
                  src_size[i] = 1;
                  data.broadcast_src[i] = true;
                  data.is_src_var[i] = false;
                  child_src_tp[i] = src_tp[i];
                  finished &= src_ndim == 0;
                }
                else if (src_tp[i].get_id() == fixed_dim_id) { // src_tp[i].get_as_strided(src_arrmeta[i],
         &src_size[i],
                                                               // &src_stride[i], &child_src_tp[i],
                  //                      &child_src_arrmeta[i])) {
                  data.broadcast_src[i] = false;
                  data.is_src_var[i] = false;
                  finished &= src_ndim == 1;
                }
                else {
                  const ndt::var_dim_type *vdd = static_cast<const ndt::var_dim_type *>(src_tp[i].extended());
                  data.is_src_var[i] = true;
                  data.broadcast_src[i] = true;
                  child_src_tp[i] = vdd->get_element_type();
                  finished &= src_ndim == 1;
                }
              }

              std::array<intptr_t, N> src_arrmeta_offsets;
              for (size_t i = 0; i < N; ++i) {
                src_arrmeta_offsets[i] = stack.arg_metadata_offsets()[i];
                if (data.is_src_var[i]) {
                  src_arrmeta_offsets[i] += sizeof(ndt::var_dim_type::metadata_type);
                }
                else {
                  src_arrmeta_offsets[i] += sizeof(size_stride_t);
                }
              }

              stack.push_back_data(data);

              // If there are still dimensions to broadcast, recursively lift more
              if (!finished) {

                callable parent = stack.parent();
                stack.push_back(parent, child_dst_tp, stack.res_metadata_offset() +
         sizeof(ndt::var_dim_type::metadata_type),
                                stack.narg(), child_src_tp.data(), src_arrmeta_offsets.data(),
         kernel_request_strided);
                parent->new_resolve(g, stack, nkwd, kwds, tp_vars);
              }
              else {
                // All the types matched, so instantiate the elementwise handler

                stack.push_back(m_child, child_dst_tp, stack.res_metadata_offset() +
         sizeof(ndt::var_dim_type::metadata_type),
                                stack.narg(), child_src_tp.data(), src_arrmeta_offsets.data(),
         kernel_request_strided);

                m_child->new_resolve(g, stack, nkwd, kwds, tp_vars);
                //          return child->instantiate(NULL, ckb, child_dst_tp, child_dst_arrmeta, nsrc,
         child_src_tp.data(),
                //                                  child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds,
         tp_vars);
              }
            }
      */

      static void elwise_instantiate(callable &self, callable &child, char *data, kernel_builder *ckb,
                                     const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                     intptr_t nkwd, const nd::array *kwds,
                                     const std::map<std::string, ndt::type> &tp_vars) {
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        std::array<const char *, N> child_src_arrmeta;
        ndt::type child_dst_tp;
        std::array<ndt::type, N> child_src_tp;

        // The dst var parameters
        const ndt::var_dim_type *dst_vdd = dst_tp.extended<ndt::var_dim_type>();
        const ndt::var_dim_type::metadata_type *dst_md =
            reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta);

        child_dst_arrmeta = dst_arrmeta + sizeof(ndt::var_dim_type::metadata_type);
        child_dst_tp = dst_vdd->get_element_type();

        std::array<intptr_t, N> src_stride, src_offset, src_size;
        std::array<bool, N> is_src_var;

        bool finished = dst_ndim == 1;
        for (size_t i = 0; i < N; ++i) {
          // The src[i] strided parameters
          intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          if (src_ndim < dst_ndim) {
            // This src value is getting broadcasted
            src_stride[i] = 0;
            src_offset[i] = 0;
            src_size[i] = 1;
            is_src_var[i] = false;
            child_src_arrmeta[i] = src_arrmeta[i];
            child_src_tp[i] = src_tp[i];
            finished &= src_ndim == 0;
          } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size[i], &src_stride[i], &child_src_tp[i],
                                              &child_src_arrmeta[i])) {
            src_offset[i] = 0;
            is_src_var[i] = false;
            finished &= src_ndim == 1;
          } else {
            const ndt::var_dim_type *vdd = static_cast<const ndt::var_dim_type *>(src_tp[i].extended());
            const ndt::var_dim_type::metadata_type *src_md =
                reinterpret_cast<const ndt::var_dim_type::metadata_type *>(src_arrmeta[i]);
            src_stride[i] = src_md->stride;
            src_offset[i] = src_md->offset;
            is_src_var[i] = true;
            child_src_arrmeta[i] = src_arrmeta[i] + sizeof(ndt::var_dim_type::metadata_type);
            child_src_tp[i] = vdd->get_element_type();
            finished &= src_ndim == 1;
          }
        }

        ckb->emplace_back<elwise_kernel<var_dim_id, fixed_dim_id, N>>(
            kernreq, dst_md->blockref.get(), dst_vdd->get_target_alignment(), dst_md->stride, dst_md->offset,
            src_stride.data(), src_offset.data(), src_size.data(), is_src_var.data());

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return self->instantiate(data, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                   child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
        }
        // All the types matched, so instantiate the elementwise handler
        return child->instantiate(NULL, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                  child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
      }

      /*
            void new_instantiate(char *data, kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                                 const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc), const ndt::type
         *DYND_UNUSED(src_tp),
                                 const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                                 const array *DYND_UNUSED(kwds))
            {
              const ndt::var_dim_type::metadata_type *dst_md =
                  reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta);

              ckb->emplace_back<elwise_kernel<var_dim_id, fixed_dim_id, N>>(
                  kernreq, dst_md->blockref.get(), dst_vdd->get_target_alignment(), dst_md->stride, dst_md->offset,
                  src_stride.data(), src_offset.data(), src_size.data(), is_src_var.data());
            }
      */

      virtual void instantiate(char *DYND_UNUSED(data), kernel_builder *DYND_UNUSED(ckb),
                               const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                               intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                               const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t DYND_UNUSED(kernreq),
                               intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                               const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {}

      /*
            virtual void new_instantiate(char *data, kernel_builder *ckb, const ndt::type &dst_tp, const char
         *dst_arrmeta,
                                         intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                         const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t
         DYND_UNUSED(nkwd),
                                         const array *DYND_UNUSED(kwds))
            {
              const ndt::var_dim_type::metadata_type *dst_md =
                  reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta);
              const ndt::var_dim_type *dst_vdd = dst_tp.extended<ndt::var_dim_type>();

              std::array<intptr_t, N> src_stride;
              std::array<intptr_t, N> src_offset;
              std::array<intptr_t, N> src_size;
              for (size_t i = 0; i < N; ++i) {
                if (reinterpret_cast<data_type *>(data)->broadcast_src[i] &&
                    !reinterpret_cast<data_type *>(data)->is_src_var[i]) {
                  src_stride[i] = 0;
                  src_offset[i] = 0;
                  src_size[i] = 1;
                }
                else if (reinterpret_cast<data_type *>(data)->is_src_var[i]) {
                  const ndt::var_dim_type::metadata_type *src_md =
                      reinterpret_cast<const ndt::var_dim_type::metadata_type *>(src_arrmeta[i]);

                  src_stride[i] = src_md->stride;
                  src_offset[i] = src_md->offset;
                }
                else {
                  src_stride[i] = reinterpret_cast<const size_stride_t *>(src_arrmeta[i])->stride;
                  src_offset[i] = 0;
                }
              }

              ckb->emplace_back<elwise_kernel<var_dim_id, fixed_dim_id, N>>(
                  kernreq, dst_md->blockref.get(), dst_vdd->get_target_alignment(), dst_md->stride, dst_md->offset,
                  src_stride.data(), src_offset.data(), src_size.data(),
                  reinterpret_cast<data_type *>(data)->is_src_var.data());
            }
      */
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
