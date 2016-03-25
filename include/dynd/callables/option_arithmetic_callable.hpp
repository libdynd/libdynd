//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_instantiable_callable.hpp>
#include <dynd/kernels/arithmetic.hpp>

namespace dynd {
namespace nd {

  template <typename FuncType, bool Src0IsOption, bool Src1IsOption>
  class option_arithmetic_callable;

  template <typename FuncType>
  class option_arithmetic_callable<FuncType, true, false> : public base_callable {
  public:
    option_arithmetic_callable() : base_callable(ndt::type("(?Scalar, Scalar) -> ?Scalar")) {}

    void resolve_dst_type(char *data, ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                          const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      auto k = FuncType::get().get();
      const ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::option_type>()->get_value_type(), src_tp[1]};
      k->resolve_dst_type(data, dst_tp, nsrc, child_src_tp, nkwd, kwds, tp_vars);
      dst_tp = ndt::make_type<ndt::option_type>(dst_tp);
    }

    void instantiate(char *data, kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                     const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t ckb_offset = ckb->size();
      intptr_t option_arith_offset = ckb_offset;
      ckb->emplace_back<option_arithmetic_kernel<FuncType, true, false>>(kernreq);
      ckb_offset = ckb->size();

      auto is_na = is_na::get();
      is_na.get()->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernel_request_single, nkwd,
                               kwds, tp_vars);
      ckb_offset = ckb->size();
      option_arithmetic_kernel<FuncType, true, false> *self =
          ckb->get_at<option_arithmetic_kernel<FuncType, true, false>>(option_arith_offset);
      self->arith_offset = ckb_offset - option_arith_offset;
      auto arith = FuncType::get();
      const ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::option_type>()->get_value_type(), src_tp[1]};
      arith.get()->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, child_src_tp, src_arrmeta, kernel_request_single,
                               nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
      self = ckb->get_at<option_arithmetic_kernel<FuncType, true, false>>(option_arith_offset);
      self->assign_na_offset = ckb_offset - option_arith_offset;
      auto assign_na = nd::assign_na::get();
      assign_na.get()->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, child_src_tp, src_arrmeta,
                                   kernel_request_single, nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
    }
  };

  template <typename FuncType>
  class option_arithmetic_callable<FuncType, false, true> : public base_callable {
  public:
    option_arithmetic_callable() : base_callable(ndt::type("(Scalar, ?Scalar) -> ?Scalar")) {}

    void resolve_dst_type(char *data, ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                          const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      auto k = FuncType::get().get();
      const ndt::type child_src_tp[2] = {src_tp[0], src_tp[1].extended<ndt::option_type>()->get_value_type()};
      k->resolve_dst_type(data, dst_tp, nsrc, child_src_tp, nkwd, kwds, tp_vars);
      dst_tp = ndt::make_type<ndt::option_type>(dst_tp);
    }

    void instantiate(char *data, kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                     const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t ckb_offset = ckb->size();
      intptr_t option_arith_offset = ckb_offset;
      ckb->emplace_back<option_arithmetic_kernel<FuncType, false, true>>(kernreq);
      ckb_offset = ckb->size();

      auto is_na = is_na::get();
      is_na.get()->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, &src_tp[1], &src_arrmeta[1], kernel_request_single,
                               nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
      option_arithmetic_kernel<FuncType, false, true> *self =
          ckb->get_at<option_arithmetic_kernel<FuncType, false, true>>(option_arith_offset);
      self->arith_offset = ckb_offset - option_arith_offset;
      auto arith = FuncType::get();
      const ndt::type child_src_tp[2] = {src_tp[0], src_tp[1].extended<ndt::option_type>()->get_value_type()};
      arith.get()->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, child_src_tp, src_arrmeta, kernel_request_single,
                               nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
      self = ckb->get_at<option_arithmetic_kernel<FuncType, false, true>>(option_arith_offset);
      self->assign_na_offset = ckb_offset - option_arith_offset;
      auto assign_na = nd::assign_na::get();
      assign_na.get()->instantiate(data, ckb, src_tp[1], src_arrmeta[1], 0, nullptr, nullptr, kernel_request_single,
                                   nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
    }
  };

  template <typename FuncType>
  class option_arithmetic_callable<FuncType, true, true> : public base_callable {
  public:
    option_arithmetic_callable() : base_callable(ndt::type("(?Scalar, ?Scalar) -> ?Scalar")) {}

    void resolve_dst_type(char *data, ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                          const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      auto k = FuncType::get().get();
      const ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::option_type>()->get_value_type(),
                                         src_tp[1].extended<ndt::option_type>()->get_value_type()};
      k->resolve_dst_type(data, dst_tp, nsrc, child_src_tp, nkwd, kwds, tp_vars);
      dst_tp = ndt::make_type<ndt::option_type>(dst_tp);
    }

    void instantiate(char *data, kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                     const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t ckb_offset = ckb->size();
      intptr_t option_arith_offset = ckb_offset;
      ckb->emplace_back<option_arithmetic_kernel<FuncType, true, true>>(kernreq);
      ckb_offset = ckb->size();

      auto is_na_lhs = is_na::get();
      is_na_lhs.get()->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernel_request_single,
                                   nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
      option_arithmetic_kernel<FuncType, true, true> *self =
          ckb->get_at<option_arithmetic_kernel<FuncType, true, true>>(option_arith_offset);
      self->is_na_rhs_offset = ckb_offset - option_arith_offset;
      auto is_na_rhs = is_na::get();
      is_na_rhs.get()->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernel_request_single,
                                   nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
      self = ckb->get_at<option_arithmetic_kernel<FuncType, true, true>>(option_arith_offset);
      self->arith_offset = ckb_offset - option_arith_offset;
      auto arith = FuncType::get();
      const ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::option_type>()->get_value_type(),
                                         src_tp[1].extended<ndt::option_type>()->get_value_type()};
      arith.get()->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, child_src_tp, src_arrmeta, kernel_request_single,
                               nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
      self = ckb->get_at<option_arithmetic_kernel<FuncType, true, true>>(option_arith_offset);
      self->assign_na_offset = ckb_offset - option_arith_offset;
      auto assign_na = nd::assign_na::get();
      assign_na.get()->instantiate(data, ckb, dst_tp, dst_arrmeta, 0, nullptr, nullptr, kernel_request_single, nkwd,
                                   kwds, tp_vars);
      ckb_offset = ckb->size();
    }
  };

} // namespace dynd::nd
} // namespace dynd
