//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/arithmetic.hpp>

namespace dynd {
namespace nd {

  template <callable &Callable, bool Src0IsOption, bool Src1IsOption>
  class option_arithmetic_callable;

  template <callable &Callable>
  class option_arithmetic_callable<Callable, true, false> : public base_callable {
  public:
    option_arithmetic_callable() : base_callable(ndt::type("(?Scalar, Scalar) -> ?Scalar")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), call_graph &cg, const ndt::type &dst_tp,
                      size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back(this);
      return dst_tp;
    }

    void resolve_dst_type(char *data, ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                          const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      auto k = Callable.get();
      const ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::option_type>()->get_value_type(), src_tp[1]};
      k->resolve_dst_type(data, dst_tp, nsrc, child_src_tp, nkwd, kwds, tp_vars);
      dst_tp = ndt::make_type<ndt::option_type>(dst_tp);
    }

    void instantiate(char *data, kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                     const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      intptr_t ckb_offset = ckb->size();
      intptr_t option_arith_offset = ckb_offset;
      ckb->emplace_back<option_arithmetic_kernel<true, false>>(kernreq);
      ckb_offset = ckb->size();

      is_na->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernel_request_single, nkwd, kwds,
                         tp_vars);
      ckb_offset = ckb->size();
      option_arithmetic_kernel<true, false> *self =
          ckb->get_at<option_arithmetic_kernel<true, false>>(option_arith_offset);
      self->arith_offset = ckb_offset - option_arith_offset;
      const ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::option_type>()->get_value_type(), src_tp[1]};
      Callable->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, child_src_tp, src_arrmeta, kernel_request_single,
                            nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
      self = ckb->get_at<option_arithmetic_kernel<true, false>>(option_arith_offset);
      self->assign_na_offset = ckb_offset - option_arith_offset;
      assign_na->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, child_src_tp, src_arrmeta, kernel_request_single,
                             nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
    }
  };

  template <callable &Callable>
  class option_arithmetic_callable<Callable, false, true> : public base_callable {
  public:
    option_arithmetic_callable() : base_callable(ndt::type("(Scalar, ?Scalar) -> ?Scalar")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), call_graph &cg, const ndt::type &dst_tp,
                      size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back(this);
      return dst_tp;
    }

    void resolve_dst_type(char *data, ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                          const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      auto k = Callable.get();
      const ndt::type child_src_tp[2] = {src_tp[0], src_tp[1].extended<ndt::option_type>()->get_value_type()};
      k->resolve_dst_type(data, dst_tp, nsrc, child_src_tp, nkwd, kwds, tp_vars);
      dst_tp = ndt::make_type<ndt::option_type>(dst_tp);
    }

    void instantiate(char *data, kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                     const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      intptr_t ckb_offset = ckb->size();
      intptr_t option_arith_offset = ckb_offset;
      ckb->emplace_back<option_arithmetic_kernel<false, true>>(kernreq);
      ckb_offset = ckb->size();

      is_na->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, &src_tp[1], &src_arrmeta[1], kernel_request_single, nkwd,
                         kwds, tp_vars);
      ckb_offset = ckb->size();
      option_arithmetic_kernel<false, true> *self =
          ckb->get_at<option_arithmetic_kernel<false, true>>(option_arith_offset);
      self->arith_offset = ckb_offset - option_arith_offset;
      auto arith = Callable;
      const ndt::type child_src_tp[2] = {src_tp[0], src_tp[1].extended<ndt::option_type>()->get_value_type()};
      arith.get()->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, child_src_tp, src_arrmeta, kernel_request_single,
                               nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
      self = ckb->get_at<option_arithmetic_kernel<false, true>>(option_arith_offset);
      self->assign_na_offset = ckb_offset - option_arith_offset;
      assign_na->instantiate(data, ckb, src_tp[1], src_arrmeta[1], 0, nullptr, nullptr, kernel_request_single, nkwd,
                             kwds, tp_vars);
      ckb_offset = ckb->size();
    }
  };

  template <callable &Callable>
  class option_arithmetic_callable<Callable, true, true> : public base_callable {
  public:
    option_arithmetic_callable() : base_callable(ndt::type("(?Scalar, ?Scalar) -> ?Scalar")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), call_graph &cg, const ndt::type &dst_tp,
                      size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back(this);
      return dst_tp;
    }

    void resolve_dst_type(char *data, ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                          const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      auto k = Callable.get();
      const ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::option_type>()->get_value_type(),
                                         src_tp[1].extended<ndt::option_type>()->get_value_type()};
      k->resolve_dst_type(data, dst_tp, nsrc, child_src_tp, nkwd, kwds, tp_vars);
      dst_tp = ndt::make_type<ndt::option_type>(dst_tp);
    }

    void instantiate(char *data, kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                     const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      intptr_t ckb_offset = ckb->size();
      intptr_t option_arith_offset = ckb_offset;
      ckb->emplace_back<option_arithmetic_kernel<true, true>>(kernreq);
      ckb_offset = ckb->size();

      is_na->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernel_request_single, nkwd, kwds,
                         tp_vars);
      ckb_offset = ckb->size();
      option_arithmetic_kernel<true, true> *self =
          ckb->get_at<option_arithmetic_kernel<true, true>>(option_arith_offset);
      self->is_na_rhs_offset = ckb_offset - option_arith_offset;
      is_na->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernel_request_single, nkwd, kwds,
                         tp_vars);
      ckb_offset = ckb->size();
      self = ckb->get_at<option_arithmetic_kernel<true, true>>(option_arith_offset);
      self->arith_offset = ckb_offset - option_arith_offset;
      auto arith = Callable;
      const ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::option_type>()->get_value_type(),
                                         src_tp[1].extended<ndt::option_type>()->get_value_type()};
      arith.get()->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, child_src_tp, src_arrmeta, kernel_request_single,
                               nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
      self = ckb->get_at<option_arithmetic_kernel<true, true>>(option_arith_offset);
      self->assign_na_offset = ckb_offset - option_arith_offset;
      assign_na->instantiate(data, ckb, dst_tp, dst_arrmeta, 0, nullptr, nullptr, kernel_request_single, nkwd, kwds,
                             tp_vars);
      ckb_offset = ckb->size();
    }
  };

} // namespace dynd::nd
} // namespace dynd
