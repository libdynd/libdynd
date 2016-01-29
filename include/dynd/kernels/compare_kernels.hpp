//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/option.hpp>
#include <dynd/kernels/less_kernel.hpp>
#include <dynd/kernels/less_equal_kernel.hpp>
#include <dynd/kernels/equal_kernel.hpp>
#include <dynd/kernels/not_equal_kernel.hpp>
#include <dynd/kernels/greater_equal_kernel.hpp>
#include <dynd/kernels/greater_kernel.hpp>
#include <dynd/kernels/total_order_kernel.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/type.hpp>

namespace dynd {
namespace nd {

  template <typename FuncType, bool Src0IsOption, bool Src1IsOption>
  struct option_comparison_kernel;

  template <typename FuncType>
  struct option_comparison_kernel<FuncType, true, false>
      : base_kernel<option_comparison_kernel<FuncType, true, false>, 2> {
    intptr_t comp_offset;
    intptr_t assign_na_offset;

    void single(char *dst, char *const *src)
    {
      auto is_missing = this->get_child();
      bool1 child_dst;
      is_missing->single(reinterpret_cast<char *>(&child_dst), &src[0]);
      if (!child_dst) {
        this->get_child(comp_offset)->single(dst, src);
      }
      else {
        this->get_child(assign_na_offset)->single(dst, nullptr);
      }
    }

    static void instantiate(char *DYND_UNUSED(static_data), char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                            const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                            const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                            const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t ckb_offset = ckb->m_size;
      intptr_t option_comp_offset = ckb_offset;
      ckb->emplace_back<option_comparison_kernel>(kernreq);
      ckb_offset = ckb->m_size;

      auto is_missing = is_missing::get();
      is_missing.get()->instantiate(is_missing.get()->static_data(), data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp,
                                    src_arrmeta, kernel_request_single, nkwd, kwds, tp_vars);
      ckb_offset = ckb->m_size;
      option_comparison_kernel *self = ckb->get_at<option_comparison_kernel>(option_comp_offset);
      self->comp_offset = ckb_offset - option_comp_offset;
      auto cmp = FuncType::get();
      const ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::option_type>()->get_value_type(), src_tp[1]};
      cmp.get()->instantiate(cmp.get()->static_data(), data, ckb, dst_tp.extended<ndt::option_type>()->get_value_type(),
                             dst_arrmeta, nsrc, child_src_tp, src_arrmeta, kernel_request_single, nkwd, kwds, tp_vars);
      ckb_offset = ckb->m_size;
      self = ckb->get_at<option_comparison_kernel>(option_comp_offset);
      self->assign_na_offset = ckb_offset - option_comp_offset;
      auto assign_na = nd::assign_na::get();
      assign_na.get()->instantiate(assign_na.get()->static_data(), data, ckb,
                                   ndt::make_type<ndt::option_type>(ndt::make_type<bool1>()), nullptr, 0, nullptr,
                                   nullptr, kernel_request_single, nkwd, kwds, tp_vars);
      ckb_offset = ckb->m_size;
    }
  };

  template <typename FuncType>
  struct option_comparison_kernel<FuncType, false, true>
      : base_kernel<option_comparison_kernel<FuncType, false, true>, 2> {
    intptr_t comp_offset;
    intptr_t assign_na_offset;

    void single(char *dst, char *const *src)
    {
      auto is_missing = this->get_child();
      bool1 child_dst;
      is_missing->single(reinterpret_cast<char *>(&child_dst), &src[1]);
      if (!child_dst) {
        this->get_child(comp_offset)->single(dst, src);
      }
      else {
        this->get_child(assign_na_offset)->single(dst, nullptr);
      }
    }

    static void instantiate(char *DYND_UNUSED(static_data), char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                            const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                            const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                            const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t ckb_offset = ckb->m_size;
      intptr_t option_comp_offset = ckb_offset;
      ckb->emplace_back<option_comparison_kernel>(kernreq);
      ckb_offset = ckb->m_size;

      auto is_missing = is_missing::get();
      is_missing.get()->instantiate(is_missing.get()->static_data(), data, ckb, dst_tp, dst_arrmeta, nsrc, &src_tp[1],
                                    &src_arrmeta[1], kernel_request_single, nkwd, kwds, tp_vars);
      ckb_offset = ckb->m_size;
      option_comparison_kernel *self = ckb->get_at<option_comparison_kernel>(option_comp_offset);
      self->comp_offset = ckb_offset - option_comp_offset;
      auto cmp = FuncType::get();
      const ndt::type child_src_tp[2] = {
          src_tp[0], src_tp[1].extended<ndt::option_type>()->get_value_type(),
      };
      cmp.get()->instantiate(cmp.get()->static_data(), data, ckb, dst_tp.extended<ndt::option_type>()->get_value_type(),
                             dst_arrmeta, nsrc, child_src_tp, src_arrmeta, kernel_request_single, nkwd, kwds, tp_vars);
      ckb_offset = ckb->m_size;
      self = ckb->get_at<option_comparison_kernel>(option_comp_offset);
      self->assign_na_offset = ckb_offset - option_comp_offset;
      auto assign_na = nd::assign_na::get();
      assign_na.get()->instantiate(assign_na.get()->static_data(), data, ckb,
                                   ndt::make_type<ndt::option_type>(ndt::make_type<bool1>()), nullptr, 0, nullptr,
                                   nullptr, kernel_request_single, nkwd, kwds, tp_vars);
      ckb_offset = ckb->m_size;
    }
  };

  template <typename FuncType>
  struct option_comparison_kernel<FuncType, true, true>
      : base_kernel<option_comparison_kernel<FuncType, true, true>, 2> {
    intptr_t is_missing_rhs_offset;
    intptr_t comp_offset;
    intptr_t assign_na_offset;

    void single(char *dst, char *const *src)
    {
      auto is_missing_lhs = this->get_child();
      auto is_missing_rhs = this->get_child(is_missing_rhs_offset);
      bool child_dst_lhs;
      bool child_dst_rhs;
      is_missing_lhs->single(reinterpret_cast<char *>(&child_dst_lhs), &src[0]);
      is_missing_rhs->single(reinterpret_cast<char *>(&child_dst_rhs), &src[1]);
      if (!child_dst_lhs && !child_dst_rhs) {
        this->get_child(comp_offset)->single(dst, src);
      }
      else {
        this->get_child(assign_na_offset)->single(dst, nullptr);
      }
    }

    static void instantiate(char *DYND_UNUSED(static_data), char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                            const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                            const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                            const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t ckb_offset = ckb->m_size;
      intptr_t option_comp_offset = ckb_offset;
      ckb->emplace_back<option_comparison_kernel>(kernreq);
      ckb_offset = ckb->m_size;

      auto is_missing_lhs = is_missing::get();
      is_missing_lhs.get()->instantiate(is_missing_lhs.get()->static_data(), data, ckb, dst_tp, dst_arrmeta, nsrc,
                                        &src_tp[0], &src_arrmeta[0], kernel_request_single, nkwd, kwds, tp_vars);
      ckb_offset = ckb->m_size;
      option_comparison_kernel *self = ckb->get_at<option_comparison_kernel>(option_comp_offset);
      self->is_missing_rhs_offset = ckb_offset - option_comp_offset;

      auto is_missing_rhs = is_missing::get();
      is_missing_rhs.get()->instantiate(is_missing_rhs.get()->static_data(), data, ckb, dst_tp, dst_arrmeta, nsrc,
                                        &src_tp[1], &src_arrmeta[1], kernel_request_single, nkwd, kwds, tp_vars);
      ckb_offset = ckb->m_size;
      self = ckb->get_at<option_comparison_kernel>(option_comp_offset);
      self->comp_offset = ckb_offset - option_comp_offset;
      auto cmp = FuncType::get();
      const ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::option_type>()->get_value_type(),
                                         src_tp[1].extended<ndt::option_type>()->get_value_type()};
      cmp.get()->instantiate(cmp.get()->static_data(), data, ckb, dst_tp.extended<ndt::option_type>()->get_value_type(),
                             dst_arrmeta, nsrc, child_src_tp, src_arrmeta, kernel_request_single, nkwd, kwds, tp_vars);
      ckb_offset = ckb->m_size;
      self = ckb->get_at<option_comparison_kernel>(option_comp_offset);
      self->assign_na_offset = ckb_offset - option_comp_offset;
      auto assign_na = nd::assign_na::get();
      assign_na.get()->instantiate(assign_na.get()->static_data(), data, ckb,
                                   ndt::make_type<ndt::option_type>(ndt::make_type<bool1>()), nullptr, 0, nullptr,
                                   nullptr, kernel_request_single, nkwd, kwds, tp_vars);
      ckb_offset = ckb->m_size;
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <typename FuncType>
  struct traits<nd::option_comparison_kernel<FuncType, true, false>> {
    static type equivalent() { return type("(?Scalar, Scalar) -> ?bool"); }
  };

  template <typename FuncType>
  struct traits<nd::option_comparison_kernel<FuncType, false, true>> {
    static type equivalent() { return type("(Scalar, ?Scalar) -> ?bool"); }
  };

  template <typename FuncType>
  struct traits<nd::option_comparison_kernel<FuncType, true, true>> {
    static type equivalent() { return type("(?Scalar, ?Scalar) -> ?bool"); }
  };

} // namespace dynd::ndt
} // namespace dynd
