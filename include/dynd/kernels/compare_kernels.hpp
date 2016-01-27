//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/option.hpp>
#include <dynd/kernels/total_order_kernel.hpp>
#include <dynd/kernels/tuple_comparison_kernels.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/type.hpp>

namespace dynd {
namespace nd {

  template <type_id_t I0, type_id_t I1>
  struct less_kernel : base_kernel<less_kernel<I0, I1>, 2> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) < static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct less_kernel<I0, I0> : base_kernel<less_kernel<I0, I0>, 2> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<A0 *>(src[0]) < *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct less_equal_kernel : base_kernel<less_equal_kernel<I0, I1>, 2> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) <= static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct less_equal_kernel<I0, I0> : base_kernel<less_equal_kernel<I0, I0>, 2> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<A0 *>(src[0]) <= *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct equal_kernel : base_kernel<equal_kernel<I0, I1>, 2> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) == static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct equal_kernel<I0, I0> : base_kernel<equal_kernel<I0, I0>, 2> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<A0 *>(src[0]) == *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <>
  struct equal_kernel<tuple_type_id, tuple_type_id> : base_kernel<equal_kernel<tuple_type_id, tuple_type_id>, 2> {
    typedef equal_kernel extra_type;

    size_t field_count;
    const size_t *src0_data_offsets, *src1_data_offsets;
    // After this are field_count sorting_less kernel offsets, for
    // src0.field_i <op> src1.field_i
    // with each 0 <= i < field_count

    equal_kernel(size_t field_count, const size_t *src0_data_offsets, const size_t *src1_data_offsets)
        : field_count(field_count), src0_data_offsets(src0_data_offsets), src1_data_offsets(src1_data_offsets)
    {
    }

    void single(char *dst, char *const *src)
    {
      const size_t *kernel_offsets = reinterpret_cast<const size_t *>(this + 1);
      char *child_src[2];
      for (size_t i = 0; i != field_count; ++i) {
        ckernel_prefix *echild = reinterpret_cast<ckernel_prefix *>(reinterpret_cast<char *>(this) + kernel_offsets[i]);
        kernel_single_t opchild = echild->get_function<kernel_single_t>();
        // if (src0.field_i < src1.field_i) return true
        child_src[0] = src[0] + src0_data_offsets[i];
        child_src[1] = src[1] + src1_data_offsets[i];
        bool1 child_dst;
        opchild(echild, reinterpret_cast<char *>(&child_dst), child_src);
        if (!child_dst) {
          *reinterpret_cast<bool1 *>(dst) = false;
          return;
        }
      }
      *reinterpret_cast<bool1 *>(dst) = true;
    }

    static void destruct(ckernel_prefix *self)
    {
      extra_type *e = reinterpret_cast<extra_type *>(self);
      const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e + 1);
      size_t field_count = e->field_count;
      for (size_t i = 0; i != field_count; ++i) {
        self->get_child(kernel_offsets[i])->destroy();
      }
    }

    static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                            const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                            intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                            kernel_request_t DYND_UNUSED(kernreq), intptr_t DYND_UNUSED(nkwd),
                            const nd::array *DYND_UNUSED(kwds),
                            const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars));
  };

  template <type_id_t I0, type_id_t I1>
  struct not_equal_kernel : base_kernel<not_equal_kernel<I0, I1>, 2> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) != static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct not_equal_kernel<I0, I0> : base_kernel<not_equal_kernel<I0, I0>, 2> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<A0 *>(src[0]) != *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <>
  struct not_equal_kernel<tuple_type_id, tuple_type_id>
      : base_kernel<not_equal_kernel<tuple_type_id, tuple_type_id>, 2> {
    typedef not_equal_kernel extra_type;

    size_t field_count;
    const size_t *src0_data_offsets, *src1_data_offsets;
    // After this are field_count sorting_less kernel offsets, for
    // src0.field_i <op> src1.field_i
    // with each 0 <= i < field_count

    not_equal_kernel(size_t field_count, const size_t *src0_data_offsets, const size_t *src1_data_offsets)
        : field_count(field_count), src0_data_offsets(src0_data_offsets), src1_data_offsets(src1_data_offsets)
    {
    }

    void single(char *dst, char *const *src)
    {
      const size_t *kernel_offsets = reinterpret_cast<const size_t *>(this + 1);
      char *child_src[2];
      for (size_t i = 0; i != field_count; ++i) {
        ckernel_prefix *echild = reinterpret_cast<ckernel_prefix *>(reinterpret_cast<char *>(this) + kernel_offsets[i]);
        kernel_single_t opchild = echild->get_function<kernel_single_t>();
        // if (src0.field_i < src1.field_i) return true
        child_src[0] = src[0] + src0_data_offsets[i];
        child_src[1] = src[1] + src1_data_offsets[i];
        bool1 child_dst;
        opchild(echild, reinterpret_cast<char *>(&child_dst), child_src);
        if (child_dst) {
          *reinterpret_cast<bool1 *>(dst) = true;
          return;
        }
      }
      *reinterpret_cast<bool1 *>(dst) = false;
    }

    static void destruct(ckernel_prefix *self)
    {
      extra_type *e = reinterpret_cast<extra_type *>(self);
      const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e + 1);
      size_t field_count = e->field_count;
      for (size_t i = 0; i != field_count; ++i) {
        self->get_child(kernel_offsets[i])->destroy();
      }
    }

    static void instantiate(char *static_data, char *DYND_UNUSED(data), kernel_builder *ckb,
                            const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                            intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                            kernel_request_t DYND_UNUSED(kernreq), intptr_t DYND_UNUSED(nkwd),
                            const nd::array *DYND_UNUSED(kwds),
                            const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars));
  };

  template <type_id_t I0, type_id_t I1>
  struct greater_equal_kernel : base_kernel<greater_equal_kernel<I0, I1>, 2> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) >= static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct greater_equal_kernel<I0, I0> : base_kernel<greater_equal_kernel<I0, I0>, 2> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<A0 *>(src[0]) >= *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct greater_kernel : base_kernel<greater_kernel<I0, I1>, 2> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) > static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct greater_kernel<I0, I0> : base_kernel<greater_kernel<I0, I0>, 2> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<A0 *>(src[0]) > *reinterpret_cast<A0 *>(src[1]);
    }
  };

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

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct traits<nd::less_kernel<Src0TypeID, Src1TypeID>> {
    static type equivalent() { return callable_type::make(make_type<bool1>(), {type(Src0TypeID), type(Src1TypeID)}); }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct traits<nd::less_equal_kernel<Src0TypeID, Src1TypeID>> {
    static type equivalent() { return callable_type::make(make_type<bool1>(), {type(Src0TypeID), type(Src1TypeID)}); }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct traits<nd::equal_kernel<Src0TypeID, Src1TypeID>> {
    static type equivalent() { return callable_type::make(make_type<bool1>(), {type(Src0TypeID), type(Src1TypeID)}); }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct traits<nd::not_equal_kernel<Src0TypeID, Src1TypeID>> {
    static type equivalent() { return callable_type::make(make_type<bool1>(), {type(Src0TypeID), type(Src1TypeID)}); }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct traits<nd::greater_equal_kernel<Src0TypeID, Src1TypeID>> {
    static type equivalent() { return callable_type::make(make_type<bool1>(), {type(Src0TypeID), type(Src1TypeID)}); }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct traits<nd::greater_kernel<Src0TypeID, Src1TypeID>> {
    static type equivalent() { return callable_type::make(make_type<bool1>(), {type(Src0TypeID), type(Src1TypeID)}); }
  };

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
