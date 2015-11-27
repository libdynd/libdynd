//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/total_order_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>
#include <dynd/kernels/tuple_comparison_kernels.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/func/option.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/type.hpp>

namespace dynd {
namespace nd {

  template <typename K>
  struct base_comparison_kernel;

  template <template <type_id_t, type_id_t> class K, type_id_t I0, type_id_t I1>
  struct base_comparison_kernel<K<I0, I1>> : base_kernel<K<I0, I1>, 2> {
    static const std::size_t data_size = 0;
  };

  template <type_id_t I0, type_id_t I1>
  struct less_kernel : base_comparison_kernel<less_kernel<I0, I1>> {
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
  struct less_kernel<I0, I0> : base_comparison_kernel<less_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<A0 *>(src[0]) < *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct less_equal_kernel : base_comparison_kernel<less_equal_kernel<I0, I1>> {
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
  struct less_equal_kernel<I0, I0> : base_comparison_kernel<less_equal_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<A0 *>(src[0]) <= *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct equal_kernel : base_comparison_kernel<equal_kernel<I0, I1>> {
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
  struct equal_kernel<I0, I0> : base_comparison_kernel<equal_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<A0 *>(src[0]) == *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <>
  struct equal_kernel<tuple_type_id,
                      tuple_type_id> : base_comparison_kernel<equal_kernel<tuple_type_id, tuple_type_id>> {
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
        expr_single_t opchild = echild->get_function<expr_single_t>();
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

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *ectx,
                                intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                                const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars));
  };

  template <type_id_t I0, type_id_t I1>
  struct not_equal_kernel : base_comparison_kernel<not_equal_kernel<I0, I1>> {
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
  struct not_equal_kernel<I0, I0> : base_comparison_kernel<not_equal_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<A0 *>(src[0]) != *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <>
  struct not_equal_kernel<tuple_type_id,
                          tuple_type_id> : base_comparison_kernel<not_equal_kernel<tuple_type_id, tuple_type_id>> {
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
        expr_single_t opchild = echild->get_function<expr_single_t>();
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

    static intptr_t instantiate(char *static_data, char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *ectx,
                                intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                                const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars));
  };

  template <type_id_t I0, type_id_t I1>
  struct greater_equal_kernel : base_comparison_kernel<greater_equal_kernel<I0, I1>> {
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
  struct greater_equal_kernel<I0, I0> : base_comparison_kernel<greater_equal_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<A0 *>(src[0]) >= *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct greater_kernel : base_comparison_kernel<greater_kernel<I0, I1>> {
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
  struct greater_kernel<I0, I0> : base_comparison_kernel<greater_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<A0 *>(src[0]) > *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <typename FuncType, bool Src0IsOption, bool Src1IsOption>
  struct option_comparison_kernel;

  template <typename FuncType>
  struct option_comparison_kernel<FuncType, true, false> : base_kernel<option_comparison_kernel<FuncType, true, false>,
                                                                       2> {
    static const size_t data_size = 0;
    intptr_t comp_offset;
    intptr_t assign_na_offset;

    void single(char *dst, char *const *src)
    {
      auto is_avail = this->get_child();
      bool1 child_dst;
      is_avail->single(reinterpret_cast<char *>(&child_dst), &src[0]);
      if (child_dst) {
        this->get_child(comp_offset)->single(dst, src);
      } else {
        this->get_child(assign_na_offset)->single(dst, nullptr);
      }
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t option_comp_offset = ckb_offset;
      option_comparison_kernel::make(ckb, kernreq, ckb_offset);

      auto is_avail = is_avail::get();
      ckb_offset =
          is_avail.get()->instantiate(is_avail.get()->static_data(), data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
                                      src_tp, src_arrmeta, kernel_request_single, ectx, nkwd, kwds, tp_vars);
      option_comparison_kernel *self = option_comparison_kernel::get_self(
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb), option_comp_offset);
      self->comp_offset = ckb_offset - option_comp_offset;
      auto cmp = FuncType::get();
      const ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::option_type>()->get_value_type(), src_tp[1]};
      ckb_offset = cmp.get()->instantiate(cmp.get()->static_data(), data, ckb, ckb_offset,
                                          dst_tp.extended<ndt::option_type>()->get_value_type(), dst_arrmeta, nsrc,
                                          child_src_tp, src_arrmeta, kernel_request_single, ectx, nkwd, kwds, tp_vars);
      self = option_comparison_kernel::get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
                                                option_comp_offset);
      self->assign_na_offset = ckb_offset - option_comp_offset;
      auto assign_na = nd::assign_na_decl::get();
      ckb_offset = assign_na.get()->instantiate(assign_na.get()->static_data(), data, ckb, ckb_offset,
                                                ndt::option_type::make(ndt::type::make<bool1>()), nullptr, 0, nullptr,
                                                nullptr, kernel_request_single, ectx, nkwd, kwds, tp_vars);
      return ckb_offset;
    }
  };

  template <typename FuncType>
  struct option_comparison_kernel<FuncType, false, true> : base_kernel<option_comparison_kernel<FuncType, false, true>,
                                                                       2> {
    static const size_t data_size = 0;
    intptr_t comp_offset;
    intptr_t assign_na_offset;

    void single(char *dst, char *const *src)
    {
      auto is_avail = this->get_child();
      bool1 child_dst;
      is_avail->single(reinterpret_cast<char *>(&child_dst), &src[1]);
      if (child_dst) {
        this->get_child(comp_offset)->single(dst, src);
      } else {
        this->get_child(assign_na_offset)->single(dst, nullptr);
      }
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t option_comp_offset = ckb_offset;
      option_comparison_kernel::make(ckb, kernreq, ckb_offset);

      auto is_avail = is_avail::get();
      ckb_offset =
          is_avail.get()->instantiate(is_avail.get()->static_data(), data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
                                      &src_tp[1], &src_arrmeta[1], kernel_request_single, ectx, nkwd, kwds, tp_vars);
      option_comparison_kernel *self = option_comparison_kernel::get_self(
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb), option_comp_offset);
      self->comp_offset = ckb_offset - option_comp_offset;
      auto cmp = FuncType::get();
      const ndt::type child_src_tp[2] = {src_tp[0], src_tp[1].extended<ndt::option_type>()->get_value_type(), };
      ckb_offset = cmp.get()->instantiate(cmp.get()->static_data(), data, ckb, ckb_offset,
                                          dst_tp.extended<ndt::option_type>()->get_value_type(), dst_arrmeta, nsrc,
                                          child_src_tp, src_arrmeta, kernel_request_single, ectx, nkwd, kwds, tp_vars);
      self = option_comparison_kernel::get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
                                                option_comp_offset);
      self->assign_na_offset = ckb_offset - option_comp_offset;
      auto assign_na = nd::assign_na_decl::get();
      ckb_offset = assign_na.get()->instantiate(assign_na.get()->static_data(), data, ckb, ckb_offset,
                                                ndt::option_type::make(ndt::type::make<bool1>()), nullptr, 0, nullptr,
                                                nullptr, kernel_request_single, ectx, nkwd, kwds, tp_vars);
      return ckb_offset;
    }
  };

  template <typename FuncType>
  struct option_comparison_kernel<FuncType, true, true> : base_kernel<option_comparison_kernel<FuncType, true, true>,
                                                                      2> {
    static const size_t data_size = 0;
    intptr_t is_avail_rhs_offset;
    intptr_t comp_offset;
    intptr_t assign_na_offset;

    void single(char *dst, char *const *src)
    {
      auto is_avail_lhs = this->get_child();
      auto is_avail_rhs = this->get_child(is_avail_rhs_offset);
      bool1 child_dst_lhs;
      bool1 child_dst_rhs;
      is_avail_lhs->single(reinterpret_cast<char *>(&child_dst_lhs), &src[0]);
      is_avail_rhs->single(reinterpret_cast<char *>(&child_dst_rhs), &src[1]);
      if (child_dst_lhs && child_dst_rhs) {
        this->get_child(comp_offset)->single(dst, src);
      } else {
        this->get_child(assign_na_offset)->single(dst, nullptr);
      }
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t option_comp_offset = ckb_offset;
      option_comparison_kernel::make(ckb, kernreq, ckb_offset);

      auto is_avail_lhs = is_avail::get();
      ckb_offset = is_avail_lhs.get()->instantiate(is_avail_lhs.get()->static_data(), data, ckb, ckb_offset, dst_tp,
                                                   dst_arrmeta, nsrc, &src_tp[0], &src_arrmeta[0],
                                                   kernel_request_single, ectx, nkwd, kwds, tp_vars);
      option_comparison_kernel *self = option_comparison_kernel::get_self(
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb), option_comp_offset);
      self->is_avail_rhs_offset = ckb_offset - option_comp_offset;

      auto is_avail_rhs = is_avail::get();
      ckb_offset = is_avail_rhs.get()->instantiate(is_avail_rhs.get()->static_data(), data, ckb, ckb_offset, dst_tp,
                                                   dst_arrmeta, nsrc, &src_tp[1], &src_arrmeta[1],
                                                   kernel_request_single, ectx, nkwd, kwds, tp_vars);
      self = option_comparison_kernel::get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
                                                option_comp_offset);
      self->comp_offset = ckb_offset - option_comp_offset;
      auto cmp = FuncType::get();
      const ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::option_type>()->get_value_type(),
                                         src_tp[1].extended<ndt::option_type>()->get_value_type()};
      ckb_offset = cmp.get()->instantiate(cmp.get()->static_data(), data, ckb, ckb_offset,
                                          dst_tp.extended<ndt::option_type>()->get_value_type(), dst_arrmeta, nsrc,
                                          child_src_tp, src_arrmeta, kernel_request_single, ectx, nkwd, kwds, tp_vars);
      self = option_comparison_kernel::get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
                                                option_comp_offset);
      self->assign_na_offset = ckb_offset - option_comp_offset;
      auto assign_na = nd::assign_na_decl::get();
      ckb_offset = assign_na.get()->instantiate(assign_na.get()->static_data(), data, ckb, ckb_offset,
                                                ndt::option_type::make(ndt::type::make<bool1>()), nullptr, 0, nullptr,
                                                nullptr, kernel_request_single, ectx, nkwd, kwds, tp_vars);
      return ckb_offset;
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::less_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      return callable_type::make(type::make<bool1>(), {type(Src0TypeID), type(Src1TypeID)});
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::less_equal_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      return callable_type::make(type::make<bool1>(), {type(Src0TypeID), type(Src1TypeID)});
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::equal_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      return callable_type::make(type::make<bool1>(), {type(Src0TypeID), type(Src1TypeID)});
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::not_equal_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      return callable_type::make(type::make<bool1>(), {type(Src0TypeID), type(Src1TypeID)});
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::greater_equal_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      return callable_type::make(type::make<bool1>(), {type(Src0TypeID), type(Src1TypeID)});
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::greater_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      return callable_type::make(type::make<bool1>(), {type(Src0TypeID), type(Src1TypeID)});
    }
  };

  template <typename FuncType>
  struct type::equivalent<nd::option_comparison_kernel<FuncType, true, false>> {
    static type make()
    {
      return type("(?Scalar, Scalar) -> ?bool");
    }
  };

  template <typename FuncType>
  struct type::equivalent<nd::option_comparison_kernel<FuncType, false, true>> {
    static type make()
    {
      return type("(Scalar, ?Scalar) -> ?bool");
    }
  };

  template <typename FuncType>
  struct type::equivalent<nd::option_comparison_kernel<FuncType, true, true>> {
    static type make()
    {
      return type("(?Scalar, ?Scalar) -> ?bool");
    }
  };

} // namespace dynd::ndt
} // namespace dynd
