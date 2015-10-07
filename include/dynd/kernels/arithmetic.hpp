#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/func/option.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/kernels/apply.hpp>

namespace dynd {
namespace nd {

#define DYND_DeclUnaryOp(OP, NAME)                                                                                    \
  template <type_id_t Src0TypeID>                                                                                     \
  struct inline_ ## NAME {                                                                                            \
    static auto f(typename type_of<Src0TypeID>::type a) { return OP a; }                                              \
  };                                                                                                                  \
  template <type_id_t Src0TypeID>                                                                                     \
  struct NAME ## _kernel : functional::as_apply_function_ck<decltype(&inline_ ## NAME <Src0TypeID>::f),               \
                                                           &inline_ ## NAME <Src0TypeID>::f> {};                      \

  DYND_DeclUnaryOp(+, plus)
  DYND_DeclUnaryOp(-, minus)
  DYND_DeclUnaryOp(!, logical_not)
  DYND_DeclUnaryOp(~, bitwise_not)

#undef DYND_DeclUnaryOp

#define DYND_DeclBinopKernel(OP, NAME)                                                                               \
  template<type_id_t Src0TypeID, type_id_t Src1TypeID>                                                               \
  struct inline_ ## NAME {                                                                                           \
    static auto f(typename type_of<Src0TypeID>::type a, typename type_of<Src1TypeID>::type b) { return a OP b; }     \
  };                                                                                                                 \
  template<type_id_t Src0TypeID, type_id_t Src1TypeID>                                                               \
  struct NAME ## _kernel : functional::as_apply_function_ck<decltype(&inline_ ## NAME <Src0TypeID, Src1TypeID>::f),  \
                                                            &inline_ ## NAME <Src0TypeID, Src1TypeID>::f> {};        \

  DYND_DeclBinopKernel(+, add)
  DYND_DeclBinopKernel(-, subtract)
  DYND_DeclBinopKernel(*, multiply)
  DYND_DeclBinopKernel(/, divide)
  DYND_DeclBinopKernel(%, mod)
  DYND_DeclBinopKernel(&, bitwise_and)
  DYND_DeclBinopKernel(&&, logical_and)
  DYND_DeclBinopKernel(|, bitwise_or)
  DYND_DeclBinopKernel(||, logical_or)
  DYND_DeclBinopKernel(^, bitwise_xor)
  DYND_DeclBinopKernel(<<, left_shift)
  DYND_DeclBinopKernel(>>, right_shift)

  template<type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct inline_logical_xor {
    static auto f(typename type_of<Src0TypeID>::type a, typename type_of<Src1TypeID>::type b) { return (!a) ^ (!b); }
  };
  template<type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct logical_xor_kernel : functional::as_apply_function_ck<
                                              decltype(&inline_logical_xor<Src0TypeID, Src1TypeID>::f),
                                              &inline_logical_xor<Src0TypeID, Src1TypeID>::f> {};

#undef DYND_DeclBinopKernel

  template <typename FuncType, bool Src0IsOption, bool Src1IsOption>
  struct option_arithmetic_kernel;

  template <typename FuncType>
  struct option_arithmetic_kernel<FuncType, true, false> : base_kernel<option_arithmetic_kernel<FuncType, true, false>, 2> {
    static const size_t data_size = 0;
    intptr_t arith_offset;
    intptr_t assign_na_offset;

    void single(char *dst, char *const *src)
    {
      auto is_avail = this->get_child();
      bool1 child_dst;
      is_avail->single(reinterpret_cast<char*>(&child_dst), &src[0]);
      if (child_dst) {
        this->get_child(arith_offset)->single(dst, src);
      } else {
        this->get_child(assign_na_offset)->single(dst, nullptr);
      }
    }

    static void
    resolve_dst_type(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
                     char *data, ndt::type &dst_tp, intptr_t nsrc,
                     const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars)
    {
      auto k = FuncType::get().get();
      const ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::option_type>()->get_value_type(),
                                         src_tp[1]};
      k->resolve_dst_type(
          k->static_data, k->data_size, data, dst_tp, nsrc,
          child_src_tp, nkwd, kwds, tp_vars);
      dst_tp = ndt::option_type::make(dst_tp);
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data),
                                size_t DYND_UNUSED(data_size),
                                char *data,
                                void *ckb,
                                intptr_t ckb_offset,
                                const ndt::type &dst_tp,
                                const char *dst_arrmeta,
                                intptr_t nsrc,
                                const ndt::type *src_tp,
                                const char *const *src_arrmeta,
                                kernel_request_t kernreq,
                                const eval::eval_context *ectx,
                                intptr_t nkwd,
                                const array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t option_arith_offset = ckb_offset;
      option_arithmetic_kernel::make(ckb, kernreq, ckb_offset);

      auto is_avail = is_avail::get();
      ckb_offset = is_avail.get()->instantiate(is_avail.get()->static_data, is_avail.get()->data_size,
                                                    data,
                                                    ckb,
                                                    ckb_offset,
                                                    dst_tp,
                                                    dst_arrmeta,
                                                    nsrc,
                                                    src_tp,
                                                    src_arrmeta,
                                                    kernel_request_single,
                                                    ectx,
                                                    nkwd,
                                                    kwds,
                                                    tp_vars);
      option_arithmetic_kernel *self =
          option_arithmetic_kernel::get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
                   option_arith_offset);
      self->arith_offset = ckb_offset - option_arith_offset;
      auto arith = FuncType::get();
      const ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::option_type>()->get_value_type(),
                                         src_tp[1]};
      ckb_offset = arith.get()->instantiate(arith.get()->static_data, arith.get()->data_size,
                                                    data,
                                                    ckb,
                                                    ckb_offset,
                                                    dst_tp,
                                                    dst_arrmeta,
                                                    nsrc,
                                                    child_src_tp,
                                                    src_arrmeta,
                                                    kernel_request_single,
                                                    ectx,
                                                    nkwd,
                                                    kwds,
                                                    tp_vars);
      self = option_arithmetic_kernel::get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
                   option_arith_offset);
      self->assign_na_offset = ckb_offset - option_arith_offset;
      auto assign_na = nd::assign_na_decl::get();
      ckb_offset = assign_na.get()->instantiate(assign_na.get()->static_data, assign_na.get()->data_size,
                                                    data,
                                                    ckb,
                                                    ckb_offset,
                                                    dst_tp,
                                                    dst_arrmeta,
                                                    nsrc,
                                                    child_src_tp,
                                                    src_arrmeta,
                                                    kernel_request_single,
                                                    ectx,
                                                    nkwd,
                                                    kwds,
                                                    tp_vars);
      return ckb_offset;
    }
  };

  template <typename FuncType>
  struct option_arithmetic_kernel<FuncType, false, true> : base_kernel<option_arithmetic_kernel<FuncType, false, true>, 2> {
    static const size_t data_size = 0;
    intptr_t arith_offset;
    intptr_t assign_na_offset;

    void single(char *dst, char *const *src)
    {
      auto is_avail = this->get_child();
      bool1 child_dst;
      is_avail->single(reinterpret_cast<char*>(&child_dst), &src[1]);
      if (child_dst) {
        this->get_child(arith_offset)->single(dst, src);
      } else {
        this->get_child(assign_na_offset)->single(dst, nullptr);
      }
    }

    static void
    resolve_dst_type(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
                     char *data, ndt::type &dst_tp, intptr_t nsrc,
                     const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars)
    {
      auto k = FuncType::get().get();
      const ndt::type child_src_tp[2] = {
          src_tp[0], src_tp[1].extended<ndt::option_type>()->get_value_type()
      };
      k->resolve_dst_type(
          k->static_data, k->data_size, data, dst_tp, nsrc,
          child_src_tp, nkwd, kwds, tp_vars);
      dst_tp = ndt::option_type::make(dst_tp);
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data),
                                size_t DYND_UNUSED(data_size),
                                char *data,
                                void *ckb,
                                intptr_t ckb_offset,
                                const ndt::type &dst_tp,
                                const char *dst_arrmeta,
                                intptr_t nsrc,
                                const ndt::type *src_tp,
                                const char *const *src_arrmeta,
                                kernel_request_t kernreq,
                                const eval::eval_context *ectx,
                                intptr_t nkwd,
                                const array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t option_arith_offset = ckb_offset;
      option_arithmetic_kernel::make(ckb, kernreq, ckb_offset);

      auto is_avail = is_avail::get();
      ckb_offset = is_avail.get()->instantiate(is_avail.get()->static_data, is_avail.get()->data_size,
                                                    data,
                                                    ckb,
                                                    ckb_offset,
                                                    dst_tp,
                                                    dst_arrmeta,
                                                    nsrc,
                                                    &src_tp[1],
                                                    &src_arrmeta[1],
                                                    kernel_request_single,
                                                    ectx,
                                                    nkwd,
                                                    kwds,
                                                    tp_vars);
      option_arithmetic_kernel *self =
          option_arithmetic_kernel::get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
                   option_arith_offset);
      self->arith_offset = ckb_offset - option_arith_offset;
      auto arith = FuncType::get();
      const ndt::type child_src_tp[2] = {
          src_tp[0], src_tp[1].extended<ndt::option_type>()->get_value_type()
      };
      ckb_offset = arith.get()->instantiate(arith.get()->static_data, arith.get()->data_size,
                                                    data,
                                                    ckb,
                                                    ckb_offset,
                                                    dst_tp,
                                                    dst_arrmeta,
                                                    nsrc,
                                                    child_src_tp,
                                                    src_arrmeta,
                                                    kernel_request_single,
                                                    ectx,
                                                    nkwd,
                                                    kwds,
                                                    tp_vars);
      self = option_arithmetic_kernel::get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
                   option_arith_offset);
      self->assign_na_offset = ckb_offset - option_arith_offset;
      auto assign_na = nd::assign_na_decl::get();
      ckb_offset = assign_na.get()->instantiate(assign_na.get()->static_data, assign_na.get()->data_size,
                                                    data,
                                                    ckb,
                                                    ckb_offset,
                                                    src_tp[1],
                                                    src_arrmeta[1],
                                                    0,
                                                    nullptr,
                                                    nullptr,
                                                    kernel_request_single,
                                                    ectx,
                                                    nkwd,
                                                    kwds,
                                                    tp_vars);
      return ckb_offset;
    }
  };

  template <typename FuncType>
  struct option_arithmetic_kernel<FuncType, true, true> : base_kernel<option_arithmetic_kernel<FuncType, true, true>, 2> {
    static const size_t data_size = 0;
    intptr_t is_avail_rhs_offset;
    intptr_t arith_offset;
    intptr_t assign_na_offset;

    void single(char *dst, char *const *src)
    {
      auto is_avail_lhs = this->get_child();
      auto is_avail_rhs = this->get_child(is_avail_rhs_offset);
      bool1 child_dst_lhs;
      bool1 child_dst_rhs;
      is_avail_lhs->single(reinterpret_cast<char*>(&child_dst_lhs), &src[0]);
      is_avail_rhs->single(reinterpret_cast<char*>(&child_dst_rhs), &src[1]);
      if (child_dst_lhs && child_dst_rhs) {
        this->get_child(arith_offset)->single(dst, src);
      } else {
        this->get_child(assign_na_offset)->single(dst, nullptr);
      }
    }

    static void
    resolve_dst_type(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
                     char *data, ndt::type &dst_tp, intptr_t nsrc,
                     const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars)
    {
      auto k = FuncType::get().get();
      const ndt::type child_src_tp[2] = {
          src_tp[0].extended<ndt::option_type>()->get_value_type(),
          src_tp[1].extended<ndt::option_type>()->get_value_type()
      };
      k->resolve_dst_type(
          k->static_data, k->data_size, data, dst_tp, nsrc,
          child_src_tp, nkwd, kwds, tp_vars);
      dst_tp = ndt::option_type::make(dst_tp);
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data),
                                size_t DYND_UNUSED(data_size),
                                char *data,
                                void *ckb,
                                intptr_t ckb_offset,
                                const ndt::type &dst_tp,
                                const char *dst_arrmeta,
                                intptr_t nsrc,
                                const ndt::type *src_tp,
                                const char *const *src_arrmeta,
                                kernel_request_t kernreq,
                                const eval::eval_context *ectx,
                                intptr_t nkwd,
                                const array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t option_arith_offset = ckb_offset;
      option_arithmetic_kernel::make(ckb, kernreq, ckb_offset);

      auto is_avail_lhs = is_avail::get();
      ckb_offset = is_avail_lhs.get()->instantiate(is_avail_lhs.get()->static_data, is_avail_lhs.get()->data_size,
                                                    data,
                                                    ckb,
                                                    ckb_offset,
                                                    dst_tp,
                                                    dst_arrmeta,
                                                    nsrc,
                                                    src_tp,
                                                    src_arrmeta,
                                                    kernel_request_single,
                                                    ectx,
                                                    nkwd,
                                                    kwds,
                                                    tp_vars);
      option_arithmetic_kernel *self =
          option_arithmetic_kernel::get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
                   option_arith_offset);
      self->is_avail_rhs_offset = ckb_offset - option_arith_offset;
      auto is_avail_rhs = is_avail::get();
      ckb_offset = is_avail_rhs.get()->instantiate(is_avail_rhs.get()->static_data, is_avail_rhs.get()->data_size,
                                                    data,
                                                    ckb,
                                                    ckb_offset,
                                                    dst_tp,
                                                    dst_arrmeta,
                                                    nsrc,
                                                    src_tp,
                                                    src_arrmeta,
                                                    kernel_request_single,
                                                    ectx,
                                                    nkwd,
                                                    kwds,
                                                    tp_vars);
      self = option_arithmetic_kernel::get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
                   option_arith_offset);
      self->arith_offset = ckb_offset - option_arith_offset;
      auto arith = FuncType::get();
      const ndt::type child_src_tp[2] = {
          src_tp[0].extended<ndt::option_type>()->get_value_type(),
          src_tp[1].extended<ndt::option_type>()->get_value_type()
      };
      ckb_offset = arith.get()->instantiate(arith.get()->static_data, arith.get()->data_size,
                                                    data,
                                                    ckb,
                                                    ckb_offset,
                                                    dst_tp,
                                                    dst_arrmeta,
                                                    nsrc,
                                                    child_src_tp,
                                                    src_arrmeta,
                                                    kernel_request_single,
                                                    ectx,
                                                    nkwd,
                                                    kwds,
                                                    tp_vars);
      self = option_arithmetic_kernel::get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
                   option_arith_offset);
      self->assign_na_offset = ckb_offset - option_arith_offset;
      auto assign_na = nd::assign_na_decl::get();
      ckb_offset = assign_na.get()->instantiate(assign_na.get()->static_data, assign_na.get()->data_size,
                                                    data,
                                                    ckb,
                                                    ckb_offset,
                                                    dst_tp,
                                                    dst_arrmeta,
                                                    0,
                                                    nullptr,
                                                    nullptr,
                                                    kernel_request_single,
                                                    ectx,
                                                    nkwd,
                                                    kwds,
                                                    tp_vars);
      return ckb_offset;
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Src0TypeID>
  struct type::equivalent<nd::plus_kernel<Src0TypeID>> {
    typedef typename dynd::type_of<Src0TypeID>::type A0;
    typedef decltype(+std::declval<A0>()) R;

    static type make()
    {
      std::map<std::string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::type::make<A0>();
      tp_vars["R"] = ndt::type::make<R>();

      return ndt::substitute(ndt::type("(A0) -> R"), tp_vars, true);
    }
  };

  template <type_id_t Src0TypeID>
  struct type::equivalent<nd::minus_kernel<Src0TypeID>> {
    typedef typename dynd::type_of<Src0TypeID>::type A0;
    typedef decltype(-std::declval<A0>()) R;

    static type make()
    {
      std::map<std::string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::type::make<A0>();
      tp_vars["R"] = ndt::type::make<R>();

      return ndt::substitute(ndt::type("(A0) -> R"), tp_vars, true);
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::add_kernel<Src0TypeID, Src1TypeID>> {
    typedef typename dynd::type_of<Src0TypeID>::type A0;
    typedef typename dynd::type_of<Src1TypeID>::type A1;
    typedef decltype(std::declval<A0>() + std::declval<A1>()) R;

    static type make()
    {
      std::map<std::string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::type::make<A0>();
      tp_vars["A1"] = ndt::type::make<A1>();
      tp_vars["R"] = ndt::type::make<R>();

      return ndt::substitute(ndt::type("(A0, A1) -> R"), tp_vars, true);
    }
  };

  template<typename FuncType>
  struct type::equivalent<nd::option_arithmetic_kernel<FuncType, true, false>> {
    static type make() {
      return type("(?Scalar, Scalar) -> ?Scalar");
    }
  };

  template<typename FuncType>
  struct type::equivalent<nd::option_arithmetic_kernel<FuncType, false, true>> {
    static type make() {
      return type("(Scalar, ?Scalar) -> ?Scalar");
    }
  };

  template<typename FuncType>
  struct type::equivalent<nd::option_arithmetic_kernel<FuncType, true, true>> {
    static type make() {
      return type("(?Scalar, ?Scalar) -> ?Scalar");
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::subtract_kernel<Src0TypeID, Src1TypeID>> {
    typedef typename dynd::type_of<Src0TypeID>::type A0;
    typedef typename dynd::type_of<Src1TypeID>::type A1;
    typedef decltype(std::declval<A0>() - std::declval<A1>()) R;

    static type make()
    {
      std::map<std::string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::type::make<A0>();
      tp_vars["A1"] = ndt::type::make<A1>();
      tp_vars["R"] = ndt::type::make<R>();

      return ndt::substitute(ndt::type("(A0, A1) -> R"), tp_vars, true);
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::multiply_kernel<Src0TypeID, Src1TypeID>> {
    typedef typename dynd::type_of<Src0TypeID>::type A0;
    typedef typename dynd::type_of<Src1TypeID>::type A1;
    typedef decltype(std::declval<A0>() * std::declval<A1>()) R;

    static type make()
    {
      std::map<std::string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::type::make<A0>();
      tp_vars["A1"] = ndt::type::make<A1>();
      tp_vars["R"] = ndt::type::make<R>();

      return ndt::substitute(ndt::type("(A0, A1) -> R"), tp_vars, true);
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::divide_kernel<Src0TypeID, Src1TypeID>> {
    typedef typename dynd::type_of<Src0TypeID>::type A0;
    typedef typename dynd::type_of<Src1TypeID>::type A1;
    typedef decltype(std::declval<A0>() / std::declval<A1>()) R;

    static type make()
    {
      std::map<std::string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::type::make<A0>();
      tp_vars["A1"] = ndt::type::make<A1>();
      tp_vars["R"] = ndt::type::make<R>();

      return ndt::substitute(ndt::type("(A0, A1) -> R"), tp_vars, true);
    }
  };

} // namespace dynd::ndt
} // namespace dynd
