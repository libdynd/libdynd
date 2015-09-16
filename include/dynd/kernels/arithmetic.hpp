#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/func/option.hpp>
#include <dynd/types/option_type.hpp>

namespace dynd {
namespace nd {

  template <type_id_t I0>
  struct plus_kernel : base_kernel<plus_kernel<I0>, 1> {
    typedef typename type_of<I0>::type A0;
    typedef decltype(+std::declval<A0>()) R;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<R *>(dst) = +*reinterpret_cast<A0 *>(src[0]);
    }
  };

  template <type_id_t I0>
  struct minus_kernel : base_kernel<minus_kernel<I0>, 1> {
    typedef typename type_of<I0>::type A0;
    typedef decltype(-std::declval<A0>()) R;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<R *>(dst) = -*reinterpret_cast<A0 *>(src[0]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct add_kernel : base_kernel<add_kernel<I0, I1>, 2> {
    typedef add_kernel self_type;
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef decltype(std::declval<A0>() + std::declval<A1>()) R;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<R *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) + *reinterpret_cast<A1 *>(src[1]);
    }
  };

  template <typename FuncType>
  struct option_arithmetic_kernel : base_kernel<option_arithmetic_kernel<FuncType>, 2> {
    static const size_t data_size = 0;
    intptr_t arith_offset;

    void single(char *dst, char *const *src)
    {
      auto is_avail = this->get_child();
      bool1 child_dst;
      is_avail->single(reinterpret_cast<char*>(&child_dst), &src[0]);
      if (child_dst) {
        this->get_child(arith_offset)->single(dst, src);
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
                                                    kernreq,
                                                    ectx,
                                                    nkwd,
                                                    kwds,
                                                    tp_vars);
      option_arithmetic_kernel *self =
          option_arithmetic_kernel::get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
                   option_arith_offset);
      self->arith_offset = ckb_offset;
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
                                                    kernreq,
                                                    ectx,
                                                    nkwd,
                                                    kwds,
                                                    tp_vars);
      return ckb_offset;
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct subtract_kernel : base_kernel<subtract_kernel<I0, I1>, 2> {
    typedef subtract_kernel self_type;
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef decltype(std::declval<A0>() - std::declval<A1>()) R;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<R *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) - *reinterpret_cast<A1 *>(src[1]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct multiply_kernel : base_kernel<multiply_kernel<I0, I1>, 2> {
    typedef multiply_kernel self_type;
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef decltype(std::declval<A0>() * std::declval<A1>()) R;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<R *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) * *reinterpret_cast<A1 *>(src[1]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct divide_kernel : base_kernel<divide_kernel<I0, I1>, 2> {
    typedef divide_kernel self_type;
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef decltype(std::declval<A0>() / std::declval<A1>()) R;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<R *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) / *reinterpret_cast<A1 *>(src[1]);
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
  struct type::equivalent<nd::option_arithmetic_kernel<FuncType>> {
    static type make() {
      return type("(?Scalar, Scalar) -> ?Scalar");
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
