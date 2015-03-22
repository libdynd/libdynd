#pragma once

#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/kernels/multidispatch_by_type_id.hpp>

namespace dynd {
namespace nd {

  template <type_id_t I0>
  struct plus_ck : expr_ck<plus_ck<I0>, kernel_request_cuda_host_device, 1> {
    typedef plus_ck self_type;
    typedef typename type_of<I0>::type A0;
    typedef decltype(+std::declval<A0>()) R;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<R *>(dst) = +*reinterpret_cast<A0 *>(src[0]);
    }

    static ndt::type make_type()
    {
      std::map<string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::make_type<A0>();
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(A0) -> R"), tp_vars, true);
    }

    static int resolve_dst_type(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp), int DYND_UNUSED(throw_on_error),
        ndt::type &out_dst_tp, const dynd::nd::array &DYND_UNUSED(kwds),
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      out_dst_tp = ndt::make_type<R>();
      return 1;
    }

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx),
        const dynd::nd::array &DYND_UNUSED(kwds),
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      self_type::create(ckb, kernreq, ckb_offset);
      return ckb_offset;
    }
  };

  struct multidispatch_plus_ck
      : multidispatch_by_type_id_ck<multidispatch_plus_ck, 1> {
    static dynd::detail::array_by_type_id<arrfunc, 1> children;

    static ndt::type make_type();
  };

  template <typename A0>
  struct minus_ck : expr_ck<minus_ck<A0>, kernel_request_cuda_host_device, 1> {
    typedef minus_ck self_type;
    typedef decltype(-std::declval<A0>()) R;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<R *>(dst) = -*reinterpret_cast<A0 *>(src[0]);
    }

    static ndt::type make_type()
    {
      std::map<string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::make_type<A0>();
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(A0) -> R"), tp_vars, true);
    }

    static int resolve_dst_type(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp), int DYND_UNUSED(throw_on_error),
        ndt::type &out_dst_tp, const dynd::nd::array &DYND_UNUSED(kwds),
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      out_dst_tp = ndt::make_type<R>();
      return 1;
    }

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx),
        const dynd::nd::array &DYND_UNUSED(kwds),
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      self_type::create(ckb, kernreq, ckb_offset);
      return ckb_offset;
    }
  };

  struct multidispatch_minus_ck
      : multidispatch_by_type_id_ck<multidispatch_minus_ck, 1> {
    static dynd::detail::array_by_type_id<arrfunc, 1> children;

    static ndt::type make_type();
  };

  template <type_id_t I0, type_id_t I1>
  struct add_ck : expr_ck<add_ck<I0, I1>, kernel_request_cuda_host_device, 2> {
    typedef add_ck self_type;
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef decltype(std::declval<A0>() + std::declval<A1>()) R;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<R *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) + *reinterpret_cast<A1 *>(src[1]);
    }

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx),
        const dynd::nd::array &DYND_UNUSED(kwds),
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      self_type::create(ckb, kernreq, ckb_offset);
      return ckb_offset;
    }

    static int resolve_dst_type(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp), int DYND_UNUSED(throw_on_error),
        ndt::type &out_dst_tp, const dynd::nd::array &DYND_UNUSED(kwds),
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      out_dst_tp = ndt::make_type<R>();
      return 1;
    }

    static ndt::type make_type()
    {
      std::map<string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::make_type<A0>();
      tp_vars["A1"] = ndt::make_type<A1>();
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(A0, A1) -> R"), tp_vars, true);
    }
  };

  struct multidispatch_add_ck : multidispatch_by_type_id_ck<multidispatch_add_ck, 2> {
    static dynd::detail::array_by_type_id<arrfunc, 2> children;

    static ndt::type make_type();
  };

  template <type_id_t I0, type_id_t I1>
  struct subtract_ck
      : expr_ck<subtract_ck<I0, I1>, kernel_request_cuda_host_device, 2> {
    typedef subtract_ck self_type;
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef decltype(std::declval<A0>() - std::declval<A1>()) R;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<R *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) - *reinterpret_cast<A1 *>(src[1]);
    }

    static ndt::type make_type()
    {
      std::map<string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::make_type<A0>();
      tp_vars["A1"] = ndt::make_type<A1>();
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(A0, A1) -> R"), tp_vars, true);
    }

    static int resolve_dst_type(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp), int DYND_UNUSED(throw_on_error),
        ndt::type &out_dst_tp, const dynd::nd::array &DYND_UNUSED(kwds),
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      out_dst_tp = ndt::make_type<R>();
      return 1;
    }

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx),
        const dynd::nd::array &DYND_UNUSED(kwds),
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      self_type::create(ckb, kernreq, ckb_offset);
      return ckb_offset;
    }
  };

  struct multidispatch_subtract_ck
      : multidispatch_by_type_id_ck<multidispatch_subtract_ck, 2> {
    static dynd::detail::array_by_type_id<arrfunc, 2> children;

    static ndt::type make_type();
  };

  template <type_id_t I0, type_id_t I1>
  struct multiply_ck
      : expr_ck<multiply_ck<I0, I1>, kernel_request_cuda_host_device, 2> {
    typedef multiply_ck self_type;
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef decltype(std::declval<A0>() * std::declval<A1>()) R;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<R *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) * *reinterpret_cast<A1 *>(src[1]);
    }

    static ndt::type make_type()
    {
      std::map<string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::make_type<A0>();
      tp_vars["A1"] = ndt::make_type<A1>();
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(A0, A1) -> R"), tp_vars, true);
    }

    static int resolve_dst_type(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp), int DYND_UNUSED(throw_on_error),
        ndt::type &out_dst_tp, const dynd::nd::array &DYND_UNUSED(kwds),
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      out_dst_tp = ndt::make_type<R>();
      return 1;
    }

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx),
        const dynd::nd::array &DYND_UNUSED(kwds),
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      self_type::create(ckb, kernreq, ckb_offset);
      return ckb_offset;
    }
  };

  struct multidispatch_multiply_ck
      : multidispatch_by_type_id_ck<multidispatch_multiply_ck, 2> {
    static dynd::detail::array_by_type_id<arrfunc, 2> children;

    static ndt::type make_type();
  };

  template <type_id_t I0, type_id_t I1>
  struct divide_ck
      : expr_ck<divide_ck<I0, I1>, kernel_request_cuda_host_device, 2> {
    typedef divide_ck self_type;
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef decltype(std::declval<A0>() / std::declval<A1>()) R;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<R *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) / *reinterpret_cast<A1 *>(src[1]);
    }

    static ndt::type make_type()
    {
      std::map<string, ndt::type> tp_vars;
      tp_vars["A0"] = ndt::make_type<A0>();
      tp_vars["A1"] = ndt::make_type<A1>();
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(A0, A1) -> R"), tp_vars, true);
    }

    static int resolve_dst_type(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp), int DYND_UNUSED(throw_on_error),
        ndt::type &out_dst_tp, const dynd::nd::array &DYND_UNUSED(kwds),
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      out_dst_tp = ndt::make_type<R>();
      return 1;
    }

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx),
        const dynd::nd::array &DYND_UNUSED(kwds),
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      self_type::create(ckb, kernreq, ckb_offset);
      return ckb_offset;
    }
  };

  struct multidispatch_divide_ck
      : multidispatch_by_type_id_ck<multidispatch_divide_ck, 2> {
    static dynd::detail::array_by_type_id<arrfunc, 2> children;

    static ndt::type make_type();
  };

} // namespace dynd::nd
} // namespace dynd