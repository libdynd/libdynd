#pragma once

#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/kernels/virtual.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {
namespace nd {

  template <typename CKT>
  struct single_dispatch_ck : virtual_ck<single_dispatch_ck<CKT>> {
    static int
    resolve_dst_type(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                     intptr_t nsrc, const ndt::type *src_tp, int throw_on_error,
                     ndt::type &out_dst_tp, const dynd::nd::array &kwds,
                     const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      const arrfunc &child = CKT::children(src_tp[0].get_type_id());
      return child.get()->resolve_dst_type(self, self_tp, nsrc, src_tp,
                                           throw_on_error, out_dst_tp, kwds,
                                           tp_vars);
    }

    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                const char *const *src_arrmeta, kernel_request_t kernreq,
                const eval::eval_context *ectx, const dynd::nd::array &kwds,
                const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      const arrfunc &child = CKT::children(src_tp[0].get_type_id());
      return child.get()->instantiate(self, self_tp, ckb, ckb_offset, dst_tp,
                                      dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                      kernreq, ectx, kwds, tp_vars);
    }
  };

  template <typename CKT>
  struct double_dispatch_ck : virtual_ck<double_dispatch_ck<CKT>> {
    static int
    resolve_dst_type(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                     intptr_t nsrc, const ndt::type *src_tp, int throw_on_error,
                     ndt::type &out_dst_tp, const dynd::nd::array &kwds,
                     const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      std::cout << "double_dispatch_ck::resolve_dst_type" << std::endl;
      std::cout << src_tp[0] << std::endl;
      std::cout << src_tp[1] << std::endl;
      const arrfunc &child =
          CKT::children(src_tp[0].get_type_id(), src_tp[1].get_type_id());
      return child.get()->resolve_dst_type(self, self_tp, nsrc, src_tp,
                                           throw_on_error, out_dst_tp, kwds,
                                           tp_vars);
    }

    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                const char *const *src_arrmeta, kernel_request_t kernreq,
                const eval::eval_context *ectx, const dynd::nd::array &kwds,
                const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      const arrfunc &child =
          CKT::children(src_tp[0].get_type_id(), src_tp[1].get_type_id());
      return child.get()->instantiate(self, self_tp, ckb, ckb_offset, dst_tp,
                                      dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                      kernreq, ectx, kwds, tp_vars);
    }
  };

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

  struct multidispatch_plus_ck : single_dispatch_ck<multidispatch_plus_ck> {
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

  struct multidispatch_minus_ck : single_dispatch_ck<multidispatch_minus_ck> {
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

  struct virtual_add_ck : double_dispatch_ck<virtual_add_ck> {
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

  struct virtual_subtract_ck : double_dispatch_ck<virtual_subtract_ck> {
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

  struct virtual_multiply_ck : double_dispatch_ck<virtual_multiply_ck> {
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

  struct virtual_divide_ck : double_dispatch_ck<virtual_divide_ck> {
    static dynd::detail::array_by_type_id<arrfunc, 2> children;

    static ndt::type make_type();
  };

} // namespace dynd::nd
} // namespace dynd