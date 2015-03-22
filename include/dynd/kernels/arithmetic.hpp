#pragma once

#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/kernels/virtual.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {
namespace nd {

  template <typename CKT>
  struct single_dispatch_ck : virtual_ck<single_dispatch_ck<CKT>> {
    static ndt::type make_type() { return ndt::type("(Any) -> Any"); }

    static int
    resolve_dst_type(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                     intptr_t nsrc, const ndt::type *src_tp, int throw_on_error,
                     ndt::type &out_dst_tp, const dynd::nd::array &kwds,
                     const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      const arrfunc &child = CKT::children[src_tp[0].get_type_id() - bool_type_id];
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
      const arrfunc &child = CKT::children[src_tp[0].get_type_id() - bool_type_id];
      return child.get()->instantiate(self, self_tp, ckb, ckb_offset, dst_tp,
                                      dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                      kernreq, ectx, kwds, tp_vars);
    }
  };

  template <typename R>
  struct plus_ck : expr_ck<plus_ck<R>, kernel_request_cuda_host_device, 1> {
    typedef plus_ck self_type;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<R *>(dst) = +*reinterpret_cast<R *>(src[0]);
    }

    static ndt::type make_type()
    {
      std::map<string, ndt::type> tp_vars;
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(R) -> R"), tp_vars, true);
    }

    static int resolve_dst_type(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, int DYND_UNUSED(throw_on_error),
        ndt::type &out_dst_tp, const dynd::nd::array &DYND_UNUSED(kwds),
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      out_dst_tp = src_tp[0];
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
    static const arrfunc children[builtin_type_id_count - 2];

    static ndt::type make_type();
  };

  template <typename R>
  struct minus_ck : expr_ck<minus_ck<R>, kernel_request_cuda_host_device, 1> {
    typedef minus_ck self_type;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<R *>(dst) = -*reinterpret_cast<R *>(src[0]);
    }

    static ndt::type make_type()
    {
      std::map<string, ndt::type> tp_vars;
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(R) -> R"), tp_vars, true);
    }

    static int resolve_dst_type(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, int DYND_UNUSED(throw_on_error),
        ndt::type &out_dst_tp, const dynd::nd::array &DYND_UNUSED(kwds),
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      out_dst_tp = src_tp[0];
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
    static const arrfunc children[builtin_type_id_count - 2];

    static ndt::type make_type();
  };

#define ARITHMETIC_OPERATOR(NAME, SYMBOL)                                      \
  template <typename A0, typename A1>                                          \
  struct NAME##_ck                                                             \
      : expr_ck<NAME##_ck<A0, A1>, kernel_request_cuda_host_device, 2> {       \
    typedef decltype(std::declval<A0>() SYMBOL std::declval<A1>()) R;          \
                                                                               \
    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)             \
    {                                                                          \
      *reinterpret_cast<R *>(dst) = *reinterpret_cast<A0 *>(src[0]) SYMBOL *   \
                                    reinterpret_cast<A1 *>(src[1]);            \
    }                                                                          \
                                                                               \
    DYND_CUDA_HOST_DEVICE void strided(R *__restrict dst,                      \
                                       const A0 *__restrict src0,              \
                                       const A1 *__restrict src1,              \
                                       size_t count)                           \
    {                                                                          \
      for (size_t i = DYND_THREAD_ID(0); i < count;                            \
           i += DYND_THREAD_COUNT(0)) {                                        \
        dst[i] = src0[i] SYMBOL src1[i];                                       \
      }                                                                        \
    }                                                                          \
                                                                               \
    DYND_CUDA_HOST_DEVICE void strided(char *__restrict dst,                   \
                                       intptr_t dst_stride,                    \
                                       char *__restrict const *src,            \
                                       const intptr_t *__restrict src_stride,  \
                                       size_t count)                           \
    {                                                                          \
      if (dst_stride == sizeof(R) && src_stride[0] == sizeof(A0) &&            \
          src_stride[1] == sizeof(A1)) {                                       \
        strided(reinterpret_cast<R *>(dst),                                    \
                reinterpret_cast<const A0 *>(src[0]),                          \
                reinterpret_cast<const A1 *>(src[1]), count);                  \
      } else {                                                                 \
        const char *__restrict src0 = src[0], *__restrict src1 = src[1];       \
        intptr_t src0_stride = src_stride[0], src1_stride = src_stride[1];     \
        for (size_t i = 0; i != count; ++i) {                                  \
          *reinterpret_cast<R *>(dst) =                                        \
              *reinterpret_cast<const A0 *>(src0) SYMBOL *                     \
              reinterpret_cast<const A1 *>(src1);                              \
          dst += dst_stride;                                                   \
          src0 += src0_stride;                                                 \
          src1 += src1_stride;                                                 \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  };

  ARITHMETIC_OPERATOR(add, +);
  ARITHMETIC_OPERATOR(sub, -);
  ARITHMETIC_OPERATOR(mul, *);
  ARITHMETIC_OPERATOR(div, / );

#undef ARITHMETIC_OPERATOR

} // namespace dynd::nd
} // namespace dynd