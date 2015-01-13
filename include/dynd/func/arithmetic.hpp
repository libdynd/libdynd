//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type_promotion.hpp>
#include <dynd/func/elwise.hpp>

namespace dynd {

#define ARITHMETIC_OPERATOR(NAME, SYMBOL)                                      \
  namespace kernels {                                                          \
    template <typename A0, typename A1>                                        \
    struct NAME##_ck                                                           \
        : expr_ck<NAME##_ck<A0, A1>, kernel_request_cuda_host_device, 2> {     \
      typedef decltype(declval<A0>() SYMBOL declval<A1>()) R;                  \
                                                                               \
      DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)           \
      {                                                                        \
        *reinterpret_cast<R *>(dst) = *reinterpret_cast<A0 *>(src[0]) SYMBOL * \
                                      reinterpret_cast<A1 *>(src[1]);          \
      }                                                                        \
                                                                               \
      DYND_CUDA_HOST_DEVICE void strided(char *dst, intptr_t dst_stride,       \
                                         char *const *src,                     \
                                         const intptr_t *src_stride,           \
                                         size_t count)                         \
      {                                                                        \
        char *src0 = src[0], *src1 = src[1];                                   \
        intptr_t src0_stride = src_stride[0], src1_stride = src_stride[1];     \
        for (size_t i = 0; i != count; ++i) {                                  \
          *reinterpret_cast<R *>(dst) = *reinterpret_cast<A0 *>(src0) SYMBOL * \
                                        reinterpret_cast<A1 *>(src1);          \
          dst += dst_stride;                                                   \
          src0 += src0_stride;                                                 \
          src1 += src1_stride;                                                 \
        }                                                                      \
      }                                                                        \
    };                                                                         \
  }                                                                            \
                                                                               \
  namespace decl {                                                             \
    namespace nd {                                                             \
      class NAME : public arrfunc<NAME> {                                      \
        static const kernels::create_t                                         \
            builtin_table[builtin_type_id_count - 2][builtin_type_id_count -   \
                                                     2];                       \
                                                                               \
      public:                                                                  \
        static int resolve_dst_type(                                           \
            const arrfunc_type_data *DYND_UNUSED(self),                        \
            const arrfunc_type *DYND_UNUSED(self_tp),                          \
            intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,               \
            int DYND_UNUSED(throw_on_error), ndt::type &out_dst_tp,            \
            const dynd::nd::array &DYND_UNUSED(kwds),                          \
            const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars)) \
        {                                                                      \
          out_dst_tp =                                                         \
              promote_types_arithmetic(src_tp[0].without_memory_type(),        \
                                       src_tp[1].without_memory_type());       \
          if (src_tp[0].get_kind() == memory_kind) {                           \
            out_dst_tp = src_tp[0]                                             \
                             .extended<base_memory_type>()                     \
                             ->with_replaced_storage_type(out_dst_tp);         \
          }                                                                    \
          return 1;                                                            \
        }                                                                      \
                                                                               \
        static intptr_t instantiate(                                           \
            const arrfunc_type_data *DYND_UNUSED(self),                        \
            const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,               \
            intptr_t ckb_offset, const ndt::type &dst_tp,                      \
            const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp,     \
            const char *const *DYND_UNUSED(src_arrmeta),                       \
            kernel_request_t kernreq,                                          \
            const eval::eval_context *DYND_UNUSED(ectx),                       \
            const dynd::nd::array &DYND_UNUSED(kwds),                          \
            const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars)) \
        {                                                                      \
          if (dst_tp.is_builtin()) {                                           \
            if (src_tp[0].is_builtin() && src_tp[1].is_builtin()) {            \
              kernels::create_t create =                                       \
                  builtin_table[src_tp[0].get_type_id() -                      \
                                bool_type_id][src_tp[1].get_type_id() -        \
                                              bool_type_id];                   \
              create(ckb, kernreq, ckb_offset);                                \
              return ckb_offset;                                               \
            }                                                                  \
          }                                                                    \
                                                                               \
          std::stringstream ss;                                                \
          ss << "arithmetic is not yet implemented for types " << src_tp[0]    \
             << " and " << src_tp[1];                                          \
          throw std::runtime_error(ss.str());                                  \
        }                                                                      \
                                                                               \
        static dynd::nd::arrfunc make()                                        \
        {                                                                      \
          dynd::nd::arrfunc child_af(ndt::type("(Any, Any) -> Any"),           \
                                     &instantiate, NULL, &resolve_dst_type);   \
                                                                               \
          return elwise::bind("func", child_af);                               \
        }                                                                      \
      };                                                                       \
    }                                                                          \
  }                                                                            \
                                                                               \
  namespace nd {                                                               \
    extern decl::nd::NAME NAME;                                                \
                                                                               \
    array operator SYMBOL(const array &a0, const array &a1)                    \
    {                                                                          \
      return NAME(a0, a1);                                                     \
    }                                                                          \
  }

ARITHMETIC_OPERATOR(add, +);
ARITHMETIC_OPERATOR(sub, -);
ARITHMETIC_OPERATOR(mul, *);
ARITHMETIC_OPERATOR(div, / );

#undef ARITHMETIC_OPERATOR

} // namespace dynd
