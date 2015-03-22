//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type_promotion.hpp>
#include <dynd/func/elwise.hpp>

namespace dynd {
namespace nd {

  extern struct plus : declfunc<plus> {
    static arrfunc make();
  } plus;

  extern struct minus : declfunc<minus> {
    static arrfunc make();
  } minus;

#define ARITHMETIC_OPERATOR(NAME)                                              \
  extern struct NAME : declfunc<NAME> {                                        \
    static const create_t                                                      \
        builtin_table[builtin_type_id_count - 2][builtin_type_id_count - 2];   \
                                                                               \
    static int resolve_dst_type(                                               \
        const arrfunc_type_data *DYND_UNUSED(self),                            \
        const arrfunc_type *DYND_UNUSED(self_tp), intptr_t DYND_UNUSED(nsrc),  \
        const ndt::type *src_tp, int DYND_UNUSED(throw_on_error),              \
        ndt::type &out_dst_tp, const dynd::nd::array &DYND_UNUSED(kwds),       \
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))     \
    {                                                                          \
      out_dst_tp = promote_types_arithmetic(src_tp[0].without_memory_type(),   \
                                            src_tp[1].without_memory_type());  \
      if (src_tp[0].get_kind() == memory_kind) {                               \
        out_dst_tp = src_tp[0]                                                 \
                         .extended<base_memory_type>()                         \
                         ->with_replaced_storage_type(out_dst_tp);             \
      }                                                                        \
      return 1;                                                                \
    }                                                                          \
                                                                               \
    static intptr_t instantiate(                                               \
        const arrfunc_type_data *DYND_UNUSED(self),                            \
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,                   \
        intptr_t ckb_offset, const ndt::type &dst_tp,                          \
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),      \
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),  \
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx), \
        const dynd::nd::array &DYND_UNUSED(kwds),                              \
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))     \
    {                                                                          \
      if (dst_tp.is_builtin()) {                                               \
        if (src_tp[0].is_builtin() && src_tp[1].is_builtin()) {                \
          create_t create =                                                    \
              builtin_table[src_tp[0].get_type_id() -                          \
                            bool_type_id][src_tp[1].get_type_id() -            \
                                          bool_type_id];                       \
          create(ckb, kernreq, ckb_offset);                                    \
          return ckb_offset;                                                   \
        }                                                                      \
      }                                                                        \
                                                                               \
      std::stringstream ss;                                                    \
      ss << "arithmetic is not yet implemented for types " << src_tp[0]        \
         << " and " << src_tp[1];                                              \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
                                                                               \
    static arrfunc make()                                                      \
    {                                                                          \
      arrfunc child_af(ndt::type("(Any, Any) -> Any"), &instantiate, NULL,     \
                       &resolve_dst_type);                                     \
                                                                               \
      return functional::elwise(child_af);                                     \
    }                                                                          \
  } NAME;

  ARITHMETIC_OPERATOR(add);
  ARITHMETIC_OPERATOR(sub);
  ARITHMETIC_OPERATOR(mul);
  ARITHMETIC_OPERATOR(div);

#undef ARITHMETIC_OPERATOR

} // namespace nd
} // namespace dynd