//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/strided_vals.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/pp/meta.hpp>
#include <dynd/pp/list.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/types/base_struct_type.hpp>

namespace dynd { namespace detail {

template <typename T>
class typed_arg {
public:
  typed_arg(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(arrmeta),
            const nd::array &DYND_UNUSED(kwds))
  {
  }

  T &val(char *data) { return *reinterpret_cast<T *>(data); }
};

template <typename T, int N>
class typed_arg<nd::strided_vals<T, N> > {
  nd::strided_vals<T, N> m_vals;

public:
  typed_arg(const ndt::type &DYND_UNUSED(tp), const char *arrmeta,
            const nd::array &kwds)
  {
    m_vals.set_data(
        NULL, reinterpret_cast<const size_stride_t *>(arrmeta),
        reinterpret_cast<start_stop_t *>(kwds.p("start_stop").as<intptr_t>()));

    ndt::type dt = kwds.get_dtype();
    // TODO: Remove all try/catch(...) in the code
    try {
      const nd::array &mask = kwds.p("mask").f("dereference");
      m_vals.set_mask(
          mask.get_readonly_originptr(),
          reinterpret_cast<const size_stride_t *>(mask.get_arrmeta()));
    }
    catch (...) {
      m_vals.set_mask(NULL);
    }
  }

  nd::strided_vals<T, N> &val(char *data)
  {
    m_vals.set_data(data);
    return m_vals;
  }
};

} // namespace detail

#define DECL_TYPED_ARG(TYPENAME, NAME)                                         \
  DYND_PP_META_DECL(detail::typed_arg<TYPENAME>, NAME)
#define PARTIAL_DECAY(TYPENAME)                                                \
  std::remove_cv<typename std::remove_reference<TYPENAME>::type>::type
#define PASS(NAME, ARG) NAME.val(ARG)
#define AS(NAME, TYPE) NAME.as<TYPE>()
#define COPY_CONSTRUCT(NAME) NAME(NAME)
#define CONSTRUCT_TYPED_ARG(TYPENAME, TP, ARRMETA)                             \
  detail::typed_arg<TYPENAME>(TP, ARRMETA, kwds)

template <typename func_type, typename arrfunc_type, int naux, bool construct>
struct functor_ck;

#define FUNCTOR_CK(NARG)                                                       \
  DYND_PP_JOIN_ELWISE_1(_FUNCTOR_CK, (), DYND_PP_RANGE(DYND_PP_INC(NARG)),     \
                        DYND_PP_REPEAT(NARG, DYND_PP_INC(NARG)))
#define _FUNCTOR_CK(NSRC, NARG)                                                \
  __FUNCTOR_CK(NSRC, DYND_PP_SUB(NARG, NSRC), NARG)

/**
 * This generates code to instantiate a ckernel calling a C++ function
 * object which returns a value.
 */
#define __FUNCTOR_CK(NSRC, NAUX, NARG)                                         \
  template <typename func_type,                                                \
            DYND_PP_JOIN_MAP_2(                                                \
                DYND_PP_META_TYPENAME, (, ),                                   \
                DYND_PP_PREPEND(R, DYND_PP_META_NAME_RANGE(A, NARG)))>         \
  struct functor_ck<func_type, R DYND_PP_META_NAME_RANGE(A, NARG), NAUX,       \
                    false>                                                     \
      : kernels::expr_ck<                                                      \
            functor_ck<func_type, R DYND_PP_META_NAME_RANGE(A, NARG), NAUX,    \
                       false>,                                                 \
            NARG> {                                                            \
    typedef functor_ck self_type;                                              \
    DYND_PP_JOIN_ELWISE_2(DYND_PP_META_TYPEDEF_TYPENAME, (;),                  \
                          DYND_PP_MAP_2(PARTIAL_DECAY,                         \
                                        DYND_PP_META_NAME_RANGE(A, NSRC)),     \
                          DYND_PP_META_NAME_RANGE(D, NSRC));                   \
                                                                               \
    func_type func;                                                            \
    DYND_PP_JOIN_ELWISE_2(DECL_TYPED_ARG, (;),                                 \
                          DYND_PP_META_NAME_RANGE(D, NSRC),                    \
                          DYND_PP_META_NAME_RANGE(src, NSRC));                 \
    DYND_PP_JOIN_ELWISE_2(DYND_PP_META_DECL, (;),                              \
                          DYND_PP_META_NAME_RANGE(A, NSRC, NARG),              \
                          DYND_PP_META_NAME_RANGE(aux, NAUX));                 \
                                                                               \
    functor_ck                                                                 \
    DYND_PP_CHAIN((const func_type &func),                                     \
                  DYND_PP_ELWISE_2(DECL_TYPED_ARG,                             \
                                   DYND_PP_META_NAME_RANGE(D, NSRC),           \
                                   DYND_PP_META_NAME_RANGE(src, NSRC)),        \
                  DYND_PP_ELWISE_2(DYND_PP_META_DECL,                          \
                                   DYND_PP_META_NAME_RANGE(A, NSRC, NARG),     \
                                   DYND_PP_META_NAME_RANGE(aux, NAUX)))        \
        : DYND_PP_JOIN_MAP_2(                                                  \
              COPY_CONSTRUCT, (, ),                                            \
              DYND_PP_CHAIN((func), DYND_PP_META_NAME_RANGE(src, NSRC),        \
                            DYND_PP_META_NAME_RANGE(aux, NAUX)))               \
    {                                                                          \
    }                                                                          \
                                                                               \
    void single(char *dst, char **DYND_PP_IF(NSRC)(src))                       \
    {                                                                          \
      *reinterpret_cast<R *>(dst) = func DYND_PP_CHAIN(                        \
          DYND_PP_ELWISE_2(PASS, DYND_PP_META_NAME_RANGE(src, NSRC),           \
                           DYND_PP_META_AT_RANGE(src, NSRC)),                  \
          DYND_PP_META_NAME_RANGE(aux, NAUX));                                 \
    }                                                                          \
                                                                               \
    void strided(char *dst, intptr_t dst_stride, char **DYND_PP_IF(NSRC)(src), \
                 const intptr_t *DYND_PP_IF(NSRC)(src_stride), size_t count)   \
    {                                                                          \
      DYND_PP_JOIN_ELWISE_2(DYND_PP_META_DECL_ASGN, (;),                       \
                            DYND_PP_REPEAT_2(char *, NSRC),                    \
                            DYND_PP_META_NAME_RANGE(src, NSRC),                \
                            DYND_PP_META_AT_RANGE(src, NSRC));                 \
      DYND_PP_JOIN_ELWISE_2(DYND_PP_META_DECL_ASGN, (;),                       \
                            DYND_PP_REPEAT_1(intptr_t, NSRC),                  \
                            DYND_PP_META_NAME_RANGE(src_stride, NSRC),         \
                            DYND_PP_META_AT_RANGE(src_stride, NSRC));          \
      for (size_t i = 0; i < count; ++i) {                                     \
        *reinterpret_cast<R *>(dst) = func DYND_PP_CHAIN(                      \
            DYND_PP_ELWISE_2(PASS, DYND_PP_META_NAME_RANGE(this->src, NSRC),   \
                             DYND_PP_META_NAME_RANGE(src, NSRC)),              \
            DYND_PP_META_NAME_RANGE(aux, NAUX));                               \
        dst += dst_stride;                                                     \
        DYND_PP_JOIN_ELWISE_2(DYND_PP_META_ADD_ASGN, (;),                      \
                              DYND_PP_META_NAME_RANGE(src, NSRC),              \
                              DYND_PP_META_NAME_RANGE(src_stride, NSRC));      \
      }                                                                        \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *af_self, const arrfunc_type *af_tp,   \
                dynd::ckernel_builder *ckb, intptr_t ckb_offset,               \
                const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), \
                const ndt::type *src_tp,                                       \
                const char *const *DYND_PP_IF(NSRC)(src_arrmeta),              \
                kernel_request_t kernreq,                                      \
                const eval::eval_context *DYND_UNUSED(ectx),                   \
                const nd::array &DYND_PP_IF(NAUX)(args),                       \
                const nd::array &DYND_PP_IF(NSRC)(kwds))                       \
    {                                                                          \
      for (intptr_t i = 0; i < NSRC; ++i) {                                    \
        if (src_tp[i] != af_tp->get_arg_type(i)) {                             \
          std::stringstream ss;                                                \
          ss << "Provided types " << ndt::make_funcproto(NSRC, src_tp, dst_tp) \
             << " do not match the arrfunc proto " << af_tp;                   \
          throw type_error(ss.str());                                          \
        }                                                                      \
      }                                                                        \
      if (dst_tp != af_tp->get_return_type()) {                                \
        std::stringstream ss;                                                  \
        ss << "Provided types " << ndt::make_funcproto(NSRC, src_tp, dst_tp)   \
           << " do not match the arrfunc proto " << af_tp;                     \
        throw type_error(ss.str());                                            \
      }                                                                        \
                                                                               \
      self_type::create DYND_PP_CHAIN(                                         \
          (ckb, kernreq, ckb_offset, *af_self->get_data_as<func_type>()),      \
          DYND_PP_ELWISE_2(CONSTRUCT_TYPED_ARG,                                \
                           DYND_PP_META_NAME_RANGE(D, NSRC),                   \
                           DYND_PP_META_AT_RANGE(src_tp, NSRC),                \
                           DYND_PP_META_AT_RANGE(src_arrmeta, NSRC)),          \
          DYND_PP_ELWISE_2(                                                    \
              AS, DYND_PP_META_CALL_RANGE(args, NAUX),                         \
              DYND_PP_META_NAME_RANGE(A, NSRC, DYND_PP_INC(NARG))));           \
      return ckb_offset;                                                       \
    }                                                                          \
  };

DYND_PP_JOIN_MAP(FUNCTOR_CK, (), DYND_PP_RANGE(DYND_PP_INC(DYND_ARG_MAX)))

#undef __FUNCTOR_CK

#define __FUNCTOR_CK(NSRC, NAUX, NARG)                                         \
  template <typename func_type,                                                \
            DYND_PP_JOIN_MAP_2(                                                \
                DYND_PP_META_TYPENAME, (, ),                                   \
                DYND_PP_PREPEND(R, DYND_PP_META_NAME_RANGE(A, NARG)))>         \
  struct functor_ck<func_type, R DYND_PP_META_NAME_RANGE(A, NARG), NAUX, true> \
      : kernels::expr_ck<                                                      \
            functor_ck<func_type, R DYND_PP_META_NAME_RANGE(A, NARG), NAUX,    \
                       true>,                                                  \
            NARG> {                                                            \
    typedef functor_ck self_type;                                              \
    DYND_PP_JOIN_ELWISE_2(DYND_PP_META_TYPEDEF_TYPENAME, (;),                  \
                          DYND_PP_MAP_2(PARTIAL_DECAY,                         \
                                        DYND_PP_META_NAME_RANGE(A, NSRC)),     \
                          DYND_PP_META_NAME_RANGE(D, NSRC));                   \
                                                                               \
    func_type func;                                                            \
    DYND_PP_JOIN_ELWISE_2(DECL_TYPED_ARG, (;),                                 \
                          DYND_PP_META_NAME_RANGE(D, NSRC),                    \
                          DYND_PP_META_NAME_RANGE(src, NSRC));                 \
                                                                               \
    functor_ck                                                                 \
        DYND_PP_CHAIN(DYND_PP_ELWISE_2(DECL_TYPED_ARG,                         \
                                       DYND_PP_META_NAME_RANGE(D, NSRC),       \
                                       DYND_PP_META_NAME_RANGE(src, NSRC)),    \
                      DYND_PP_ELWISE_2(DYND_PP_META_DECL,                      \
                                       DYND_PP_META_NAME_RANGE(A, NSRC, NARG), \
                                       DYND_PP_META_NAME_RANGE(aux, NAUX)))    \
        : DYND_PP_JOIN_2(                                                      \
              (, ), DYND_PP_PREPEND(                                           \
                        func DYND_PP_META_NAME_RANGE(aux, NAUX),               \
                        DYND_PP_MAP_2(COPY_CONSTRUCT,                          \
                                      DYND_PP_META_NAME_RANGE(src, NSRC))))    \
    {                                                                          \
    }                                                                          \
                                                                               \
    void single(char *dst, char **DYND_PP_IF(NSRC)(src))                       \
    {                                                                          \
      *reinterpret_cast<R *>(dst) =                                            \
          func DYND_PP_ELWISE_2(PASS, DYND_PP_META_NAME_RANGE(src, NSRC),      \
                                DYND_PP_META_AT_RANGE(src, NSRC));             \
    }                                                                          \
                                                                               \
    void strided(char *dst, intptr_t dst_stride, char **DYND_PP_IF(NSRC)(src), \
                 const intptr_t *DYND_PP_IF(NSRC)(src_stride), size_t count)   \
    {                                                                          \
      DYND_PP_JOIN_ELWISE_2(DYND_PP_META_DECL_ASGN, (;),                       \
                            DYND_PP_REPEAT_2(char *, NSRC),                    \
                            DYND_PP_META_NAME_RANGE(src, NSRC),                \
                            DYND_PP_META_AT_RANGE(src, NSRC));                 \
      DYND_PP_JOIN_ELWISE_2(DYND_PP_META_DECL_ASGN, (;),                       \
                            DYND_PP_REPEAT_1(intptr_t, NSRC),                  \
                            DYND_PP_META_NAME_RANGE(src_stride, NSRC),         \
                            DYND_PP_META_AT_RANGE(src_stride, NSRC));          \
      for (size_t i = 0; i < count; ++i) {                                     \
        *reinterpret_cast<R *>(dst) = func DYND_PP_ELWISE_2(                   \
            PASS, DYND_PP_META_NAME_RANGE(this->src, NSRC),                    \
            DYND_PP_META_NAME_RANGE(src, NSRC));                               \
        dst += dst_stride;                                                     \
        DYND_PP_JOIN_ELWISE_2(DYND_PP_META_ADD_ASGN, (;),                      \
                              DYND_PP_META_NAME_RANGE(src, NSRC),              \
                              DYND_PP_META_NAME_RANGE(src_stride, NSRC));      \
      }                                                                        \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *af_self, const arrfunc_type *af_tp,   \
                dynd::ckernel_builder *ckb, intptr_t ckb_offset,               \
                const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), \
                const ndt::type *src_tp,                                       \
                const char *const *DYND_PP_IF(NSRC)(src_arrmeta),              \
                kernel_request_t kernreq,                                      \
                const eval::eval_context *DYND_UNUSED(ectx),                   \
                const nd::array &DYND_PP_IF(NAUX)(args),                       \
                const nd::array &DYND_PP_IF(NSRC)(kwds))                       \
    {                                                                          \
      for (intptr_t i = 0; i < NSRC; ++i) {                                    \
        if (src_tp[i] != af_tp->get_arg_type(i)) {                             \
          std::stringstream ss;                                                \
          ss << "Provided types " << ndt::make_funcproto(NSRC, src_tp, dst_tp) \
             << " do not match the arrfunc proto " << af_tp;                   \
          throw type_error(ss.str());                                          \
        }                                                                      \
      }                                                                        \
      if (dst_tp != af_tp->get_return_type()) {                                \
        std::stringstream ss;                                                  \
        ss << "Provided types " << ndt::make_funcproto(NSRC, src_tp, dst_tp)   \
           << " do not match the arrfunc proto " << af_tp;                     \
        throw type_error(ss.str());                                            \
      }                                                                        \
                                                                               \
      self_type::create DYND_PP_CHAIN(                                         \
          (ckb, kernreq, ckb_offset),                                          \
          DYND_PP_ELWISE_2(CONSTRUCT_TYPED_ARG,                                \
                           DYND_PP_META_NAME_RANGE(D, NSRC),                   \
                           DYND_PP_META_AT_RANGE(src_tp, NSRC),                \
                           DYND_PP_META_AT_RANGE(src_arrmeta, NSRC)),          \
          DYND_PP_ELWISE_2(                                                    \
              AS, DYND_PP_META_CALL_RANGE(args, NAUX),                         \
              DYND_PP_META_NAME_RANGE(A, NSRC, DYND_PP_INC(NARG))));           \
      return ckb_offset;                                                       \
    }                                                                          \
  };

DYND_PP_JOIN_MAP(FUNCTOR_CK, (), DYND_PP_RANGE(DYND_PP_INC(DYND_ARG_MAX)))

#undef __FUNCTOR_CK

/**
 * This generates code to instantiate a ckernel calling a C++ function
 * object which returns a value in the first parameter as an output reference.
 */
#define __FUNCTOR_CK(NSRC, NAUX, NARG)                                         \
  template <typename func_type,                                                \
            DYND_PP_JOIN_MAP_2(                                                \
                DYND_PP_META_TYPENAME, (, ),                                   \
                DYND_PP_PREPEND(R, DYND_PP_META_NAME_RANGE(A, NARG)))>         \
  struct functor_ck<func_type, void DYND_PP_PREPEND(                           \
                                   R &, DYND_PP_META_NAME_RANGE(A, NARG)),     \
                    NAUX, false>                                               \
      : kernels::expr_ck<                                                      \
            functor_ck<func_type, void DYND_PP_PREPEND(                        \
                                      R &, DYND_PP_META_NAME_RANGE(A, NARG)),  \
                       NAUX, false>,                                           \
            NARG> {                                                            \
    typedef functor_ck self_type;                                              \
    DYND_PP_JOIN_ELWISE_2(DYND_PP_META_TYPEDEF_TYPENAME, (;),                  \
                          DYND_PP_MAP_2(PARTIAL_DECAY,                         \
                                        DYND_PP_META_NAME_RANGE(A, NSRC)),     \
                          DYND_PP_META_NAME_RANGE(D, NSRC));                   \
                                                                               \
    func_type func;                                                            \
    DYND_PP_JOIN_ELWISE_2(                                                     \
        DECL_TYPED_ARG, (;),                                                   \
        DYND_PP_PREPEND(R, DYND_PP_META_NAME_RANGE(D, NSRC)),                  \
        DYND_PP_PREPEND(dst, DYND_PP_META_NAME_RANGE(src, NSRC)));             \
    DYND_PP_JOIN_ELWISE_2(DYND_PP_META_DECL, (;),                              \
                          DYND_PP_META_NAME_RANGE(A, NSRC, NARG),              \
                          DYND_PP_META_NAME_RANGE(aux, NAUX));                 \
                                                                               \
    functor_ck DYND_PP_CHAIN(                                                  \
        (const func_type &func),                                               \
        DYND_PP_ELWISE_2(DECL_TYPED_ARG,                                       \
                         DYND_PP_PREPEND(R, DYND_PP_META_NAME_RANGE(D, NSRC)), \
                         DYND_PP_PREPEND(dst,                                  \
                                         DYND_PP_META_NAME_RANGE(src, NSRC))), \
        DYND_PP_ELWISE_2(DYND_PP_META_DECL,                                    \
                         DYND_PP_META_NAME_RANGE(A, NSRC, NARG),               \
                         DYND_PP_META_NAME_RANGE(aux, NAUX)))                  \
        : DYND_PP_JOIN_MAP_2(                                                  \
              COPY_CONSTRUCT, (, ),                                            \
              DYND_PP_CHAIN((func, dst), DYND_PP_META_NAME_RANGE(src, NSRC),   \
                            DYND_PP_META_NAME_RANGE(aux, NAUX)))               \
    {                                                                          \
    }                                                                          \
                                                                               \
    void single(char *dst, char **DYND_PP_IF(NSRC)(src))                       \
    {                                                                          \
      func DYND_PP_CHAIN(                                                      \
          DYND_PP_ELWISE_2(                                                    \
              PASS,                                                            \
              DYND_PP_PREPEND(this->dst, DYND_PP_META_NAME_RANGE(src, NSRC)),  \
              DYND_PP_PREPEND(dst, DYND_PP_META_AT_RANGE(src, NSRC))),         \
          DYND_PP_META_NAME_RANGE(aux, NAUX));                                 \
    }                                                                          \
                                                                               \
    void strided(char *dst, intptr_t dst_stride, char **DYND_PP_IF(NSRC)(src), \
                 const intptr_t *DYND_PP_IF(NSRC)(src_stride), size_t count)   \
    {                                                                          \
      DYND_PP_JOIN_ELWISE_2(DYND_PP_META_DECL_ASGN, (;),                       \
                            DYND_PP_REPEAT_2(char *, NSRC),                    \
                            DYND_PP_META_NAME_RANGE(src, NSRC),                \
                            DYND_PP_META_AT_RANGE(src, NSRC));                 \
      DYND_PP_JOIN_ELWISE_2(DYND_PP_META_DECL_ASGN, (;),                       \
                            DYND_PP_REPEAT_1(intptr_t, NSRC),                  \
                            DYND_PP_META_NAME_RANGE(src_stride, NSRC),         \
                            DYND_PP_META_AT_RANGE(src_stride, NSRC));          \
      for (size_t i = 0; i < count; ++i) {                                     \
        func DYND_PP_CHAIN(                                                    \
            DYND_PP_ELWISE_2(                                                  \
                PASS, DYND_PP_PREPEND(this->dst, DYND_PP_META_NAME_RANGE(      \
                                                     this->src, NSRC)),        \
                DYND_PP_PREPEND(dst, DYND_PP_META_NAME_RANGE(src, NSRC))),     \
            DYND_PP_META_NAME_RANGE(aux, NAUX));                               \
        dst += dst_stride;                                                     \
        DYND_PP_JOIN_ELWISE_2(DYND_PP_META_ADD_ASGN, (;),                      \
                              DYND_PP_META_NAME_RANGE(src, NSRC),              \
                              DYND_PP_META_NAME_RANGE(src_stride, NSRC));      \
      }                                                                        \
    }                                                                          \
                                                                               \
    static intptr_t instantiate(                                               \
        const arrfunc_type_data *af_self, const arrfunc_type *af_tp,           \
        dynd::ckernel_builder *ckb, intptr_t ckb_offset,                       \
        const ndt::type &dst_tp, const char *dst_arrmeta,                      \
        const ndt::type *src_tp,                                               \
        const char *const *DYND_PP_IF(NSRC)(src_arrmeta),                      \
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx), \
        const nd::array &DYND_PP_IF(NAUX)(args), const nd::array &kwds)        \
    {                                                                          \
      for (intptr_t i = 0; i < NSRC; ++i) {                                    \
        if (src_tp[i] != af_tp->get_arg_type(i)) {                             \
          std::stringstream ss;                                                \
          ss << "Provided types " << ndt::make_funcproto(NSRC, src_tp, dst_tp) \
             << " do not match the arrfunc proto " << af_tp;                   \
          throw type_error(ss.str());                                          \
        }                                                                      \
      }                                                                        \
      if (dst_tp != af_tp->get_return_type()) {                                \
        std::stringstream ss;                                                  \
        ss << "Provided types " << ndt::make_funcproto(NSRC, src_tp, dst_tp)   \
           << " do not match the arrfunc proto " << af_tp;                     \
        throw type_error(ss.str());                                            \
      }                                                                        \
                                                                               \
      self_type::create DYND_PP_CHAIN(                                         \
          (ckb, kernreq, ckb_offset, *af_self->get_data_as<func_type>(),       \
           detail::typed_arg<R>(dst_tp, dst_arrmeta, kwds)),                   \
          DYND_PP_ELWISE_2(CONSTRUCT_TYPED_ARG,                                \
                           DYND_PP_META_NAME_RANGE(D, NSRC),                   \
                           DYND_PP_META_AT_RANGE(src_tp, NSRC),                \
                           DYND_PP_META_AT_RANGE(src_arrmeta, NSRC)),          \
          DYND_PP_ELWISE_2(                                                    \
              AS, DYND_PP_META_CALL_RANGE(args, NAUX),                         \
              DYND_PP_META_NAME_RANGE(A, NSRC, DYND_PP_INC(NARG))));           \
      return ckb_offset;                                                       \
    }                                                                          \
  };

DYND_PP_JOIN_MAP(FUNCTOR_CK, (), DYND_PP_RANGE(DYND_PP_INC(DYND_ARG_MAX)))

#undef __FUNCTOR_CK
#undef _FUNCTOR_CK
#undef FUNCTOR_CK

#undef DECL_TYPED_ARG
#undef INIT_TYPED_PARAM_FROM_BYTES
#undef PARTIAL_DECAY
#undef PASS

} // namespace dynd
