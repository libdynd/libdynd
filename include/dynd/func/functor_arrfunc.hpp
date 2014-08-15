//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND__FUNC_FUNCTOR_ARRFUNC_HPP
#define DYND__FUNC_FUNCTOR_ARRFUNC_HPP

#include <dynd/types/funcproto_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/pp/list.hpp>
#include <dynd/pp/meta.hpp>

// TODO: Move some/all of these into the PP library

/**
 * Generates comma-separated names typically used for arguments.
 *
 * NAME0, NAME1, ...
 */
#define DYND_PP_ARG_RANGE_1(NAME, N)                                           \
  DYND_PP_JOIN_1((, ), DYND_PP_META_NAME_RANGE(NAME, N))

/**
 * Applies a template metafunction to a range of named types.
 *
 * (TMPFUNC<NAME0>::value, TMPFUNC<NAME1>::value, ...)
 */
#define DYND_PP_APPLY_TMPFUNC_RANGE_1(TMPFUNC, NAME, N)                        \
  DYND_PP_ELWISE_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE,                  \
                   DYND_PP_REPEAT(TMPFUNC, N),                                 \
                   DYND_PP_META_NAME_RANGE(NAME, N), DYND_PP_REPEAT(value, N))

/**
 * Greates a range of named types which are cleaned versions of an input
 * range of named types.
 *
 * typedef remove_const<remove_reference<TYPE0>::type>::type CLEANED_TYPE0;
 * typedef remove_const<remove_reference<TYPE1>::type>::type CLEANED_TYPE1;
 * ...
 */
#define DYND_PP_CLEAN_TYPE_RANGE_1(CLEANED_TYPE, TYPE, N)                      \
  DYND_PP_JOIN_ELWISE_1(                                                       \
      DYND_PP_META_TYPEDEF_TYPENAME, (;),                                      \
      DYND_PP_ELWISE_1(                                                        \
          DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE,                           \
          DYND_PP_REPEAT(remove_const, N),                                     \
          DYND_PP_ELWISE_1(DYND_PP_META_TYPENAME_TEMPLATE_INSTANTIATION_SCOPE, \
                           DYND_PP_REPEAT(typename remove_reference, N),       \
                           DYND_PP_META_NAME_RANGE(TYPE, N),                   \
                           DYND_PP_REPEAT(type, N)),                           \
          DYND_PP_REPEAT(type, N)),                                            \
      DYND_PP_META_NAME_RANGE(CLEANED_TYPE, N));

/**
 * Static_assert that a template metafunction returns true for each type in a
 * range of named types.
 *
 * static_assert(TMPFUNC<NAME0>::value, MSG);
 * static_assert(TMPFUNC<NAME1>::value, MSG);
 * ...
 */
#define DYND_PP_STATIC_ASSERT_RANGE_1(MSG, TMPFUNC, NAME, N)                   \
  DYND_PP_JOIN_ELWISE_1(DYND_PP_META_STATIC_ASSERT, (;),                       \
                        DYND_PP_APPLY_TMPFUNC_RANGE_1(TMPFUNC, NAME, N),       \
                        DYND_PP_REPEAT(MSG, N));

/**
 * For generating ckernel function calls, casts each ``ARG_NAME[#]`` input to
 * the type ``TYPE#`` and dereferences it, output with a comma separator.
 *
 * *reinterpret_cast<const TYPE0 *>(ARG_NAME[0]), ...",
 * *reinterpret_cast<const TYPE1 *>(ARG_NAME[1]), ...",
 * ...
 */
#define DYND_PP_DEREF_CAST_ARRAY_RANGE_1(TYPE, ARG_NAME, N)                    \
  DYND_PP_JOIN_MAP_1(                                                          \
      DYND_PP_META_DEREFERENCE, (, ),                                          \
      DYND_PP_ELWISE_1(DYND_PP_META_REINTERPRET_CAST,                          \
                       DYND_PP_MAP_1(DYND_PP_META_MAKE_CONST_PTR,              \
                                     DYND_PP_META_NAME_RANGE(TYPE, N)),        \
                       DYND_PP_META_AT_RANGE(ARG_NAME, N)))

/**
 * For generating ckernel function calls, casts each ``ARG_NAME#`` input to
 * the type ``TYPE#`` and dereferences it, output with a comma separator.
 *
 * *reinterpret_cast<const TYPE0 *>(ARG_NAME0), ...",
 * *reinterpret_cast<const TYPE1 *>(ARG_NAME1), ...",
 * ...
 */
#define DYND_PP_DEREF_CAST_ARG_RANGE_1(TYPE, ARG_NAME, N)                      \
  DYND_PP_JOIN_MAP_1(                                                          \
      DYND_PP_META_DEREFERENCE, (, ),                                          \
      DYND_PP_ELWISE_1(DYND_PP_META_REINTERPRET_CAST,                          \
                       DYND_PP_MAP_1(DYND_PP_META_MAKE_CONST_PTR,              \
                                     DYND_PP_META_NAME_RANGE(TYPE, N)),        \
                       DYND_PP_META_NAME_RANGE(ARG_NAME, N)))

/**
 * Declares a range of ``TYPE`` variables ``ARG_NAME#`` and
 * initializes them to ``ARG_NAME[#]``.
 *
 * const char *ARG_NAME0 = ARG_NAME[0];
 * const char *ARG_NAME1 = ARG_NAME[1];
 * ...
 */
#define DYND_PP_VARIABLE_RANGE_FROM_ARRAY_1(TYPE, ARG_NAME, N)                 \
  DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;),                           \
                        DYND_PP_REPEAT_1(TYPE, N),                             \
                        DYND_PP_META_NAME_RANGE(ARG_NAME, N),                  \
                        DYND_PP_META_AT_RANGE(ARG_NAME, N));

/**
 * Initializes ``src#`` and ``src_stride#`` variables from the ``src`` and
 * ``src_stride`` input arrays.
 *
 * const char *src0 = src[0];
 * const char *src1 = src[1];
 * ...
 * intptr_t src_stride0 = src_stride[0];
 * intptr_t src_stride1 = src_stride[1];
 * ...
 *
 */
#define DYND_PP_INIT_SRC_VARIABLES(N)                                          \
  DYND_PP_VARIABLE_RANGE_FROM_ARRAY_1(const char *, src, N);                   \
  DYND_PP_VARIABLE_RANGE_FROM_ARRAY_1(intptr_t, src_stride, N);

/**
 * Increment ``dst``, ``src0``, etc. with ``dst_stride``, ``src_stride0``, etc.
 *
 * dst += dst_stride;
 * src0 += src_stride0;
 * src1 += src_stride1;
 */
#define DYND_PP_STRIDED_INCREMENT(N)                                           \
  dst += dst_stride;                                                           \
  DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ADD_ASGN, (;),                            \
                        DYND_PP_META_NAME_RANGE(src, N),                       \
                        DYND_PP_META_NAME_RANGE(src_stride, N));

/**
 * Create ``dst_tp`` and ``src_tp[N]``, etc. from the types DST_TYPE, SRC_TYPE0,
 * etc.
 *
 */
#define DYND_PP_NDT_TYPES_FROM_TYPES(DST_TYPE, SRC_TYPE, N)                    \
  ndt::type dst_tp = ndt::cfixed_dim_from_array<DST_TYPE>::make();             \
  ndt::type src_tp[N] = {DYND_PP_JOIN_ELWISE_1(                                \
      DYND_PP_META_SCOPE_CALL, (, ),                                           \
      DYND_PP_ELWISE_1(DYND_PP_META_TEMPLATE_INSTANTIATION,                    \
                       DYND_PP_REPEAT_1(ndt::cfixed_dim_from_array, N),        \
                       DYND_PP_META_NAME_RANGE(SRC_TYPE, N)),                  \
      DYND_PP_REPEAT_1(make, N))};

namespace dynd { namespace nd {

namespace detail {

  template <typename Functor, typename FuncProto>
  struct functor_ckernel_instantiator;

/**
 * This generates code to instantiate a ckernel calling a C++ function
 * object which returns a value.
 */
#define DYND_CODE(NSRC)                                                        \
  template <typename Functor, typename R,                                      \
            DYND_PP_ARG_RANGE_1(typename A, NSRC)>                             \
  struct functor_ckernel_instantiator<Functor,                                 \
                                      R (*)(DYND_PP_ARG_RANGE_1(A, NSRC))> {   \
    typedef functor_ckernel_instantiator self_type;                            \
                                                                               \
    DYND_PP_CLEAN_TYPE_RANGE_1(D, A, NSRC);                                    \
                                                                               \
    ckernel_prefix base;                                                       \
    Functor func;                                                              \
                                                                               \
    static void single(char *dst, const char *const *src, ckernel_prefix *ckp) \
    {                                                                          \
      self_type *e = reinterpret_cast<self_type *>(ckp);                       \
      *reinterpret_cast<R *>(dst) =                                            \
          e->func(DYND_PP_DEREF_CAST_ARRAY_RANGE_1(D, src, NSRC));             \
    }                                                                          \
                                                                               \
    static void strided(char *dst, intptr_t dst_stride,                        \
                        const char *const *src, const intptr_t *src_stride,    \
                        size_t count, ckernel_prefix *ckp)                     \
    {                                                                          \
      self_type *e = reinterpret_cast<self_type *>(ckp);                       \
      Functor func = e->func;                                                  \
      /* const char *src# = src[#]; */                                         \
      /* intptr_t src_stride# = src_stride[#]; */                              \
      DYND_PP_INIT_SRC_VARIABLES(NSRC);                                        \
      for (size_t i = 0; i < count; ++i) {                                     \
        /* *(R *)dst = func(*(const D0 *)src0, ...); */                        \
        *reinterpret_cast<R *>(dst) =                                          \
            func(DYND_PP_DEREF_CAST_ARG_RANGE_1(D, src, NSRC));                \
                                                                               \
        /* Increment ``dst``, ``src#`` by their respective strides */          \
        DYND_PP_STRIDED_INCREMENT(NSRC);                                       \
      }                                                                        \
    }                                                                          \
                                                                               \
    static intptr_t instantiate(const arrfunc_type_data *af_self,              \
                                dynd::ckernel_builder *ckb,                    \
                                intptr_t ckb_offset, const ndt::type &dst_tp,  \
                                const char *DYND_UNUSED(dst_arrmeta),          \
                                const ndt::type *src_tp,                       \
                                const char *const *DYND_UNUSED(src_arrmeta),   \
                                kernel_request_t kernreq,                      \
                                const eval::eval_context *DYND_UNUSED(ectx))   \
    {                                                                          \
      for (intptr_t i = 0; i < NSRC; ++i) {                                    \
        if (src_tp[i] != af_self->get_param_type(i)) {                         \
          std::stringstream ss;                                                \
          ss << "Provided types " << ndt::make_funcproto(1, src_tp, dst_tp)    \
             << " do not match the arrfunc proto " << af_self->func_proto;     \
          throw type_error(ss.str());                                          \
        }                                                                      \
      }                                                                        \
      if (dst_tp != af_self->get_return_type()) {                              \
        std::stringstream ss;                                                  \
        ss << "Provided types " << ndt::make_funcproto(1, src_tp, dst_tp)      \
           << " do not match the arrfunc proto " << af_self->func_proto;       \
        throw type_error(ss.str());                                            \
      }                                                                        \
      self_type *e = ckb->alloc_ck_leaf<self_type>(ckb_offset);                \
      e->base.template set_expr_function<self_type>(kernreq);                  \
      e->func = *af_self->get_data_as<Functor>();                              \
                                                                               \
      return ckb_offset;                                                       \
    }                                                                          \
  };

DYND_PP_JOIN_MAP(DYND_CODE, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))
#undef DYND_CODE

/**
 * This generates code to instantiate a ckernel calling a C++ function
 * object which returns a value in the first parameter as an output reference.
 */
#define DYND_CODE(NSRC)                                                        \
  template <typename Functor, typename R,                                      \
            DYND_PP_ARG_RANGE_1(typename A, NSRC)>                             \
  struct functor_ckernel_instantiator<                                         \
      Functor, void (*)(R &, DYND_PP_ARG_RANGE_1(A, NSRC))> {                  \
    typedef functor_ckernel_instantiator self_type;                            \
                                                                               \
    DYND_PP_CLEAN_TYPE_RANGE_1(D, A, NSRC);                                    \
                                                                               \
    ckernel_prefix base;                                                       \
    Functor func;                                                              \
                                                                               \
    static void single(char *dst, const char *const *src, ckernel_prefix *ckp) \
    {                                                                          \
      self_type *e = reinterpret_cast<self_type *>(ckp);                       \
      e->func(*reinterpret_cast<R *>(dst),                                     \
              DYND_PP_DEREF_CAST_ARRAY_RANGE_1(D, src, NSRC));                 \
    }                                                                          \
                                                                               \
    static void strided(char *dst, intptr_t dst_stride,                        \
                        const char *const *src, const intptr_t *src_stride,    \
                        size_t count, ckernel_prefix *ckp)                     \
    {                                                                          \
      self_type *e = reinterpret_cast<self_type *>(ckp);                       \
      Functor func = e->func;                                                  \
      /* const char *src# = src[#]; */                                         \
      /* intptr_t src_stride# = src_stride[#]; */                              \
      DYND_PP_INIT_SRC_VARIABLES(NSRC);                                        \
      for (size_t i = 0; i < count; ++i) {                                     \
        /*  func(*(R *)dst, *(const D0 *)src0, ...); */                        \
        func(*reinterpret_cast<R *>(dst),                                      \
             DYND_PP_DEREF_CAST_ARG_RANGE_1(D, src, NSRC));                    \
                                                                               \
        /* Increment ``dst``, ``src#`` by their respective strides */          \
        DYND_PP_STRIDED_INCREMENT(NSRC);                                       \
      }                                                                        \
    }                                                                          \
                                                                               \
    static intptr_t instantiate(const arrfunc_type_data *af_self,              \
                                dynd::ckernel_builder *ckb,                    \
                                intptr_t ckb_offset, const ndt::type &dst_tp,  \
                                const char *DYND_UNUSED(dst_arrmeta),          \
                                const ndt::type *src_tp,                       \
                                const char *const *DYND_UNUSED(src_arrmeta),   \
                                kernel_request_t kernreq,                      \
                                const eval::eval_context *DYND_UNUSED(ectx))   \
    {                                                                          \
      for (intptr_t i = 0; i < NSRC; ++i) {                                    \
        if (src_tp[i] != af_self->get_param_type(i)) {                         \
          std::stringstream ss;                                                \
          ss << "Provided types " << ndt::make_funcproto(1, src_tp, dst_tp)    \
             << " do not match the arrfunc proto " << af_self->func_proto;     \
          throw type_error(ss.str());                                          \
        }                                                                      \
      }                                                                        \
      if (dst_tp != af_self->get_return_type()) {                              \
        std::stringstream ss;                                                  \
        ss << "Provided types " << ndt::make_funcproto(1, src_tp, dst_tp)      \
           << " do not match the arrfunc proto " << af_self->func_proto;       \
        throw type_error(ss.str());                                            \
      }                                                                        \
      self_type *e = ckb->alloc_ck_leaf<self_type>(ckb_offset);                \
      e->base.template set_expr_function<self_type>(kernreq);                  \
      e->func = *af_self->get_data_as<Functor>();                              \
                                                                               \
      return ckb_offset;                                                       \
    }                                                                          \
  };

DYND_PP_JOIN_MAP(DYND_CODE, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))
#undef DYND_CODE

  template<typename T>
  struct is_suitable_input {
    enum {
      value =
          !is_reference<T>::value || is_const<remove_reference<T>::type>::value
    };
  };

  template<typename Functor>
  struct functor_arrfunc_maker;

/**
 * Creates an arrfunc from a function pointer that returns a value.
 */
#define DYND_CODE(NSRC)                                                        \
  template <typename R, DYND_PP_ARG_RANGE_1(typename A, NSRC)>                 \
  struct functor_arrfunc_maker<R (*)(DYND_PP_ARG_RANGE_1(A, NSRC))> {          \
    typedef R (*func_type)(DYND_PP_ARG_RANGE_1(A, NSRC));                      \
    static void make(func_type func, arrfunc_type_data *out_af)                \
    {                                                                          \
      DYND_PP_STATIC_ASSERT_RANGE_1("all reference arguments must be const",   \
                                    detail::is_suitable_input, A, NSRC)        \
      /* Create D0, D1, ... as cleaned version of A0, A1, ... */               \
      DYND_PP_CLEAN_TYPE_RANGE_1(D, A, NSRC);                                  \
                                                                               \
      /* Create dst_tp and the src_tp array from R and D0, D1, ... */          \
      DYND_PP_NDT_TYPES_FROM_TYPES(R, D, NSRC);                                \
                                                                               \
      out_af->func_proto = ndt::make_funcproto(src_tp, dst_tp);                \
      *out_af->get_data_as<func_type>() = func;                                \
      out_af->instantiate = &detail::functor_ckernel_instantiator<             \
                                func_type, func_type>::instantiate;            \
      out_af->free_func = NULL;                                                \
    }                                                                          \
  };

DYND_PP_JOIN_MAP(DYND_CODE, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))
#undef DYND_CODE

/**
 * Creates an arrfunc from a function pointer that puts the return value
 * in the first parameter.
 */
#define DYND_CODE(NSRC)                                                        \
  template <typename R, DYND_PP_ARG_RANGE_1(typename A, NSRC)>                 \
  struct functor_arrfunc_maker<void (*)(R &, DYND_PP_ARG_RANGE_1(A, NSRC))> {  \
    typedef void (*func_type)(R &, DYND_PP_ARG_RANGE_1(A, NSRC));              \
    static void make(func_type func, arrfunc_type_data *out_af)                \
    {                                                                          \
      DYND_PP_STATIC_ASSERT_RANGE_1("all reference arguments must be const",   \
                                    detail::is_suitable_input, A, NSRC)        \
      /* Create D0, D1, ... as cleaned version of A0, A1, ... */               \
      DYND_PP_CLEAN_TYPE_RANGE_1(D, A, NSRC);                                  \
                                                                               \
      /* Create dst_tp and the src_tp array from R and D0, D1, ... */          \
      DYND_PP_NDT_TYPES_FROM_TYPES(R, D, NSRC);                                \
                                                                               \
      out_af->func_proto = ndt::make_funcproto(src_tp, dst_tp);                \
      *out_af->get_data_as<func_type>() = func;                                \
      out_af->instantiate = &detail::functor_ckernel_instantiator<             \
                                func_type, func_type>::instantiate;            \
      out_af->free_func = NULL;                                                \
    }                                                                          \
  };

DYND_PP_JOIN_MAP(DYND_CODE, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))
#undef DYND_CODE

} // namespace detail

template <typename Functor>
void make_functor_arrfunc(Functor func, arrfunc_type_data *out_af)
{
  detail::functor_arrfunc_maker<Functor>::make(func, out_af);
}

template <typename Functor>
nd::arrfunc make_functor_arrfunc(Functor func)
{
  nd::array af = nd::empty(ndt::make_arrfunc());
  make_functor_arrfunc(func, reinterpret_cast<arrfunc_type_data *>(
                                 af.get_readwrite_originptr()));
  af.flag_as_immutable();
  return af;
}

}} // namespace dynd::nd

#endif // DYND__FUNC_FUNCTOR_ARRFUNC_HPP
