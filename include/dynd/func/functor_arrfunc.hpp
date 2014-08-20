//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND__FUNC_FUNCTOR_ARRFUNC_HPP
#define DYND__FUNC_FUNCTOR_ARRFUNC_HPP

#include <dynd/types/funcproto_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/pp/arrfunc_util.hpp>

namespace dynd {

namespace nd { namespace detail {
  /**
   * Metaprogram to tell whether an argument is const or by value.
   */
  template<typename T>
  struct is_suitable_input {
    // TODO: Reenable - error was triggering when not expected
    enum { value = true }; //is_const<T>::value || !is_reference<T>::value };
  };
}} // namespace nd::detail

/**
 * Metaprogram to detect whether a type is a function pointer.
 */
template <typename T>
struct is_function_pointer {
  enum { value = false };
};

#define DYND_CODE(N)                                                           \
  template <typename R, DYND_PP_TYPENAME_ARGRANGE_1(A, N)>                     \
  struct is_function_pointer<R (*)(DYND_PP_ARGRANGE_1(A, N))> {                \
    enum { value = true };                                                     \
  };
DYND_PP_JOIN_MAP(DYND_CODE, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))
#undef DYND_CODE

} // namespace dynd

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
            DYND_PP_TYPENAME_ARGRANGE_1(A, NSRC)>                              \
  struct functor_ckernel_instantiator<Functor,                                 \
                                      R (*)(DYND_PP_ARGRANGE_1(A, NSRC))> {    \
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
            func(DYND_PP_DEREF_CAST_ARGRANGE_1(D, src, NSRC));                 \
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
            DYND_PP_TYPENAME_ARGRANGE_1(A, NSRC)>                              \
  struct functor_ckernel_instantiator<                                         \
      Functor, void (*)(R &, DYND_PP_ARGRANGE_1(A, NSRC))> {                   \
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
             DYND_PP_DEREF_CAST_ARGRANGE_1(D, src, NSRC));                     \
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

  template<bool CallOp, typename Functor>
  struct functor_arrfunc_maker;

/**
 * Creates an arrfunc from a function pointer that returns a value.
 */
#define DYND_CODE(NSRC)                                                        \
  template <typename R, DYND_PP_TYPENAME_ARGRANGE_1(A, NSRC)>                  \
  struct functor_arrfunc_maker<true, R (*)(DYND_PP_ARGRANGE_1(A, NSRC))> {    \
    typedef R (*func_type)(DYND_PP_ARGRANGE_1(A, NSRC));                       \
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
  template <typename R, DYND_PP_TYPENAME_ARGRANGE_1(A, NSRC)>                  \
  struct functor_arrfunc_maker<true,                                           \
                               void (*)(R &, DYND_PP_ARGRANGE_1(A, NSRC))> {   \
    typedef void (*func_type)(R &, DYND_PP_ARGRANGE_1(A, NSRC));               \
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
 * Gets triggered when Functor is not a function pointer type, in
 * which case we expect it to be a class with operator() defined.
 */
template <typename Functor>
struct functor_arrfunc_maker<false, Functor> {
/**
 * The make_tagged functions include a pointer-to-member function as an
 * argument, which lets us pull out the type information from the type.
 */
#define DYND_CODE(NSRC)                                                        \
  template <typename R, DYND_PP_TYPENAME_ARGRANGE_1(A, NSRC)>                  \
  inline static void make_tagged(Functor func, arrfunc_type_data *out_af,      \
                                 R (Functor::*)(DYND_PP_ARGRANGE_1(A, NSRC))   \
                                 const)                                        \
  {                                                                            \
    DYND_PP_STATIC_ASSERT_RANGE_1("all reference arguments must be const",     \
                                  detail::is_suitable_input, A, NSRC)          \
    /* Create D0, D1, ... as cleaned version of A0, A1, ... */                 \
    DYND_PP_CLEAN_TYPE_RANGE_1(D, A, NSRC);                                    \
                                                                               \
    /* Create dst_tp and the src_tp array from R and D0, D1, ... */            \
    DYND_PP_NDT_TYPES_FROM_TYPES(R, D, NSRC);                                  \
                                                                               \
    out_af->func_proto = ndt::make_funcproto(src_tp, dst_tp);                  \
    *out_af->get_data_as<Functor>() = func;                                    \
    /* Make func ptr type to reuse functor_ckernel_instantiator */             \
    typedef R (*func_type)(DYND_PP_ARGRANGE_1(A, NSRC));                       \
    out_af->instantiate = &detail::functor_ckernel_instantiator<               \
                              Functor, func_type>::instantiate;                \
    out_af->free_func = NULL;                                                  \
  }
DYND_PP_JOIN_MAP(DYND_CODE, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))
#undef DYND_CODE

/**
 * A second round of make_tagged functions for R& return argument.
 */
#define DYND_CODE(NSRC)                                                        \
  template <typename R, DYND_PP_TYPENAME_ARGRANGE_1(A, NSRC)>                  \
  inline static void make_tagged(                                              \
      Functor func, arrfunc_type_data *out_af,                                 \
      void (Functor::*)(R &, DYND_PP_ARGRANGE_1(A, NSRC)) const)               \
  {                                                                            \
    DYND_PP_STATIC_ASSERT_RANGE_1("all reference arguments must be const",     \
                                  detail::is_suitable_input, A, NSRC)          \
    /* Create D0, D1, ... as cleaned version of A0, A1, ... */                 \
    DYND_PP_CLEAN_TYPE_RANGE_1(D, A, NSRC);                                    \
                                                                               \
    /* Create dst_tp and the src_tp array from R and D0, D1, ... */            \
    DYND_PP_NDT_TYPES_FROM_TYPES(R, D, NSRC);                                  \
                                                                               \
    out_af->func_proto = ndt::make_funcproto(src_tp, dst_tp);                  \
    *out_af->get_data_as<Functor>() = func;                                    \
    /* Make func ptr type to reuse functor_ckernel_instantiator */             \
    typedef void (*func_type)(R &, DYND_PP_ARGRANGE_1(A, NSRC));               \
    out_af->instantiate = &detail::functor_ckernel_instantiator<               \
                              Functor, func_type>::instantiate;                \
    out_af->free_func = NULL;                                                  \
  }
DYND_PP_JOIN_MAP(DYND_CODE, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))
#undef DYND_CODE

  static void make(Functor func, arrfunc_type_data *out_af)
  {
    make_tagged(func, out_af, &Functor::operator());
  }
};

} // namespace detail

template <typename Functor>
void make_functor_arrfunc(Functor func, arrfunc_type_data *out_af)
{
  detail::functor_arrfunc_maker<(bool)is_function_pointer<Functor>::value,
                                Functor>::make(func, out_af);
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
