//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND__KERNELS_FUNCTOR_KERNELS_HPP
#define DYND__KERNELS_FUNCTOR_KERNELS_HPP

#include <dynd/buffer.hpp>
#include <dynd/types/funcproto_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/func/make_callable.hpp>
#include <dynd/pp/arrfunc_util.hpp>
#include <iostream>

namespace dynd { namespace nd {

namespace detail {

  /**
   * Helper struct that casts or coerces data right before passing it to the functor.
   */
  template <typename T>
  struct val_helper;

  template <typename T>
  struct val_helper {
    void init(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(arrmeta)) {
    }

    T *make(char *data) {
        return reinterpret_cast<T *>(data);
    }

    const T *make(const char *data) {
        return reinterpret_cast<const T *>(data);
    }
  };

  template <typename T, int N>
  struct val_helper<nd::strided_vals<T, N> > {
    nd::strided_vals<T, N> vals;

    void init(const ndt::type &DYND_UNUSED(tp), const char *arrmeta) {
        vals.init(reinterpret_cast<const size_stride_t *>(arrmeta));
    }

    nd::strided_vals<T, N> *make(char *data) {
        vals.set_readonly_originptr(data);
        return &vals;
    }

    const nd::strided_vals<T, N> *make(const char *data) {
        vals.set_readonly_originptr(data);
        return &vals;
    }
  };

#define PARTIAL_DECAY(TYPENAME) remove_const<typename remove_reference<TYPENAME>::type>::type
#define PASS(TYPENAME, NAME) DYND_PP_META_DEREFERENCE(DYND_PP_META_REINTERPRET_CAST(DYND_PP_META_MAKE_CONST_PTR(TYPENAME), NAME))

template <typename func_type, typename funcproto_type, bool aux_buffered, bool thread_aux_buffered>
struct functor_ckernel;

/**
 * This generates code to instantiate a ckernel calling a C++ function
 * object which returns a value.
 */
#define FUNCTOR_CKERNEL(N) \
  template <typename func_type, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
  struct functor_ckernel<func_type, R DYND_PP_META_NAME_RANGE(A, N), false, false> { \
    typedef functor_ckernel self_type; \
    DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
      DYND_PP_MAP_1(PARTIAL_DECAY, DYND_PP_META_NAME_RANGE(A, N)), DYND_PP_META_NAME_RANGE(D, N)); \
\
    ckernel_prefix base; \
    func_type func; \
\
    static void single(char *dst, const char *const *src, ckernel_prefix *ckp) { \
      self_type *e = reinterpret_cast<self_type *>(ckp); \
      *reinterpret_cast<R *>(dst) = \
          e->func(DYND_PP_JOIN_ELWISE_1(PASS, (,), DYND_PP_META_NAME_RANGE(D, N), DYND_PP_META_AT_RANGE(src, N))); \
    } \
\
    static void strided(char *dst, intptr_t dst_stride, \
                        const char *const *src, const intptr_t *src_stride, \
                        size_t count, ckernel_prefix *ckp) { \
      self_type *e = reinterpret_cast<self_type *>(ckp); \
      func_type func = e->func; \
      DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
        DYND_PP_REPEAT_1(const char *, N), DYND_PP_META_NAME_RANGE(src, N), DYND_PP_META_AT_RANGE(src, N)); \
      DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
        DYND_PP_REPEAT_1(intptr_t, N), DYND_PP_META_NAME_RANGE(src_stride, N), DYND_PP_META_AT_RANGE(src_stride, N)); \
      for (size_t i = 0; i < count; ++i) { \
        *reinterpret_cast<R *>(dst) = \
            func(DYND_PP_JOIN_ELWISE_1(PASS, (,), DYND_PP_META_NAME_RANGE(D, N), DYND_PP_META_NAME_RANGE(src, N))); \
\
        dst += dst_stride; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ADD_ASGN, (;), \
          DYND_PP_META_NAME_RANGE(src, N), DYND_PP_META_NAME_RANGE(src_stride, N)); \
      } \
    } \
\
    static intptr_t instantiate(const arrfunc_type_data *af_self, \
                                dynd::ckernel_builder *ckb, intptr_t ckb_offset, \
                                const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), \
                                const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta), \
                                kernel_request_t kernreq, aux_buffer *DYND_UNUSED(aux), \
                                const eval::eval_context *DYND_UNUSED(ectx)) { \
      for (intptr_t i = 0; i < N; ++i) { \
        if (src_tp[i] != af_self->get_param_type(i)) { \
          std::stringstream ss; \
          ss << "Provided types " << ndt::make_funcproto(N, src_tp, dst_tp) \
             << " do not match the arrfunc proto " << af_self->func_proto; \
          throw type_error(ss.str()); \
        } \
      } \
      if (dst_tp != af_self->get_return_type()) { \
        std::stringstream ss; \
        ss << "Provided types " << ndt::make_funcproto(N, src_tp, dst_tp) \
           << " do not match the arrfunc proto " << af_self->func_proto; \
        throw type_error(ss.str()); \
      } \
      self_type *e = ckb->alloc_ck_leaf<self_type>(ckb_offset); \
      e->base.template set_expr_function<self_type>(kernreq); \
      e->func = *af_self->get_data_as<func_type>(); \
\
      return ckb_offset; \
    } \
  };

DYND_PP_JOIN_MAP(FUNCTOR_CKERNEL, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_SRC_MAX)))

#undef FUNCTOR_CKERNEL

/**
 * This generates code to instantiate a ckernel calling a C++ function
 * object which returns a value in the first parameter as an output reference.
 */
#define FUNCTOR_CKERNEL(COUNT) \
  template <typename Functor, \
    typename R, DYND_PP_TYPENAME_ARGRANGE_1(A, COUNT)> \
  struct functor_ckernel<Functor, \
    void (R &, DYND_PP_ARGRANGE_1(A, COUNT)), false, false> \
  { \
    typedef functor_ckernel self_type; \
\
    DYND_PP_CLEAN_TYPE_RANGE_1(D, A, COUNT); \
\
    ckernel_prefix base; \
    Functor func; \
    val_helper<R> dst_helper; \
    DYND_PP_DECL_HELPERS(D, src, COUNT); \
\
    static void single(char *dst, const char *const *src, ckernel_prefix *ckp) \
    { \
      self_type *e = reinterpret_cast<self_type *>(ckp); \
      e->func(*e->dst_helper.make(dst), \
        DYND_PP_DEREF_MAKE_ARRAY_RANGE_1(D, src, COUNT)); \
    } \
\
    static void strided(char *dst, intptr_t dst_stride, \
                        const char *const *src, const intptr_t *src_stride, \
                        size_t count, ckernel_prefix *ckp) \
    { \
      self_type *e = reinterpret_cast<self_type *>(ckp); \
      Functor func = e->func; \
      /* const char *src# = src[#]; */ \
      /* intptr_t src_stride# = src_stride[#]; */ \
      DYND_PP_INIT_SRC_VARIABLES(COUNT); \
      for (size_t i = 0; i < count; ++i) \
      { \
        /*  func(*(R *)dst, *(const D0 *)src0, ...); */ \
        func(*e->dst_helper.make(dst), \
             DYND_PP_DEREF_MAKE_ARG_RANGE_1(D, src, COUNT)); \
\
        /* Increment ``dst``, ``src#`` by their respective strides */ \
        DYND_PP_STRIDED_INCREMENT(COUNT); \
      } \
    } \
\
    static intptr_t instantiate(const arrfunc_type_data *af_self, \
      dynd::ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp, \
      const char *dst_arrmeta, const ndt::type *src_tp,const char *const *src_arrmeta, \
      kernel_request_t kernreq, aux_buffer *DYND_UNUSED(aux), const eval::eval_context *DYND_UNUSED(ectx)) \
    { \
      for (intptr_t i = 0; i < COUNT; ++i) \
      { \
        if (src_tp[i] != af_self->get_param_type(i)) \
        { \
          std::stringstream ss; \
          ss << "Provided types " << ndt::make_funcproto(COUNT, src_tp, dst_tp) \
             << " do not match the arrfunc proto " << af_self->func_proto; \
          throw type_error(ss.str()); \
        } \
      } \
      if (dst_tp != af_self->get_return_type()) \
      { \
        std::stringstream ss; \
        ss << "Provided types " << ndt::make_funcproto(COUNT, src_tp, dst_tp) \
           << " do not match the arrfunc proto " << af_self->func_proto; \
        throw type_error(ss.str()); \
      } \
      self_type *e = ckb->alloc_ck_leaf<self_type>(ckb_offset); \
      e->base.template set_expr_function<self_type>(kernreq); \
      e->func = *af_self->get_data_as<Functor>(); \
      e->dst_helper.init(dst_tp, dst_arrmeta); \
      DYND_PP_INIT_SRC_HELPERS(COUNT) \
\
      return ckb_offset; \
    } \
  };

DYND_PP_JOIN_MAP(FUNCTOR_CKERNEL, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))
#undef FUNCTOR_CKERNEL

#undef PASS
#undef PARTIAL_DECAY

} // namespace dynd::nd::detail

template <typename func_type, typename funcproto_type = typename func_like<func_type>::type>
struct functor_ckernel : detail::functor_ckernel<func_type, funcproto_type,
  is_aux_buffered<funcproto_type>::value, is_thread_aux_buffered<funcproto_type>::value> {
};

}} // namespace dynd::nd

#endif
