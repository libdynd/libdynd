//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND__KERNELS_FUNCTOR_KERNELS_HPP
#define DYND__KERNELS_FUNCTOR_KERNELS_HPP

#include <dynd/buffer.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/pp/meta.hpp>
#include <dynd/types/funcproto_type.hpp>

#include <dynd/pp/arrfunc_util.hpp> // Delete this
#include <dynd/types/cfixed_dim_type.hpp> // Delete this

namespace dynd { namespace nd { namespace detail {

/**
 * Helper struct that casts or coerces data right before passing it to the functor.
 */
template <typename T>
class from_bytes {
public:
  void init(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(arrmeta)) {
  }

  T &val(char *data) {
      return *reinterpret_cast<T *>(data);
  }

  const T &val(const char *data) {
      return *reinterpret_cast<const T *>(data);
  }
};

template <typename T, int N>
class from_bytes<nd::strided_vals<T, N> > {
private:
  nd::strided_vals<T, N> m_val;

public:
  void init(const ndt::type &DYND_UNUSED(tp), const char *arrmeta) {
      m_val.init(reinterpret_cast<const size_stride_t *>(arrmeta));
  }

  nd::strided_vals<T, N> &val(char *data) {
      m_val.set_readonly_originptr(data);
      return m_val;
  }

  const nd::strided_vals<T, N> &val(const char *data) {
      m_val.set_readonly_originptr(data);
      return m_val;
  }
};



#define DECL_FROM_BYTES(TYPENAME, NAME) DYND_PP_META_DECL(from_bytes<TYPENAME>, NAME)
#define INIT_FROM_BYTES(NAME, TP, ARRMETA) NAME.init(TP, ARRMETA)
#define PARTIAL_DECAY(TYPENAME) std::remove_const<typename std::remove_reference<TYPENAME>::type>::type
#define PASS(NAME, ARG) NAME.val(ARG)

#define OLDPASS(NAME0, NAME1) (e->NAME0).val(NAME1)



template <typename T, bool func_pointer>
struct test;

template <typename func_type>
struct test<func_type, true> {
  typedef func_type type;

  static func_type get(func_type func) {
    return func;
  }
};

template <typename func_type>
struct test<func_type, false> {
  typedef func_type *type;

  static func_type get(func_type *func) {
    return *func;
  }
};


template <typename func_type, typename funcproto_type, bool aux_buffered, bool thread_aux_buffered>
struct functor_ck;

/**
 * This generates code to instantiate a ckernel calling a C++ function
 * object which returns a value.
 */
#define FUNCTOR_CK(N) \
    template <typename func_type, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    struct functor_ck<func_type, R DYND_PP_META_NAME_RANGE(A, N), false, false> \
      : kernels::expr_ck<functor_ck<func_type, R DYND_PP_META_NAME_RANGE(A, N), false, false>, N> { \
        typedef functor_ck self_type; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_MAP_1(PARTIAL_DECAY, DYND_PP_META_NAME_RANGE(A, N)), DYND_PP_META_NAME_RANGE(D, N)); \
\
        func_type func; \
        DYND_PP_JOIN_ELWISE_1(DECL_FROM_BYTES, (;), \
            DYND_PP_META_NAME_RANGE(D, N), DYND_PP_META_NAME_RANGE(from_src, N)); \
\
        inline void single(char *dst, const char *const *src) { \
            *reinterpret_cast<R *>(dst) = this->func(DYND_PP_JOIN_ELWISE_1(PASS, (,), \
                DYND_PP_META_NAME_RANGE(this->from_src, N), DYND_PP_META_AT_RANGE(src, N))); \
        } \
\
        inline void strided(char *dst, intptr_t dst_stride, \
                            const char *const *src, const intptr_t *src_stride, \
                            size_t count) { \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(const char *, N), DYND_PP_META_NAME_RANGE(src, N), DYND_PP_META_AT_RANGE(src, N)); \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(intptr_t, N), DYND_PP_META_NAME_RANGE(src_stride, N), DYND_PP_META_AT_RANGE(src_stride, N)); \
            for (size_t i = 0; i < count; ++i) { \
                *reinterpret_cast<R *>(dst) = this->func(DYND_PP_JOIN_ELWISE_1(PASS, (,), \
                    DYND_PP_META_NAME_RANGE(this->from_src, N), DYND_PP_META_NAME_RANGE(src, N))); \
                dst += dst_stride; \
                DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ADD_ASGN, (;), \
                    DYND_PP_META_NAME_RANGE(src, N), DYND_PP_META_NAME_RANGE(src_stride, N)); \
            } \
        } \
\
        static intptr_t instantiate(const arrfunc_type_data *af_self, \
                                    dynd::ckernel_builder *ckb, intptr_t ckb_offset, \
                                    const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), \
                                    const ndt::type *src_tp, const char *const *src_arrmeta, \
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
\
            self_type *e = self_type::create(ckb, kernreq, ckb_offset); \
            e->func = *af_self->get_data_as<func_type>(); \
            DYND_PP_JOIN_ELWISE_1(INIT_FROM_BYTES, (;), DYND_PP_META_NAME_RANGE(e->from_src, N), \
                DYND_PP_META_AT_RANGE(src_tp, N), DYND_PP_META_AT_RANGE(src_arrmeta, N)); \
\
            return ckb_offset; \
        } \
  }; \
\
  template <typename func_type, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N)), typename aux_buffer_type> \
  struct functor_ck<func_type, R DYND_PP_APPEND(aux_buffer_type *, DYND_PP_META_NAME_RANGE(A, N)), true, false> { \
    typedef functor_ck self_type; \
    DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
      DYND_PP_MAP_1(PARTIAL_DECAY, DYND_PP_META_NAME_RANGE(A, N)), DYND_PP_META_NAME_RANGE(D, N)); \
\
    ckernel_prefix base; \
    func_type func; \
    DYND_PP_JOIN_ELWISE_1(DECL_FROM_BYTES, (;), \
      DYND_PP_META_NAME_RANGE(D, N), DYND_PP_META_NAME_RANGE(from_src, N)); \
    aux_buffer_type *aux; \
\
    static void single(char *dst, const char *const *src, ckernel_prefix *ckp) { \
      self_type *e = reinterpret_cast<self_type *>(ckp); \
      *reinterpret_cast<R *>(dst) = \
          e->func(DYND_PP_JOIN_ELWISE_1(OLDPASS, (,), \
            DYND_PP_META_NAME_RANGE(from_src, N), DYND_PP_META_AT_RANGE(src, N)), e->aux); \
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
            *reinterpret_cast<R *>(dst) = func(DYND_PP_JOIN_ELWISE_1(OLDPASS, (,), \
                DYND_PP_META_NAME_RANGE(from_src, N), DYND_PP_META_NAME_RANGE(src, N)), e->aux); \
        dst += dst_stride; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ADD_ASGN, (;), \
          DYND_PP_META_NAME_RANGE(src, N), DYND_PP_META_NAME_RANGE(src_stride, N)); \
      } \
    } \
\
    static intptr_t instantiate(const arrfunc_type_data *af_self, \
                                dynd::ckernel_builder *ckb, intptr_t ckb_offset, \
                                const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), \
                                const ndt::type *src_tp, const char *const *src_arrmeta, \
                                kernel_request_t kernreq, aux_buffer *aux, \
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
      DYND_PP_JOIN_ELWISE_1(INIT_FROM_BYTES, (;), DYND_PP_META_NAME_RANGE(e->from_src, N), \
        DYND_PP_META_AT_RANGE(src_tp, N), DYND_PP_META_AT_RANGE(src_arrmeta, N)); \
      e->aux = reinterpret_cast<aux_buffer_type *>(aux); \
\
      return ckb_offset; \
    } \
  };

DYND_PP_JOIN_MAP(FUNCTOR_CK, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_SRC_MAX)))

#undef FUNCTOR_CK

/**
 * This generates code to instantiate a ckernel calling a C++ function
 * object which returns a value in the first parameter as an output reference.
 */
#define FUNCTOR_CK(N) \
  template <typename Functor, \
    typename R, DYND_PP_TYPENAME_ARGRANGE_1(A, N)> \
  struct functor_ck<Functor, \
    void (R &, DYND_PP_ARGRANGE_1(A, N)), false, false> \
  { \
    typedef functor_ck self_type; \
\
    DYND_PP_CLEAN_TYPE_RANGE_1(D, A, N); \
\
    ckernel_prefix base; \
    Functor func; \
    from_bytes<R> dst_helper; \
    DYND_PP_JOIN_ELWISE_1(DECL_FROM_BYTES, (;), \
      DYND_PP_META_NAME_RANGE(D, N), DYND_PP_META_NAME_RANGE(from_src, N)); \
\
    static void single(char *dst, const char *const *src, ckernel_prefix *ckp) \
    { \
      self_type *e = reinterpret_cast<self_type *>(ckp); \
      e->func(e->dst_helper.val(dst), \
        DYND_PP_JOIN_ELWISE_1(OLDPASS, (,), \
              DYND_PP_META_NAME_RANGE(from_src, N), DYND_PP_META_AT_RANGE(src, N))); \
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
      DYND_PP_INIT_SRC_VARIABLES(N); \
      for (size_t i = 0; i < count; ++i) \
      { \
        /*  func(*(R *)dst, *(const D0 *)src0, ...); */ \
        func(e->dst_helper.val(dst), \
             DYND_PP_JOIN_ELWISE_1(OLDPASS, (,), \
              DYND_PP_META_NAME_RANGE(from_src, N), DYND_PP_META_NAME_RANGE(src, N))); \
\
        /* Increment ``dst``, ``src#`` by their respective strides */ \
        DYND_PP_STRIDED_INCREMENT(N); \
      } \
    } \
\
    static intptr_t instantiate(const arrfunc_type_data *af_self, \
      dynd::ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp, \
      const char *dst_arrmeta, const ndt::type *src_tp,const char *const *src_arrmeta, \
      kernel_request_t kernreq, aux_buffer *DYND_UNUSED(aux), const eval::eval_context *DYND_UNUSED(ectx)) \
    { \
      for (intptr_t i = 0; i < N; ++i) \
      { \
        if (src_tp[i] != af_self->get_param_type(i)) \
        { \
          std::stringstream ss; \
          ss << "Provided types " << ndt::make_funcproto(N, src_tp, dst_tp) \
             << " do not match the arrfunc proto " << af_self->func_proto; \
          throw type_error(ss.str()); \
        } \
      } \
      if (dst_tp != af_self->get_return_type()) \
      { \
        std::stringstream ss; \
        ss << "Provided types " << ndt::make_funcproto(N, src_tp, dst_tp) \
           << " do not match the arrfunc proto " << af_self->func_proto; \
        throw type_error(ss.str()); \
      } \
      self_type *e = ckb->alloc_ck_leaf<self_type>(ckb_offset); \
      e->base.template set_expr_function<self_type>(kernreq); \
      e->func = *af_self->get_data_as<Functor>(); \
      e->dst_helper.init(dst_tp, dst_arrmeta); \
      DYND_PP_JOIN_ELWISE_1(INIT_FROM_BYTES, (;), DYND_PP_META_NAME_RANGE(e->from_src, N), \
        DYND_PP_META_AT_RANGE(src_tp, N), DYND_PP_META_AT_RANGE(src_arrmeta, N)); \
\
      return ckb_offset; \
    } \
  };

DYND_PP_JOIN_MAP(FUNCTOR_CK, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))
#undef FUNCTOR_CK

#undef PASS
#undef PARTIAL_DECAY

} // namespace dynd::nd::detail

template <typename func_type, typename funcproto_type>
struct functor_ck : detail::functor_ck<func_type, funcproto_type,
    is_aux_buffered<funcproto_type>::value, is_thread_aux_buffered<funcproto_type>::value> {
};

}} // namespace dynd::nd

#endif
