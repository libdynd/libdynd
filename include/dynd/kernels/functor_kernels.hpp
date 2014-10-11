//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND__KERNELS_FUNCTOR_KERNELS_HPP
#define DYND__KERNELS_FUNCTOR_KERNELS_HPP

#include <dynd/strided_vals.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/pp/meta.hpp>
#include <dynd/types/funcproto_type.hpp>
#include <dynd/types/base_struct_type.hpp>

namespace dynd { namespace nd { namespace detail {

template <typename T>
class typed_param_from_bytes {
public:
    void init(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(arrmeta), const nd::array &DYND_UNUSED(kwds)) {
    }

    T &val(char *data) {
        return *reinterpret_cast<T *>(data);
    }

    const T &val(const char *data) {
        return *reinterpret_cast<const T *>(data);
    }
};

template <typename T, int N>
class typed_param_from_bytes<nd::strided_vals<T, N> > {
private:
    nd::strided_vals<T, N> m_strided;

public:
    void init(const ndt::type &DYND_UNUSED(tp), const char *arrmeta, const nd::array &kwds) {
        m_strided.set_data(NULL, reinterpret_cast<const size_stride_t *>(arrmeta),
            reinterpret_cast<start_stop_t *>(kwds.p("start_stop").as<intptr_t>()));

        ndt::type dt = kwds.get_dtype();
        try {
            const nd::array &mask = kwds.p("mask").f("dereference");
            m_strided.set_mask(mask.get_readonly_originptr(), reinterpret_cast<const size_stride_t *>(mask.get_arrmeta()));
        } catch (...) {
            m_strided.set_mask(NULL);
        }
    }

    nd::strided_vals<T, N> &val(char *data) {
        m_strided.set_data(data);
        return m_strided;
    }

    const nd::strided_vals<T, N> &val(const char *data) {
        m_strided.set_data(data);
        return m_strided;
    }
};

#define DECL_TYPED_PARAM_FROM_BYTES(TYPENAME, NAME) DYND_PP_META_DECL(typed_param_from_bytes<TYPENAME>, NAME)
#define INIT_TYPED_PARAM_FROM_BYTES(NAME, TP, ARRMETA) NAME.init(TP, ARRMETA, kwds)
#define PARTIAL_DECAY(TYPENAME) std::remove_cv<typename std::remove_reference<TYPENAME>::type>::type
#define PASS(NAME, ARG) NAME.val(ARG)

template <typename func_type, typename funcproto_type, bool buffered, bool thread_buffered>
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
        DYND_PP_JOIN_ELWISE_1(DECL_TYPED_PARAM_FROM_BYTES, (;), \
            DYND_PP_META_NAME_RANGE(D, N), DYND_PP_META_NAME_RANGE(from_src, N)); \
\
        functor_ck(const func_type &func) : func(func) { \
        } \
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
                                    kernel_request_t kernreq, const nd::array &kwds, \
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
            self_type *e = self_type::create(ckb, kernreq, ckb_offset, *af_self->get_data_as<func_type>()); \
            DYND_PP_JOIN_ELWISE_1(INIT_TYPED_PARAM_FROM_BYTES, (;), DYND_PP_META_NAME_RANGE(e->from_src, N), \
                DYND_PP_META_AT_RANGE(src_tp, N), DYND_PP_META_AT_RANGE(src_arrmeta, N)); \
\
            return ckb_offset; \
        } \
    }; \
\
    template <typename func_type, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N)), typename buffer_type> \
    struct functor_ck<func_type, R DYND_PP_APPEND(buffer_type *, DYND_PP_META_NAME_RANGE(A, N)), true, false> \
      : kernels::expr_ck<functor_ck<func_type, R DYND_PP_APPEND(buffer_type *, DYND_PP_META_NAME_RANGE(A, N)), true, false>, N> { \
        typedef functor_ck self_type; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_MAP_1(PARTIAL_DECAY, DYND_PP_META_NAME_RANGE(A, N)), DYND_PP_META_NAME_RANGE(D, N)); \
\
        func_type func; \
        DYND_PP_JOIN_ELWISE_1(DECL_TYPED_PARAM_FROM_BYTES, (;), \
            DYND_PP_META_NAME_RANGE(D, N), DYND_PP_META_NAME_RANGE(from_src, N)); \
        char buffer[sizeof(buffer_type)]; \
\
        functor_ck(const func_type &func) : func(func) { \
        } \
\
        inline void single(char *dst, const char *const *src) { \
            *reinterpret_cast<R *>(dst) = this->func(DYND_PP_JOIN_ELWISE_1(PASS, (,), \
                DYND_PP_META_NAME_RANGE(this->from_src, N), DYND_PP_META_AT_RANGE(src, N)), reinterpret_cast<buffer_type *>(this->buffer)); \
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
                    DYND_PP_META_NAME_RANGE(this->from_src, N), DYND_PP_META_NAME_RANGE(src, N)), reinterpret_cast<buffer_type *>(this->buffer)); \
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
                                    kernel_request_t kernreq, const nd::array &kwds, \
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
            self_type *e = self_type::create(ckb, kernreq, ckb_offset, *af_self->get_data_as<func_type>()); \
            DYND_PP_JOIN_ELWISE_1(INIT_TYPED_PARAM_FROM_BYTES, (;), DYND_PP_META_NAME_RANGE(e->from_src, N), \
                DYND_PP_META_AT_RANGE(src_tp, N), DYND_PP_META_AT_RANGE(src_arrmeta, N)); \
            new (e->buffer) buffer_type(kwds); \
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
    template <typename func_type, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    struct functor_ck<func_type, void DYND_PP_PREPEND(R &, DYND_PP_META_NAME_RANGE(A, N)), false, false> \
      : kernels::expr_ck<functor_ck<func_type, void DYND_PP_PREPEND(R &, DYND_PP_META_NAME_RANGE(A, N)), false, false>, N> { \
        typedef functor_ck self_type; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_MAP_1(PARTIAL_DECAY, DYND_PP_META_NAME_RANGE(A, N)), DYND_PP_META_NAME_RANGE(D, N)); \
\
        func_type func; \
        DECL_TYPED_PARAM_FROM_BYTES(R, from_dst); \
        DYND_PP_JOIN_ELWISE_1(DECL_TYPED_PARAM_FROM_BYTES, (;), \
            DYND_PP_META_NAME_RANGE(D, N), DYND_PP_META_NAME_RANGE(from_src, N)); \
\
        functor_ck(const func_type &func) : func(func) { \
        } \
\
        inline void single(char *dst, const char *const *src) { \
            this->func(PASS(this->from_dst, dst), DYND_PP_JOIN_ELWISE_1(PASS, (,), \
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
                this->func(PASS(this->from_dst, dst), DYND_PP_JOIN_ELWISE_1(PASS, (,), \
                    DYND_PP_META_NAME_RANGE(this->from_src, N), DYND_PP_META_NAME_RANGE(src, N))); \
                dst += dst_stride; \
                DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ADD_ASGN, (;), \
                    DYND_PP_META_NAME_RANGE(src, N), DYND_PP_META_NAME_RANGE(src_stride, N)); \
            } \
        } \
\
        static intptr_t instantiate(const arrfunc_type_data *af_self, \
                                    dynd::ckernel_builder *ckb, intptr_t ckb_offset, \
                                    const ndt::type &dst_tp, const char *dst_arrmeta, \
                                    const ndt::type *src_tp, const char *const *src_arrmeta, \
                                    kernel_request_t kernreq, const nd::array &kwds, \
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
            self_type *e = self_type::create(ckb, kernreq, ckb_offset, *af_self->get_data_as<func_type>()); \
            INIT_TYPED_PARAM_FROM_BYTES(e->from_dst, dst_tp, dst_arrmeta); \
            DYND_PP_JOIN_ELWISE_1(INIT_TYPED_PARAM_FROM_BYTES, (;), DYND_PP_META_NAME_RANGE(e->from_src, N), \
                DYND_PP_META_AT_RANGE(src_tp, N), DYND_PP_META_AT_RANGE(src_arrmeta, N)); \
\
            return ckb_offset; \
        } \
    }; \
\
    template <typename func_type, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N)), typename buffer_type> \
    struct functor_ck<func_type, void DYND_PP_APPEND(buffer_type *, DYND_PP_PREPEND(R &, DYND_PP_META_NAME_RANGE(A, N))), true, false> \
      : kernels::expr_ck<functor_ck<func_type, void DYND_PP_APPEND(buffer_type *, DYND_PP_PREPEND(R &, DYND_PP_META_NAME_RANGE(A, N))), true, false>, N> { \
        typedef functor_ck self_type; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_MAP_1(PARTIAL_DECAY, DYND_PP_META_NAME_RANGE(A, N)), DYND_PP_META_NAME_RANGE(D, N)); \
\
        func_type func; \
        DECL_TYPED_PARAM_FROM_BYTES(R, from_dst); \
        DYND_PP_JOIN_ELWISE_1(DECL_TYPED_PARAM_FROM_BYTES, (;), \
            DYND_PP_META_NAME_RANGE(D, N), DYND_PP_META_NAME_RANGE(from_src, N)); \
        char buffer[sizeof(buffer_type)]; \
\
        functor_ck(const func_type &func) : func(func) { \
        } \
\
        inline void single(char *dst, const char *const *src) { \
            this->func(PASS(this->from_dst, dst), DYND_PP_JOIN_ELWISE_1(PASS, (,), \
                DYND_PP_META_NAME_RANGE(this->from_src, N), DYND_PP_META_AT_RANGE(src, N)), reinterpret_cast<buffer_type *>(this->buffer)); \
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
                this->func(PASS(this->from_dst, dst), DYND_PP_JOIN_ELWISE_1(PASS, (,), \
                    DYND_PP_META_NAME_RANGE(this->from_src, N), DYND_PP_META_NAME_RANGE(src, N)), reinterpret_cast<buffer_type *>(this->buffer)); \
                dst += dst_stride; \
                DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ADD_ASGN, (;), \
                    DYND_PP_META_NAME_RANGE(src, N), DYND_PP_META_NAME_RANGE(src_stride, N)); \
            } \
        } \
\
        static intptr_t instantiate(const arrfunc_type_data *af_self, \
                                    dynd::ckernel_builder *ckb, intptr_t ckb_offset, \
                                    const ndt::type &dst_tp, const char *dst_arrmeta, \
                                    const ndt::type *src_tp, const char *const *src_arrmeta, \
                                    kernel_request_t kernreq, const nd::array &kwds, \
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
            self_type *e = self_type::create(ckb, kernreq, ckb_offset, *af_self->get_data_as<func_type>()); \
            INIT_TYPED_PARAM_FROM_BYTES(e->from_dst, dst_tp, dst_arrmeta); \
            DYND_PP_JOIN_ELWISE_1(INIT_TYPED_PARAM_FROM_BYTES, (;), DYND_PP_META_NAME_RANGE(e->from_src, N), \
                DYND_PP_META_AT_RANGE(src_tp, N), DYND_PP_META_AT_RANGE(src_arrmeta, N)); \
            new (e->buffer) buffer_type(kwds); \
\
            return ckb_offset; \
        } \
    };

DYND_PP_JOIN_MAP(FUNCTOR_CK, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_SRC_MAX)))

#undef FUNCTOR_CK

#undef DECL_TYPED_PARAM_FROM_BYTES
#undef INIT_TYPED_PARAM_FROM_BYTES
#undef PARTIAL_DECAY
#undef PASS

} // namespace dynd::nd::detail

template <typename func_type, typename funcproto_type>
struct functor_ck : detail::functor_ck<func_type, funcproto_type,
    aux::is_buffered<funcproto_type>::value, aux::is_thread_buffered<funcproto_type>::value> {
};

}} // namespace dynd::nd

#endif
