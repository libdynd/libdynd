//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FUNCPROTO_TYPE_HPP_
#define _DYND__FUNCPROTO_TYPE_HPP_

#include <vector>
#include <string>

#include <dynd/array.hpp>
#include <dynd/buffer.hpp>
#include <dynd/funcproto.hpp>
#include <dynd/pp/meta.hpp>
#include <dynd/types/strided_dim_type.hpp>

namespace dynd {

class funcproto_type : public base_type {
    intptr_t m_param_count;
    // This is always a contiguous immutable "strided * type" array
    nd::array m_param_types;
    ndt::type m_return_type;

public:
    funcproto_type(const nd::array &param_types, const ndt::type &return_type);

    virtual ~funcproto_type() {}

    inline intptr_t get_param_count() const {
        return m_param_count;
    }

    inline const nd::array& get_param_types() const {
        return m_param_types;
    }

    inline const ndt::type *get_param_types_raw() const {
        return reinterpret_cast<const ndt::type *>(
            m_param_types.get_readonly_originptr());
    }
    inline const ndt::type &get_param_type(intptr_t i) const {
        return get_param_types_raw()[i];
    }

    inline const ndt::type& get_return_type() const {
        return m_return_type;
    }

    void print_data(std::ostream& o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream& o) const;

    void transform_child_types(type_transform_fn_t transform_fn,
                               intptr_t arrmeta_offset, void *extra,
                               ndt::type &out_transformed_tp,
                               bool &out_was_transformed) const;
    ndt::type get_canonical_type() const;

    ndt::type apply_linear_index(intptr_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_tp, bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta,
                    const ndt::type& result_tp, char *out_arrmeta,
                    memory_block_data *embedded_reference,
                    size_t current_i, const ndt::type& root_tp,
                    bool leading_dimension, char **inout_data,
                    memory_block_data **inout_dataref) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta, memory_block_data *embedded_reference) const;
    void arrmeta_destruct(char *arrmeta) const;

    void get_dynamic_type_properties(
        const std::pair<std::string, gfunc::callable> **out_properties,
        size_t *out_count) const;
}; // class typevar_type

namespace ndt {
    /** Makes a funcproto type with the specified types */
    inline ndt::type make_funcproto(const nd::array &param_types,
                                    const ndt::type &return_type)
    {
        return ndt::type(
            new funcproto_type(param_types, return_type), false);
    }

    /** Makes a funcproto type with the specified types */
    inline ndt::type make_funcproto(intptr_t param_count,
                                    const ndt::type *param_types,
                                    const ndt::type &return_type)
    {
        nd::array tmp = nd::empty(param_count, ndt::make_type());
        ndt::type *tmp_vals =
            reinterpret_cast<ndt::type *>(tmp.get_readwrite_originptr());
        for (intptr_t i = 0; i != param_count; ++i) {
            tmp_vals[i] = param_types[i];
        }
        tmp.flag_as_immutable();
        return ndt::type(
            new funcproto_type(tmp, return_type), false);
    }

    /** Makes a unary funcproto type with the specified types */
    inline ndt::type make_funcproto(const ndt::type& single_param_type,
                                    const ndt::type &return_type)
    {
        ndt::type param_types[1] = {single_param_type};
        return ndt::type(
            new funcproto_type(param_types, return_type), false);
    }

namespace detail {

#define PARTIAL_DECAY(TYPENAME) std::remove_cv<typename std::remove_reference<TYPENAME>::type>::type
#define MAKE_TYPE(TYPENAME) make_type<TYPENAME>()

template<typename T, bool buffered, bool thread_buffered>
struct funcproto_type_from;

template<typename R>
struct funcproto_type_from<R (), false, false> {
    static inline ndt::type make() {
        nd::array param_types = nd::empty(0, ndt::make_type());
        param_types.flag_as_immutable();
        return make_funcproto(param_types, make_type<R>());
    }
};

#define FUNCPROTO_TYPE_FROM(N) \
    template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    struct funcproto_type_from<R DYND_PP_META_NAME_RANGE(A, N), false, false> { \
    private: \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_MAP_1(PARTIAL_DECAY, DYND_PP_META_NAME_RANGE(A, N)), DYND_PP_META_NAME_RANGE(D, N)); \
    public: \
        static inline ndt::type make() { \
            ndt::type param_types[N] = {DYND_PP_JOIN_MAP_1(MAKE_TYPE, (,), DYND_PP_META_NAME_RANGE(D, N))}; \
            return make_funcproto(param_types, make_type<R>()); \
        } \
    };

DYND_PP_JOIN_MAP(FUNCPROTO_TYPE_FROM, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef FUNCPROTO_TYPE_FROM

template<typename R>
struct funcproto_type_from<void (R &), false, false> {
    static inline ndt::type make() {
        nd::array param_types = nd::empty(0, ndt::make_type());
        param_types.flag_as_immutable();
        return make_funcproto(param_types, make_type<R>());
    }
};

#define FUNCPROTO_TYPE_FROM(N) \
    template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    struct funcproto_type_from<void DYND_PP_PREPEND(R &, DYND_PP_META_NAME_RANGE(A, N)), false, false> { \
    private: \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_MAP_1(PARTIAL_DECAY, DYND_PP_META_NAME_RANGE(A, N)), DYND_PP_META_NAME_RANGE(D, N)); \
    public: \
        static inline ndt::type make() { \
            ndt::type param_types[N] = {DYND_PP_JOIN_MAP_1(MAKE_TYPE, (,), DYND_PP_META_NAME_RANGE(D, N))}; \
            return make_funcproto(param_types, make_type<R>()); \
        } \
    };

DYND_PP_JOIN_MAP(FUNCPROTO_TYPE_FROM, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef FUNCPROTO_TYPE_FROM

template <typename R, typename A0>
struct funcproto_type_from<R (A0), true, false>
  : funcproto_type_from<R (), false, false> {
};

template <typename R, typename A0>
struct funcproto_type_from<R (A0), false, true>
  : funcproto_type_from<R (), false, false> {
};

#define FUNCPROTO_TYPE_FROM(N) \
    template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    struct funcproto_type_from<R DYND_PP_META_NAME_RANGE(A, N), true, false> \
      : funcproto_type_from<R DYND_PP_META_NAME_RANGE(A, DYND_PP_DEC(N)), false, false> { \
    }; \
\
    template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    struct funcproto_type_from<R DYND_PP_META_NAME_RANGE(A, N), false, true> \
      : funcproto_type_from<R DYND_PP_META_NAME_RANGE(A, DYND_PP_DEC(N)), false, false> { \
    };

DYND_PP_JOIN_MAP(FUNCPROTO_TYPE_FROM, (), DYND_PP_RANGE(2, DYND_PP_INC(DYND_ARG_MAX)))

#undef FUNCPROTO_TYPE_FROM

#undef PARTIAL_DECAY
#undef MAKE_TYPE

} // namespace detail

template <typename func_type>
inline ndt::type make_funcproto() {
    typedef typename funcproto_from<func_type>::type funcproto_type;

    return detail::funcproto_type_from<funcproto_type,
        aux::is_buffered<funcproto_type>::value, aux::is_thread_buffered<funcproto_type>::value>::make();
}

ndt::type make_generic_funcproto(intptr_t nargs);
} // namespace ndt

} // namespace dynd

#endif // _DYND__FUNCPROTO_TYPE_HPP_
