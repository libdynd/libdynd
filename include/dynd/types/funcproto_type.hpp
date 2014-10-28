//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FUNCPROTO_TYPE_HPP_
#define _DYND__FUNCPROTO_TYPE_HPP_

#include <vector>
#include <string>

#include <dynd/array.hpp>
#include <dynd/funcproto.hpp>
#include <dynd/pp/comparison.hpp>
#include <dynd/pp/list.hpp>
#include <dynd/pp/meta.hpp>
#include <dynd/types/fixed_dimsym_type.hpp>

namespace dynd {

class funcproto_type : public base_type {
    // This is always a contiguous immutable "N * type" array
    nd::array m_arg_types;
    intptr_t m_narg, m_nsrc, m_naux;
    ndt::type m_return_type;

public:
    funcproto_type(const nd::array &arg_types, const ndt::type &return_type, intptr_t naux = 0);

    virtual ~funcproto_type() {}

    const nd::array& get_arg_types() const {
        return m_arg_types;
    }

    const ndt::type *get_arg_types_raw() const {
        return reinterpret_cast<const ndt::type *>(
            m_arg_types.get_readonly_originptr());
    }
    const ndt::type &get_arg_type(intptr_t i) const {
        return get_arg_types_raw()[i];
    }

    intptr_t get_nsrc() const {
        return m_nsrc;
    }

    intptr_t get_naux() const {
        return m_naux;
    }

    intptr_t get_narg() const {
        return m_narg;
    }

    const ndt::type& get_return_type() const {
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

namespace ndt { namespace detail {

#define PARTIAL_DECAY(TYPENAME) std::remove_cv<typename std::remove_reference<TYPENAME>::type>::type
#define MAKE_TYPE(TYPENAME) make_type<TYPENAME>()

template <typename funcproto_type>
struct funcproto_type_factory;

#define FUNCPROTO_TYPE_FACTORY(NARG) \
    template <DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_PREPEND(R, DYND_PP_META_NAME_RANGE(A, NARG)))> \
    struct funcproto_type_factory<R DYND_PP_META_NAME_RANGE(A, NARG)> { \
        static ndt::type make(intptr_t naux) { \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
                DYND_PP_MAP_1(PARTIAL_DECAY, DYND_PP_META_NAME_RANGE(A, NARG)), DYND_PP_META_NAME_RANGE(D, NARG)); \
            DYND_PP_IF_ELSE(NARG)( \
                ndt::type arg_tp[NARG] = {DYND_PP_JOIN_MAP_1(MAKE_TYPE, (,), DYND_PP_META_NAME_RANGE(D, NARG))}; \
            )( \
                nd::array arg_tp = nd::empty(0, ndt::make_type()); \
                arg_tp.flag_as_immutable(); \
            ) \
            return make_funcproto(arg_tp, make_type<R>(), naux); \
        } \
    };

DYND_PP_JOIN_MAP(FUNCPROTO_TYPE_FACTORY, (), DYND_PP_RANGE(DYND_PP_INC(DYND_ARG_MAX)))

#undef FUNCPROTO_TYPE_FACTORY

#define FUNCPROTO_TYPE_FACTORY(NARG) \
    template <DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_PREPEND(R, DYND_PP_META_NAME_RANGE(A, NARG)))> \
    struct funcproto_type_factory<void DYND_PP_PREPEND(R &, DYND_PP_META_NAME_RANGE(A, NARG))> { \
        static ndt::type make(intptr_t naux) { \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
                DYND_PP_MAP_1(PARTIAL_DECAY, DYND_PP_META_NAME_RANGE(A, NARG)), DYND_PP_META_NAME_RANGE(D, NARG)); \
            DYND_PP_IF_ELSE(NARG)( \
                ndt::type arg_tp[NARG] = {DYND_PP_JOIN_MAP_1(MAKE_TYPE, (,), DYND_PP_META_NAME_RANGE(D, NARG))}; \
            )( \
                nd::array arg_tp = nd::empty(0, ndt::make_type()); \
                arg_tp.flag_as_immutable(); \
            ) \
            return make_funcproto(arg_tp, make_type<R>(), naux); \
        } \
    };

DYND_PP_JOIN_MAP(FUNCPROTO_TYPE_FACTORY, (), DYND_PP_RANGE(DYND_PP_INC(DYND_ARG_MAX)))

#undef FUNCPROTO_TYPE_FACTORY

#undef PARTIAL_DECAY
#undef MAKE_TYPE

} // namespace ndt::detail

/** Makes a funcproto type with the specified types */
inline ndt::type make_funcproto(const nd::array &arg_types,
                                const ndt::type &return_type,
                                intptr_t naux = 0)
{
    return ndt::type(
        new funcproto_type(arg_types, return_type, naux), false);
}

/** Makes a funcproto type with the specified types */
inline ndt::type make_funcproto(intptr_t narg,
                                const ndt::type *arg_types,
                                const ndt::type &return_type)
{
    nd::array tmp = nd::empty(narg, ndt::make_type());
    ndt::type *tmp_vals =
        reinterpret_cast<ndt::type *>(tmp.get_readwrite_originptr());
    for (intptr_t i = 0; i != narg; ++i) {
        tmp_vals[i] = arg_types[i];
    }
    tmp.flag_as_immutable();
    return ndt::type(
        new funcproto_type(tmp, return_type), false);
}

/** Makes a unary funcproto type with the specified types */
inline ndt::type make_funcproto(const ndt::type& single_arg_type,
                                const ndt::type &return_type)
{
    ndt::type arg_types[1] = {single_arg_type};
    return ndt::type(
        new funcproto_type(arg_types, return_type), false);
}

/** Makes a funcproto type from the C++ function type */
template <typename funcproto_type>
ndt::type make_funcproto(intptr_t naux = 0) {
    return detail::funcproto_type_factory<funcproto_type>::make(naux);
}

ndt::type make_generic_funcproto(intptr_t nargs);

} // namespace ndt

} // namespace dynd

#endif // _DYND__FUNCPROTO_TYPE_HPP_
