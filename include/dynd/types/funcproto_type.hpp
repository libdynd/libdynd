//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FUNCPROTO_TYPE_HPP_
#define _DYND__FUNCPROTO_TYPE_HPP_

#include <vector>
#include <string>

#include <dynd/array.hpp>
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

    void transform_child_types(type_transform_fn_t transform_fn, void *extra,
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

    void arrmeta_default_construct(char *arrmeta, intptr_t ndim,
                                   const intptr_t *shape,
                                   bool blockref_alloc) const;
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
        nd::array tmp =
            nd::typed_empty(1, &param_count, ndt::make_strided_of_type());
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
        template<typename T>
        struct make_func_proto;

        template<typename R>
        struct make_func_proto<R ()> {
            static inline ndt::type make() {
              intptr_t zero = 0;
              nd::array param_types =
                  nd::typed_empty(1, &zero, ndt::make_strided_of_type());
              param_types.flag_as_immutable();
              return make_funcproto(param_types, make_type<R>());
            }
        };

        template<typename R, typename T0>
        struct make_func_proto<R (T0)> {
            static inline ndt::type make() {
                ndt::type param_types[1] = {make_type<T0>()};
                return make_funcproto(param_types, make_type<R>());
            }
        };

        template<typename R, typename T0, typename T1>
        struct make_func_proto<R (T0, T1)> {
            static inline ndt::type make() {
                ndt::type param_types[2] = {make_type<T0>(), make_type<T1>()};
                return make_funcproto(param_types, make_type<R>());
            }
        };


        template<typename R, typename T0, typename T1, typename T2>
        struct make_func_proto<R (T0, T1, T2)> {
            static inline ndt::type make() {
                ndt::type param_types[3] = {make_type<T0>(), make_type<T1>(),
                                            make_type<T2>()};
                return make_funcproto(param_types, make_type<R>());
            }
        };
        // TODO use the pp lib to generate this
    } // namespace detail

    template<class T>
    inline ndt::type make_funcproto()
    {
        return detail::make_func_proto<T>::make();
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__FUNCPROTO_TYPE_HPP_
