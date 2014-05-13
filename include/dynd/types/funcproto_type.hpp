//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FUNCPROTO_TYPE_HPP_
#define _DYND__FUNCPROTO_TYPE_HPP_

#include <vector>
#include <string>

#include <dynd/type.hpp>
#include <dynd/memblock/memory_block.hpp>

namespace dynd {

class funcproto_type : public base_type {
    std::vector<ndt::type> m_param_types;
    ndt::type m_return_type;

public:
    funcproto_type(size_t param_count, const ndt::type *param_types,
                 const ndt::type &return_type);

    virtual ~funcproto_type() {}

    inline size_t get_param_count() const {
        return m_param_types.size();
    }

    inline const ndt::type *get_param_types() const {
        return &m_param_types[0];
    }

    inline const std::vector<ndt::type>& get_param_types_vector() const {
        return m_param_types;
    }

    inline const ndt::type& get_return_type() const {
        return m_return_type;
    }

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_type(std::ostream& o) const;

    void transform_child_types(type_transform_fn_t transform_fn, void *extra,
                               ndt::type &out_transformed_tp,
                               bool &out_was_transformed) const;
    ndt::type get_canonical_type() const;

    ndt::type apply_linear_index(intptr_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_tp, bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *metadata,
                    const ndt::type& result_tp, char *out_metadata,
                    memory_block_data *embedded_reference,
                    size_t current_i, const ndt::type& root_tp,
                    bool leading_dimension, char **inout_data,
                    memory_block_data **inout_dataref) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    void metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_destruct(char *metadata) const;

    void get_dynamic_type_properties(
        const std::pair<std::string, gfunc::callable> **out_properties,
        size_t *out_count) const;
}; // class typevar_type

namespace ndt {
    /** Makes a funcproto type with the specified types */
    inline ndt::type make_funcproto(size_t param_count,
                                    const ndt::type *param_types,
                                    const ndt::type &return_type)
    {
        return ndt::type(
            new funcproto_type(param_count, param_types, return_type), false);
    }

    namespace detail {
        template<typename T>
        struct make_func_proto;

        template<typename R>
        struct make_func_proto<R ()> {
            static inline ndt::type make() {
                return make_funcproto(0, NULL, make_type<R>());
            }
        };

        template<typename R, typename T0>
        struct make_func_proto<R (T0)> {
            static inline ndt::type make() {
                ndt::type param_types[1] = {make_type<T0>()};
                return make_funcproto(1, param_types, make_type<R>());
            }
        };

        template<typename R, typename T0, typename T1>
        struct make_func_proto<R (T0, T1)> {
            static inline ndt::type make() {
                ndt::type param_types[2] = {make_type<T0>(), make_type<T1>()};
                return make_funcproto(2, param_types, make_type<R>());
            }
        };


        template<typename R, typename T0, typename T1, typename T2>
        struct make_func_proto<R (T0, T1, T2)> {
            static inline ndt::type make() {
                ndt::type param_types[3] = {make_type<T0>(), make_type<T1>(),
                                            make_type<T2>()};
                return make_funcproto(3, param_types, make_type<R>());
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
