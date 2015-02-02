//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <vector>
#include <string>

#include <dynd/array.hpp>
#include <dynd/string.hpp>
#include <dynd/types/base_dim_type.hpp>

namespace dynd {

class ellipsis_dim_type : public base_dim_type {
    // m_name is either NULL or an immutable array of type "string"
    nd::string m_name;

public:
    ellipsis_dim_type(const nd::string &name, const ndt::type &element_type);

    virtual ~ellipsis_dim_type() {}

    inline const nd::string& get_name() const {
        return m_name;
    }

    inline std::string get_name_str() const {
        return m_name.is_null() ? "" : m_name.str();
    }

    void print_data(std::ostream& o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream& o) const;

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    ndt::type get_type_at_dimension(char **inout_arrmeta, intptr_t i,
                                    intptr_t total_ndim = 0) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                 memory_block_data *embedded_reference) const;
    size_t
    arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                                   memory_block_data *embedded_reference) const;
    void arrmeta_destruct(char *arrmeta) const;

    bool matches(const char *arrmeta, const ndt::type &other,
                 std::map<nd::string, ndt::type> &tp_vars) const;

    void get_dynamic_type_properties(
        const std::pair<std::string, gfunc::callable> **out_properties,
        size_t *out_count) const;
}; // class ellipsis_dim_type

namespace ndt {
    /** Makes an ellipsis type with the specified name and element type */
    inline ndt::type make_ellipsis_dim(const nd::string &name,
                                  const ndt::type &element_type)
    {
        return ndt::type(new ellipsis_dim_type(name, element_type), false);
    }

    /** Make an unnamed ellipsis type */
    inline ndt::type make_ellipsis_dim(const ndt::type &element_type)
    {
        return ndt::type(new ellipsis_dim_type(nd::array(), element_type),
                         false);
    }
} // namespace ndt

} // namespace dynd
