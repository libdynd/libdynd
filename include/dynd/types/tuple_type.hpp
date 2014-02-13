//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__TUPLE_TYPE_HPP_
#define _DYND__TUPLE_TYPE_HPP_

#include <vector>

#include <dynd/type.hpp>

namespace dynd {

class tuple_type : public base_type {
    std::vector<ndt::type> m_fields;
    std::vector<size_t> m_offsets;
    std::vector<size_t> m_metadata_offsets;
    bool m_is_standard_layout;

    bool compute_is_standard_layout() const;
public:
    tuple_type(const std::vector<ndt::type>& fields);
    tuple_type(const std::vector<ndt::type>& fields, const std::vector<size_t> offsets,
                        size_t data_size, size_t alignment);

    virtual ~tuple_type();

    const std::vector<ndt::type>& get_fields() const {
        return m_fields;
    }

    const std::vector<size_t>& get_offsets() const {
        return m_offsets;
    }

    /**
     * Returns true if the layout is standard, i.e. constructable without
     * specifying the offsets/alignment/data_size.
     */
    bool is_standard_layout() const {
        return m_is_standard_layout;
    }

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_type(std::ostream& o) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;
}; // class tuple_type

namespace ndt {
    /** Makes a tuple type with the specified fields, using the standard layout */
    inline ndt::type make_tuple(const std::vector<ndt::type>& fields) {
        return ndt::type(new tuple_type(fields), false);
    }

    /** Makes a tuple type with the specified fields and layout */
    inline ndt::type make_tuple(const std::vector<ndt::type>& fields, const std::vector<size_t> offsets,
                    size_t data_size, size_t alignment)
    {
        return ndt::type(new tuple_type(fields, offsets, data_size, alignment), false);
    }

    /** Makes a tuple type with the specified fields, using the standard layout */
    inline ndt::type make_tuple(const ndt::type& tp0)
    {
        std::vector<ndt::type> fields;
        fields.push_back(tp0);
        return ndt::make_tuple(fields);
    }

    /** Makes a tuple type with the specified fields, using the standard layout */
    inline ndt::type make_tuple(const ndt::type& tp0, const ndt::type& tp1)
    {
        std::vector<ndt::type> fields;
        fields.push_back(tp0);
        fields.push_back(tp1);
        return ndt::make_tuple(fields);
    }

    /** Makes a tuple type with the specified fields, using the standard layout */
    inline ndt::type make_tuple(const ndt::type& tp0, const ndt::type& tp1, const ndt::type& tp2)
    {
        std::vector<ndt::type> fields;
        fields.push_back(tp0);
        fields.push_back(tp1);
        fields.push_back(tp2);
        return ndt::make_tuple(fields);
    }

    /** Makes a tuple type with the specified fields, using the standard layout */
    inline ndt::type make_tuple(const ndt::type& tp0, const ndt::type& tp1, const ndt::type& tp2, const ndt::type& tp3)
    {
        std::vector<ndt::type> fields;
        fields.push_back(tp0);
        fields.push_back(tp1);
        fields.push_back(tp2);
        fields.push_back(tp3);
        return ndt::make_tuple(fields);
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__TUPLE_TYPE_HPP_
