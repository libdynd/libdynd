//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
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

    void print_dtype(std::ostream& o) const;

    bool is_lossless_assignment(const ndt::type& dst_dt, const ndt::type& src_dt) const;

    bool operator==(const base_type& rhs) const;
}; // class tuple_type

/** Makes a tuple dtype with the specified fields, using the standard layout */
inline ndt::type make_tuple_type(const std::vector<ndt::type>& fields) {
    return ndt::type(new tuple_type(fields), false);
}

/** Makes a tuple dtype with the specified fields and layout */
inline ndt::type make_tuple_type(const std::vector<ndt::type>& fields, const std::vector<size_t> offsets,
                size_t data_size, size_t alignment)
{
    return ndt::type(new tuple_type(fields, offsets, data_size, alignment), false);
}

/** Makes a tuple dtype with the specified fields, using the standard layout */
inline ndt::type make_tuple_type(const ndt::type& dt0)
{
    std::vector<ndt::type> fields;
    fields.push_back(dt0);
    return make_tuple_type(fields);
}

/** Makes a tuple dtype with the specified fields, using the standard layout */
inline ndt::type make_tuple_type(const ndt::type& dt0, const ndt::type& dt1)
{
    std::vector<ndt::type> fields;
    fields.push_back(dt0);
    fields.push_back(dt1);
    return make_tuple_type(fields);
}

/** Makes a tuple dtype with the specified fields, using the standard layout */
inline ndt::type make_tuple_type(const ndt::type& dt0, const ndt::type& dt1, const ndt::type& dt2)
{
    std::vector<ndt::type> fields;
    fields.push_back(dt0);
    fields.push_back(dt1);
    fields.push_back(dt2);
    return make_tuple_type(fields);
}

/** Makes a tuple dtype with the specified fields, using the standard layout */
inline ndt::type make_tuple_type(const ndt::type& dt0, const ndt::type& dt1, const ndt::type& dt2, const ndt::type& dt3)
{
    std::vector<ndt::type> fields;
    fields.push_back(dt0);
    fields.push_back(dt1);
    fields.push_back(dt2);
    fields.push_back(dt3);
    return make_tuple_type(fields);
}

} // namespace dynd

#endif // _DYND__TUPLE_TYPE_HPP_
