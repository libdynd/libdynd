//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

/**
 * The option type represents data which may or may not be there.
 */

#ifndef _DYND__OPTION_TYPE_HPP_
#define _DYND__OPTION_TYPE_HPP_

#include <dynd/type.hpp>

namespace dynd {

class option_type : public base_type {
    ndt::type m_value_tp;

public:
    option_type(const ndt::type& value_tp);

    virtual ~option_type();

    const ndt::type& get_value_type() const {
        return m_value_tp.value_type();
    }

    void print_data(std::ostream &o, const char *metadata,
                    const char *data) const;

    void print_type(std::ostream& o) const;

    bool is_expression() const;
    bool is_unique_data_owner(const char *metadata) const;
    void transform_child_types(type_transform_fn_t transform_fn, void *extra,
                    ndt::type& out_transformed_tp, bool& out_was_transformed) const;
    ndt::type get_canonical_type() const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    void metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_reset_buffers(char *metadata) const;
    void metadata_finalize_buffers(char *metadata) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    void get_dynamic_type_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
};

namespace ndt {
    ndt::type make_option(const ndt::type& value_tp);

    template<typename Tnative>
    inline ndt::type make_option() {
        return make_option(ndt::make_type<Tnative>());
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__OPTION_TYPE_HPP_
