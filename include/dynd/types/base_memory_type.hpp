//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BASE_MEMORY_TYPE_HPP_
#define _DYND__BASE_MEMORY_TYPE_HPP_

#include <dynd/types/base_type.hpp>
#include <dynd/type.hpp>

using namespace std;
using namespace dynd;

namespace dynd {

/**
 * Base class for all types of memory_kind.
 */
class base_memory_type : public base_type {
protected:
    ndt::type m_target_tp;
    size_t m_target_metadata_offset;
public:
    inline base_memory_type(type_id_t type_id, const ndt::type& target_tp, size_t data_size,
            size_t alignment, size_t target_metadata_offset, flags_type flags)
        : base_type(type_id, memory_kind, data_size, alignment, flags, target_metadata_offset + target_tp.get_metadata_size(),
            target_tp.get_ndim()), m_target_tp(target_tp), m_target_metadata_offset(target_metadata_offset)
    {
        if (target_tp.get_kind() == uniform_dim_kind || target_tp.get_kind() == memory_kind
                    || target_tp.get_kind() == pattern_kind) {
            stringstream ss;
            ss << "a memory space cannot be specified for type " << target_tp;
            throw runtime_error(ss.str());
        }
    }

    virtual ~base_memory_type();

    inline const ndt::type& get_target_type() const {
        return m_target_tp;
    }

    virtual size_t get_default_data_size(intptr_t ndim, const intptr_t *shape) const;

    virtual void print_data(std::ostream& o, const char *metadata, const char *data) const;

    virtual bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    virtual bool operator==(const base_type& rhs) const;

    inline bool is_type_subarray(const ndt::type& subarray_tp) const {
        return (!subarray_tp.is_builtin() && (*this) == (*subarray_tp.extended())) ||
                        m_target_tp.is_type_subarray(subarray_tp);
    }

    virtual void transform_child_types(type_transform_fn_t transform_fn, void *extra,
                    ndt::type& out_transformed_tp, bool& out_was_transformed) const;
    virtual ndt::type get_canonical_type() const;

    virtual ndt::type with_replaced_target_type(const ndt::type& target_tp) const = 0;

    virtual void metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const;
    virtual void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    virtual void metadata_destruct(char *metadata) const;

    virtual void data_alloc(char **data, size_t size) const = 0;
    virtual void data_zeroinit(char *data, size_t size) const = 0;
    virtual void data_free(char *data) const = 0;

    virtual void get_dynamic_type_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
};

inline const ndt::type& get_target_type(const ndt::type& tp) {
    if (tp.get_kind() == memory_kind) {
        return static_cast<const base_memory_type*>(tp.extended())->get_target_type();
    } else {
        return tp;
    }
}

} // namespace dynd

#endif // _DYND__BASE_MEMORY_TYPE_HPP_
