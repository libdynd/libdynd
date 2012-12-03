//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATE_DTYPE_HPP_
#define _DYND__DATE_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

enum date_unit_t {
    date_unit_day,
    date_unit_week,
    date_unit_month,
    date_unit_year
};

inline std::ostream& operator<<(std::ostream& o, date_unit_t unit) {
    switch (unit) {
        case date_unit_day:
            o << "day";
            break;
        case date_unit_week:
            o << "week";
            break;
        case date_unit_month:
            o << "month";
            break;
        case date_unit_year:
            o << "year";
            break;
        default:
            o << "invalid (" << (int)unit << ")";
            break;
    }
    return o;
}

class date_dtype : public extended_dtype {
    date_unit_t m_unit;

public:
    date_dtype(date_unit_t unit);

    type_id_t get_type_id() const {
        return date_type_id;
    }
    dtype_kind_t get_kind() const {
        return datetime_kind;
    }
    // Expose the storage traits here
    size_t get_alignment() const {
        return 4;
    }
    size_t get_element_size() const {
        return 4;
    }

    date_unit_t unit() const {
        return m_unit;
    }

    void print_element(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return pod_memory_management;
    }

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    unary_specialization_kernel_instance& out_kernel) const;

    bool operator==(const extended_dtype& rhs) const;

    size_t get_metadata_size() const {
        return 0;
    }
    void metadata_destruct(char *DYND_UNUSED(metadata)) const {
    }
    void metadata_debug_print(const char *DYND_UNUSED(metadata), std::ostream& DYND_UNUSED(o), const std::string& DYND_UNUSED(indent)) const {
    }
};

inline dtype make_date_dtype(date_unit_t unit) {
    return dtype(new date_dtype(unit));
}

} // namespace dynd

#endif // _DYND__DATE_DTYPE_HPP_
