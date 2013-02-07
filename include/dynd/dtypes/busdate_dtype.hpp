//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BUSDATE_DTYPE_HPP_
#define _DYND__BUSDATE_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/ndobject.hpp>

namespace dynd {

enum busdate_roll_t {
    // Go forward in time to the following business day.
    busdate_roll_following,
    // Go backward in time to the preceding business day.
    busdate_roll_preceding,
    // Go forward in time to the following business day, unless it
    // crosses a month boundary, in which case go backward
    busdate_roll_modifiedfollowing,
    // Go backward in time to the preceding business day, unless it
    // crosses a month boundary, in which case go forward.
    busdate_roll_modifiedpreceding,
    // Produce a NaT for non-business days.
    busdate_roll_nat,
    // Raise an exception for non-business days.
    busdate_roll_throw
};

class busdate_dtype : public base_dtype {
    /** Strategy for handling dates that are not business dates */
    busdate_roll_t m_roll;
    /** Which days of the week are work days vs weekend */
    bool m_workweek[7];
    /** Cache of the non-weekend day count in the weekmask */
    int m_busdays_in_weekmask;
    /**
     * If non-NULL, a one-dimensional contiguous array of day unit date_dtype
     * which has no duplicates or holidays falling on a weekend.
     */
    ndobject m_holidays;

public:
    busdate_dtype(busdate_roll_t roll, const bool *weekmask, const ndobject& holidays);

    virtual ~busdate_dtype();

    busdate_roll_t get_roll() const {
        return m_roll;
    }

    const bool *get_weekmask() const {
        return m_workweek;
    }

    ndobject get_holidays() const {
        return m_holidays;
    }

    bool is_default_workweek() const {
        return m_workweek[0] && m_workweek[1] && m_workweek[2] && m_workweek[3] &&
                m_workweek[4] && !m_workweek[5] && !m_workweek[6];
    }

    void print_workweek(std::ostream& o) const;
    void print_holidays(std::ostream& o) const;

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return pod_memory_management;
    }

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(kernel_instance<compare_operations_t>& out_kernel) const;

    bool operator==(const base_dtype& rhs) const;

    void metadata_default_construct(char *DYND_UNUSED(metadata), int DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const {
    }
    void metadata_copy_construct(char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata), memory_block_data *DYND_UNUSED(embedded_reference)) const {
    }
    void metadata_destruct(char *DYND_UNUSED(metadata)) const {
    }
    void metadata_debug_print(const char *DYND_UNUSED(metadata), std::ostream& DYND_UNUSED(o), const std::string& DYND_UNUSED(indent)) const {
    }
};

inline dtype make_busdate_dtype(busdate_roll_t roll = busdate_roll_following, const bool *weekmask = NULL, const ndobject& holidays = ndobject()) {
    return dtype(new busdate_dtype(roll, weekmask, holidays), false);
}

} // namespace dynd

#endif // _DYND__BUSDATE_DTYPE_HPP_
