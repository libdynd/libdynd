//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/dtypes/busdate_dtype.hpp>
#include <dynd/dtypes/date_dtype.hpp>
#include <dynd/kernels/single_compare_kernel_instance.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/exceptions.hpp>

#include <datetime_strings.h>

using namespace std;
using namespace dynd;

dynd::busdate_dtype::busdate_dtype(busdate_roll_t roll, const bool *weekmask, const ndarray& holidays)
    : m_roll(roll)
{
    memcpy(m_workweek, weekmask, sizeof(m_workweek));
    if (holidays.get_node() != NULL) {
        ndarray hol = holidays.as_dtype(make_date_dtype(date_unit_day)).eval_immutable();

    }
}

void dynd::busdate_dtype::print_workweek(std::ostream& o) const
{
    if (m_workweek[0]) o << "Mo";
    if (m_workweek[1]) o << "Tu";
    if (m_workweek[2]) o << "We";
    if (m_workweek[3]) o << "Th";
    if (m_workweek[4]) o << "Fr";
    if (m_workweek[5]) o << "Sa";
    if (m_workweek[6]) o << "Su";
}

void dynd::busdate_dtype::print_holidays(std::ostream& /*o*/) const
{
    throw std::runtime_error("busdate_dtype::print_holidays to be implemented");
}

void dynd::busdate_dtype::print_element(std::ostream& o, const char *DYND_UNUSED(metadata), const char *data) const
{
    int32_t value = *reinterpret_cast<const int32_t *>(data);
    o << datetime::make_iso_8601_date(value, datetime::datetime_unit_day);
}

void dynd::busdate_dtype::print_dtype(std::ostream& o) const
{
    if (m_roll == busdate_roll_following && is_default_workweek() && m_holidays.get_node() == NULL) {
        o << "busdate";
    } else {
        bool comma = false;
        o << "date<";
        if (m_roll != busdate_roll_following) {
            o << "roll=" << m_roll;
            comma = true;
        }
        if (!is_default_workweek()) {
            if (comma) o << ", ";
            o << "workweek=";
            print_workweek(o);
            comma = true;
        }
        if (m_holidays.get_node() != NULL) {
            if (comma) o << ", ";
            o << "holidays=[";
            print_holidays(o);
            o << "]";
        }
        o << ">";
    }
}

bool dynd::busdate_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.extended() == this) {
            return true;
        } else if (src_dt.get_type_id() == date_type_id) {
            const busdate_dtype *src_fs = static_cast<const busdate_dtype*>(src_dt.extended());
            // No need to compare the roll policy, just the weekmask and holidays determine this
            return memcmp(m_workweek, src_fs->m_workweek, sizeof(m_workweek)) == 0 &&
                    m_holidays.equals_exact(src_fs->m_holidays);
        } else {
            return false;
        }
    } else {
        return false;
    }
}

void dynd::busdate_dtype::get_single_compare_kernel(single_compare_kernel_instance& /*out_kernel*/) const {
    throw runtime_error("get_single_compare_kernel for date are not implemented");
}

void dynd::busdate_dtype::get_dtype_assignment_kernel(const dtype& /*dst_dt*/, const dtype& /*src_dt*/,
                assign_error_mode /*errmode*/,
                unary_specialization_kernel_instance& /*out_kernel*/) const
{
    throw runtime_error("conversions for date are not implemented");
}


bool dynd::busdate_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != busdate_type_id) {
        return false;
    } else {
        const busdate_dtype *dt = static_cast<const busdate_dtype*>(&rhs);
        return m_roll == dt->m_roll && memcmp(m_workweek, dt->m_workweek, sizeof(m_workweek)) == 0 &&
                m_holidays.equals_exact(dt->m_holidays);
    }
}
