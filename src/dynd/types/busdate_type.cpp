//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/types/busdate_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/exceptions.hpp>

#include <datetime_strings.h>

using namespace std;
using namespace dynd;

dynd::busdate_type::busdate_type(busdate_roll_t roll, const bool *weekmask,
                                 const nd::array &holidays)
    : base_type(busdate_type_id, datetime_kind, 4, 4, type_flag_scalar, 0, 0,
                0),
      m_roll(roll)
{
    memcpy(m_workweek, weekmask, sizeof(m_workweek));
    m_busdays_in_weekmask = 0;
    for (int i = 0; i < 7; ++i) {
        m_busdays_in_weekmask += weekmask[i] ? 1 : 0;
    }
    if (!holidays.is_null()) {
        nd::array hol = holidays.ucast(ndt::make_date()).eval_immutable();
        // TODO: Make sure hol is contiguous and one-dimensional
        m_holidays = hol;
    }
}

busdate_type::~busdate_type()
{
}

void dynd::busdate_type::print_workweek(std::ostream& o) const
{
    if (m_workweek[0]) o << "Mo";
    if (m_workweek[1]) o << "Tu";
    if (m_workweek[2]) o << "We";
    if (m_workweek[3]) o << "Th";
    if (m_workweek[4]) o << "Fr";
    if (m_workweek[5]) o << "Sa";
    if (m_workweek[6]) o << "Su";
}

void dynd::busdate_type::print_holidays(std::ostream& /*o*/) const
{
    throw std::runtime_error("busdate_type::print_holidays to be implemented");
}

void dynd::busdate_type::print_data(std::ostream &o,
                                    const char *DYND_UNUSED(arrmeta),
                                    const char *data) const
{
  date_ymd ymd;
  ymd.set_from_days(*reinterpret_cast<const int32_t *>(data));
  string s = ymd.to_str();
  if (s.empty()) {
    o << "NA";
  } else {
    o << s;
  }
}

void dynd::busdate_type::print_type(std::ostream& o) const
{
    if (m_roll == busdate_roll_following && is_default_workweek() && m_holidays.is_null()) {
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
        if (!m_holidays.is_null()) {
            if (comma) o << ", ";
            o << "holidays=[";
            print_holidays(o);
            o << "]";
        }
        o << ">";
    }
}

bool dynd::busdate_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    if (dst_tp.extended() == this) {
        if (src_tp.extended() == this) {
            return true;
        } else if (src_tp.get_type_id() == date_type_id) {
            const busdate_type *src_fs = static_cast<const busdate_type*>(src_tp.extended());
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

bool dynd::busdate_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != busdate_type_id) {
        return false;
    } else {
        const busdate_type *dt = static_cast<const busdate_type*>(&rhs);
        return m_roll == dt->m_roll && memcmp(m_workweek, dt->m_workweek, sizeof(m_workweek)) == 0 &&
                m_holidays.equals_exact(dt->m_holidays);
    }
}
