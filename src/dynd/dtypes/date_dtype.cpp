//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/dtypes/date_dtype.hpp>
#include <dynd/kernels/single_compare_kernel_instance.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/gfunc/make_callable.hpp>

#include <datetime_strings.h>

#define DYND_DATE_ORIGIN_YEAR 1970

using namespace std;
using namespace dynd;

date_dtype::date_dtype(date_unit_t unit)
    : m_unit(unit)
{
    switch (unit) {
        case date_unit_year:
        case date_unit_month:
        case date_unit_week:
        case date_unit_day:
            break;
        default:
            throw runtime_error("Unrecognized date unit in date dtype constructor");
    }
}

void date_dtype::print_element(std::ostream& o, const char *DYND_UNUSED(metadata), const char *data) const
{
    int32_t value = *reinterpret_cast<const int32_t *>(data);
    switch (m_unit) {
        case date_unit_year:
            o << datetime::make_iso_8601_date(value, datetime::datetime_unit_year);
            break;
        case date_unit_month:
            o << datetime::make_iso_8601_date(value, datetime::datetime_unit_month);
            break;
        case date_unit_week:
            o << datetime::make_iso_8601_date(value, datetime::datetime_unit_week);
            break;
        case date_unit_day:
            o << datetime::make_iso_8601_date(value, datetime::datetime_unit_day);
            break;
        default:
            o << "<corrupt date dtype>";
            break;
    }
}

void date_dtype::print_dtype(std::ostream& o) const
{
    if (m_unit == date_unit_day) {
        o << "date";
    } else {
        o << "date<" << m_unit << ">";
    }
}

bool date_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.extended() == this) {
            return true;
        } else if (src_dt.get_type_id() == date_type_id) {
            const date_dtype *src_fs = static_cast<const date_dtype*>(src_dt.extended());
            return src_fs->m_unit == m_unit;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

void date_dtype::get_single_compare_kernel(single_compare_kernel_instance& /*out_kernel*/) const {
    throw runtime_error("get_single_compare_kernel for date are not implemented");
}

void date_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel) const
{
    
}


bool date_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != date_type_id) {
        return false;
    } else {
        const date_dtype *dt = static_cast<const date_dtype*>(&rhs);
        return m_unit == dt->m_unit;
    }
}

static string property_get_encoding(const dtype& dt) {
    const date_dtype *d = static_cast<const date_dtype *>(dt.extended());
    stringstream ss;
    ss << d->get_unit();
    return ss.str();
}

static pair<string, gfunc::callable> date_dtype_properties[] = {
    pair<string, gfunc::callable>("unit", gfunc::make_callable(&property_get_encoding, "self"))
};

void date_dtype::get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, int *out_count) const
{
    *out_properties = date_dtype_properties;
    *out_count = sizeof(date_dtype_properties) / sizeof(date_dtype_properties[0]);
}
