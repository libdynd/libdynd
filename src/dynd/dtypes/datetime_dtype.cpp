//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <time.h>

#include <cerrno>
#include <algorithm>

#include <dynd/dtypes/datetime_dtype.hpp>
#include <dynd/dtypes/property_dtype.hpp>
#include <dynd/dtypes/cstruct_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/unary_expr_dtype.hpp>
#include <dynd/kernels/date_assignment_kernels.hpp>
#include <dynd/kernels/date_expr_kernels.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/gfunc/make_callable.hpp>
#include <dynd/ndobject_iter.hpp>

#include <datetime_strings.h>
#include <datetime_localtime.h>

using namespace std;
using namespace dynd;

std::ostream& dynd::operator<<(std::ostream& o, datetime_unit_t unit)
{
    switch (unit) {
        case datetime_unit_hour:
            return (o << "hour");
        case datetime_unit_minute:
            return (o << "minute");
        case datetime_unit_second:
            return (o << "second");
        case datetime_unit_msecond:
            return (o << "msecond");
        case datetime_unit_usecond:
            return (o << "usecond");
        case datetime_unit_nsecond:
            return (o << "nsecond");
    }
    stringstream ss;
    ss << "invalid datetime unit " << (int32_t)unit << " provided to ";
    ss << "datetime dynd type constructor";
    throw runtime_error(ss.str());
}

namespace {
    static dtype datetime_default_structs[6] = {
        make_cstruct_dtype(
            make_dtype<int32_t>(), "year", make_dtype<int16_t>(), "month",
            make_dtype<int16_t>(), "day", make_dtype<int16_t>(), "hour"),
        make_cstruct_dtype(
            make_dtype<int32_t>(), "year", make_dtype<int16_t>(), "month",
            make_dtype<int16_t>(), "day", make_dtype<int16_t>(), "hour",
            make_dtype<int16_t>(), "min"),
        make_cstruct_dtype(
            make_dtype<int32_t>(), "year", make_dtype<int16_t>(), "month",
            make_dtype<int16_t>(), "day", make_dtype<int16_t>(), "hour",
            make_dtype<int16_t>(), "min", make_dtype<int16_t>(), "sec"),
        make_cstruct_dtype(
            make_dtype<int32_t>(), "year", make_dtype<int16_t>(), "month",
            make_dtype<int16_t>(), "day", make_dtype<int16_t>(), "hour",
            make_dtype<int16_t>(), "min", make_dtype<int16_t>(), "sec",
            make_dtype<int16_t>(), "msec"),
        make_cstruct_dtype(
            make_dtype<int32_t>(), "year", make_dtype<int16_t>(), "month",
            make_dtype<int16_t>(), "day", make_dtype<int16_t>(), "hour",
            make_dtype<int16_t>(), "min", make_dtype<int16_t>(), "sec",
            make_dtype<int32_t>(), "usec"),
        make_cstruct_dtype(
            make_dtype<int32_t>(), "year", make_dtype<int16_t>(), "month",
            make_dtype<int16_t>(), "day", make_dtype<int16_t>(), "hour",
            make_dtype<int16_t>(), "min", make_dtype<int16_t>(), "sec",
            make_dtype<int32_t>(), "nsec")
    };
    /**
     * Returns a reference to a static struct for the given
     * datetime unit.
     */
    const dtype& get_default_struct_dtype(datetime_unit_t unit) {
        if ((int32_t)unit > 0 && (int32_t)unit < 6) {
            return datetime_default_structs[unit];
        } else {
            stringstream ss;
            ss << "invalid datetime unit " << (int32_t)unit << " provided to ";
            ss << "datetime dynd type constructor";
            throw runtime_error(ss.str());
        }
    }

    datetime::datetime_unit_t dynd_unit_to_datetime_unit(datetime_unit_t unit) {
        switch (unit) {
            case datetime_unit_hour:
                return datetime::datetime_unit_hour;
            case datetime_unit_minute:
                return datetime::datetime_unit_minute;
            case datetime_unit_second:
                return datetime::datetime_unit_second;
            case datetime_unit_msecond:
                return datetime::datetime_unit_ms;
            case datetime_unit_usecond:
                return datetime::datetime_unit_us;
            case datetime_unit_nsecond:
                return datetime::datetime_unit_ns;
        }
        stringstream ss;
        ss << "invalid datetime unit " << (int32_t)unit << " provided to ";
        ss << "datetime dynd type constructor";
        throw runtime_error(ss.str());
    }
} // anonymous namespace

datetime_dtype::datetime_dtype(datetime_unit_t unit, datetime_tz_t timezone)
    : base_dtype(datetime_type_id, datetime_kind, 8, scalar_align_of<int64_t>::value, dtype_flag_scalar, 0, 0),
        m_default_struct_dtype(::get_default_struct_dtype(unit)), m_unit(unit),
        m_timezone(timezone)
{
}

datetime_dtype::~datetime_dtype()
{
}

void datetime_dtype::set_cal(const char *metadata, char *data, assign_error_mode errmode,
                int32_t year, int32_t month, int32_t day,
                int32_t hour, int32_t min, int32_t sec, int32_t nsec) const
{
    if (errmode != assign_error_none) {
        if (!datetime::is_valid_ymd(year, month, day)) {
            stringstream ss;
            ss << "invalid input year/month/day " << year << "/" << month << "/" << day;
            throw runtime_error(ss.str());
        }
        if (hour < 0 || hour >= 24) {
            stringstream ss;
            ss << "invalid input hour " << hour << " for " << dtype(this, true);
            throw runtime_error(ss.str());
        }
        if (min < 0 || min >= 60 || (min != 0 && m_unit < datetime_unit_minute)) {
            stringstream ss;
            ss << "invalid input minute " << min << " for " << dtype(this, true);
            throw runtime_error(ss.str());
        }
        if (sec < 0 || sec >= 60 || (sec != 0 && m_unit < datetime_unit_second)) {
            stringstream ss;
            ss << "invalid input second " << sec << " for " << dtype(this, true);
            throw runtime_error(ss.str());
        }
        if (nsec < 0 || nsec >= 1000000000) {
            stringstream ss;
            ss << "invalid input nanosecond " << nsec << " for " << dtype(this, true);
            throw runtime_error(ss.str());
        }
    }

    int64_t result = datetime::ymd_to_days(year, month, day) * 24 + hour;
    if (m_unit >= datetime_unit_minute) {
        result = result * 60 + min;
        if (m_unit >= datetime_unit_second) {
            result = result * 60 + sec;
            if (m_unit >= datetime_unit_msecond) {
                int64_t frac;
                switch (m_unit) {
                    case datetime_unit_msecond:
                        frac = nsec % 1000000;
                        if (errmode != assign_error_none && frac * 1000000 != nsec) {
                            stringstream ss;
                            ss << "invalid input nanosecond " << nsec << " for " << dtype(this, true);
                            throw runtime_error(ss.str());
                        }
                        result = result * 1000 + frac;
                        break;
                    case datetime_unit_usecond:
                        frac = nsec % 1000;
                        if (errmode != assign_error_none && frac * 1000 != nsec) {
                            stringstream ss;
                            ss << "invalid input nanosecond " << nsec << " for " << dtype(this, true);
                            throw runtime_error(ss.str());
                        }
                        result = result * 1000000 + frac;
                        break;
                    case datetime_unit_nsecond:
                        result = result * 1000000000 + nsec;
                        break;
                    default:
                        break;
                }
            }
        }
    }

    *reinterpret_cast<int64_t *>(data) = result;
}

void datetime_dtype::set_utf8_string(const char *DYND_UNUSED(metadata),
                char *data, assign_error_mode errmode, const std::string& utf8_str) const
{
    datetime::datetime_conversion_rule_t casting;
    switch (errmode) {
        case assign_error_fractional:
        case assign_error_inexact:
            casting = datetime::datetime_conversion_strict;
            break;
        default:
            casting = datetime::datetime_conversion_relaxed;
            break;
    }
    // TODO: Parsing adjustments/error handling based on the timezone
    *reinterpret_cast<int64_t *>(data) = datetime::parse_iso_8601_datetime(
                            utf8_str, dynd_unit_to_datetime_unit(m_unit), casting);
}


void datetime_dtype::get_cal(const char *DYND_UNUSED(metadata), const char *data,
                int32_t &out_year, int32_t &out_month, int32_t &out_day,
                int32_t &out_hour, int32_t &out_min, int32_t &out_sec, int32_t &out_nsec) const
{
    datetime::datetime_fields fields;
    fields.set_from_datetime_val(*reinterpret_cast<const int64_t *>(data),
                    dynd_unit_to_datetime_unit(m_unit));
    out_year = (int32_t)fields.year;
    out_month = fields.month;
    out_day = fields.day;
    out_hour = fields.hour;
    out_min = fields.min;
    out_sec = fields.sec;
    // TODO: Adjust the datetime library to make this more efficient.
    out_nsec = fields.us * 1000 + fields.ps / 1000;
}

void datetime_dtype::print_data(std::ostream& o, const char *DYND_UNUSED(metadata), const char *data) const
{
    int64_t value = *reinterpret_cast<const int64_t *>(data);
    datetime::datetime_fields fields;
    fields.set_from_datetime_val(*reinterpret_cast<const int64_t *>(data),
                    dynd_unit_to_datetime_unit(m_unit));
    // TODO: Handle distiction between printing abstract and UTC units
    o << datetime::make_iso_8601_datetime(&fields,
                    false, dynd_unit_to_datetime_unit(m_unit));
}

void datetime_dtype::print_dtype(std::ostream& o) const
{
    o << "datetime<unit=" << m_unit << ",tz=";
    switch (m_timezone) {
        case tz_abstract:
            o << "abstract";
            break;
        case tz_utc:
            o << "utc";
            break;
        default:
            o << "(invalid " << (int32_t)m_timezone << ")";
            break;
    }
    o << ">";
}

bool datetime_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.extended() == this) {
            return true;
        } else if (src_dt.get_type_id() == date_type_id) {
            // There is only one possibility for the date dtype (TODO: timezones!)
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

bool datetime_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != datetime_type_id) {
        return false;
    } else {
        const datetime_dtype& r = static_cast<const datetime_dtype &>(rhs);
        // TODO: When "other" timezone data is supported, need to compare them too
        return m_unit == r.m_unit && m_timezone == r.m_timezone;
    }
}

size_t datetime_dtype::make_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_dt.extended()) {
        if (src_dt.get_type_id() == date_type_id) {
            return make_pod_dtype_assignment_kernel(out, offset_out,
                            get_data_size(), get_data_alignment(), kernreq);
        } else if (src_dt.get_kind() == string_kind) {
            // Assignment from strings
            return make_string_to_date_assignment_kernel(out, offset_out,
                            src_dt, src_metadata,
                            kernreq, errmode, ectx);
        } else if (src_dt.get_kind() == struct_kind) {
            // Convert to struct using the "struct" property
            return ::make_assignment_kernel(out, offset_out,
                make_property_dtype(dst_dt, "struct"), dst_metadata,
                src_dt, src_metadata,
                kernreq, errmode, ectx);
        } else if (!src_dt.is_builtin()) {
            return src_dt.extended()->make_assignment_kernel(out, offset_out,
                            dst_dt, dst_metadata,
                            src_dt, src_metadata,
                            kernreq, errmode, ectx);
        }
    } else {
        if (dst_dt.get_kind() == string_kind) {
            // Assignment to strings
            return make_date_to_string_assignment_kernel(out, offset_out,
                            dst_dt, dst_metadata,
                            kernreq, errmode, ectx);
        } else if (dst_dt.get_kind() == struct_kind) {
            // Convert to struct using the "struct" property
            return ::make_assignment_kernel(out, offset_out,
                dst_dt, dst_metadata,
                make_property_dtype(src_dt, "struct"), src_metadata,
                kernreq, errmode, ectx);
        }
        // TODO
    }

    stringstream ss;
    ss << "Cannot assign from " << src_dt << " to " << dst_dt;
    throw runtime_error(ss.str());
}


///////// properties on the dtype

//static pair<string, gfunc::callable> datetime_dtype_properties[] = {
//};

void datetime_dtype::get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = NULL; //datetime_dtype_properties;
    *out_count = 0; //sizeof(datetime_dtype_properties) / sizeof(datetime_dtype_properties[0]);
}

///////// functions on the dtype

static ndobject function_dtype_now(const dtype& dt) {
    throw runtime_error("TODO: implement datetime.now function");
    datetime::datetime_fields fields;
    //datetime::fill_current_local_datetime(&fields);
    ndobject result = empty(dt);
    //*reinterpret_cast<int32_t *>(result.get_readwrite_originptr()) = datetime::ymd_to_days(ymd);
    // Make the result immutable (we own the only reference to the data at this point)
    result.flag_as_immutable();
    return result;
}

static ndobject function_dtype_construct(const dtype& DYND_UNUSED(dt), const ndobject& year, const ndobject& month, const ndobject& day)
{
    throw runtime_error("dynd type datetime __construct__");
    /*
    // TODO proper buffering
    ndobject year_as_int = year.ucast(make_dtype<int32_t>()).eval();
    ndobject month_as_int = month.ucast(make_dtype<int32_t>()).eval();
    ndobject day_as_int = day.ucast(make_dtype<int32_t>()).eval();
    ndobject result;

    ndobject_iter<1,3> iter(make_datetime_dtype(), result, year_as_int, month_as_int, day_as_int);
    if (!iter.empty()) {
        datetime::date_ymd ymd;
        do {
            ymd.year = *reinterpret_cast<const int32_t *>(iter.data<1>());
            ymd.month = *reinterpret_cast<const int32_t *>(iter.data<2>());
            ymd.day = *reinterpret_cast<const int32_t *>(iter.data<3>());
            if (!datetime::is_valid_ymd(ymd)) {
                stringstream ss;
                ss << "invalid year/month/day " << ymd.year << "/" << ymd.month << "/" << ymd.day;
                throw runtime_error(ss.str());
            }
            *reinterpret_cast<int32_t *>(iter.data<0>()) = datetime::ymd_to_days(ymd);
        } while (iter.next());
    }

    return result;
    */
}

static pair<string, gfunc::callable> datetime_dtype_functions[] = {
    pair<string, gfunc::callable>("now", gfunc::make_callable(&function_dtype_now, "self")),
    pair<string, gfunc::callable>("__construct__", gfunc::make_callable(&function_dtype_construct, "self", "year", "month", "day"))
};

void datetime_dtype::get_dynamic_dtype_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const
{
    *out_functions = datetime_dtype_functions;
    *out_count = sizeof(datetime_dtype_functions) / sizeof(datetime_dtype_functions[0]);
}

///////// properties on the ndobject

static ndobject property_ndo_get_year(const ndobject& n) {
    return n.replace_udtype(make_property_dtype(n.get_udtype(), "year"));
}

static ndobject property_ndo_get_month(const ndobject& n) {
    return n.replace_udtype(make_property_dtype(n.get_udtype(), "month"));
}

static ndobject property_ndo_get_day(const ndobject& n) {
    return n.replace_udtype(make_property_dtype(n.get_udtype(), "day"));
}

static pair<string, gfunc::callable> date_ndobject_properties[] = {
    pair<string, gfunc::callable>("year", gfunc::make_callable(&property_ndo_get_year, "self")),
    pair<string, gfunc::callable>("month", gfunc::make_callable(&property_ndo_get_month, "self")),
    pair<string, gfunc::callable>("day", gfunc::make_callable(&property_ndo_get_day, "self"))
};

void datetime_dtype::get_dynamic_ndobject_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = date_ndobject_properties;
    *out_count = sizeof(date_ndobject_properties) / sizeof(date_ndobject_properties[0]);
}

///////// functions on the ndobject

static ndobject function_ndo_to_struct(const ndobject& n) {
    return n.replace_udtype(make_property_dtype(n.get_udtype(), "struct"));
}

static ndobject function_ndo_strftime(const ndobject& n, const std::string& format) {
    // TODO: Allow 'format' itself to be an array, with broadcasting, etc.
    if (format.empty()) {
        throw runtime_error("format string for strftime should not be empty");
    }
    return n.replace_udtype(make_unary_expr_dtype(make_string_dtype(), n.get_udtype(),
                    make_strftime_kernelgen(format)));
}

static pair<string, gfunc::callable> date_ndobject_functions[] = {
    pair<string, gfunc::callable>("to_struct", gfunc::make_callable(&function_ndo_to_struct, "self")),
    pair<string, gfunc::callable>("strftime", gfunc::make_callable(&function_ndo_strftime, "self", "format")),
};

void datetime_dtype::get_dynamic_ndobject_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const
{
    *out_functions = date_ndobject_functions;
    *out_count = sizeof(date_ndobject_functions) / sizeof(date_ndobject_functions[0]);
}

///////// property accessor kernels (used by property_dtype)

namespace {
    void get_property_kernel_struct_single(char *dst, const char *src,
                    kernel_data_prefix *DYND_UNUSED(extra))
    {
        throw runtime_error("TODO: get_property_kernel_struct_single");
    }

    void set_property_kernel_struct_single(char *dst, const char *src,
                    kernel_data_prefix *DYND_UNUSED(extra))
    {
        throw runtime_error("TODO: set_property_kernel_struct_single");
    }

    void get_property_kernel_year_single(char *dst, const char *src,
                    kernel_data_prefix *DYND_UNUSED(extra))
    {
        throw runtime_error("TODO: get_property_kernel_year_single");
    }

    void get_property_kernel_month_single(char *dst, const char *src,
                    kernel_data_prefix *DYND_UNUSED(extra))
    {
        throw runtime_error("TODO: get_property_kernel_month_single");
    }

    void get_property_kernel_day_single(char *dst, const char *src,
                    kernel_data_prefix *DYND_UNUSED(extra))
    {
        throw runtime_error("TODO: get_property_kernel_day_single");
    }

    void get_property_kernel_hour_single(char *dst, const char *src,
                    kernel_data_prefix *DYND_UNUSED(extra))
    {
        throw runtime_error("TODO: get_property_kernel_hour_single");
    }

    void get_property_kernel_minute_single(char *dst, const char *src,
                    kernel_data_prefix *DYND_UNUSED(extra))
    {
        throw runtime_error("TODO: get_property_kernel_minute_single");
    }

    void get_property_kernel_second_single(char *dst, const char *src,
                    kernel_data_prefix *DYND_UNUSED(extra))
    {
        throw runtime_error("TODO: get_property_kernel_second_single");
    }

    void get_property_kernel_msecond_single(char *dst, const char *src,
                    kernel_data_prefix *DYND_UNUSED(extra))
    {
        throw runtime_error("TODO: get_property_kernel_msecond_single");
    }

    void get_property_kernel_usecond_single(char *dst, const char *src,
                    kernel_data_prefix *DYND_UNUSED(extra))
    {
        throw runtime_error("TODO: get_property_kernel_usecond_single");
    }

    void get_property_kernel_nsecond_single(char *dst, const char *src,
                    kernel_data_prefix *DYND_UNUSED(extra))
    {
        throw runtime_error("TODO: get_property_kernel_nsecond_single");
    }
} // anonymous namespace

namespace {
    enum date_properties_t {
        datetimeprop_struct,
        datetimeprop_year,
        datetimeprop_month,
        datetimeprop_day,
        datetimeprop_hour,
        datetimeprop_minute,
        datetimeprop_second,
        datetimeprop_msecond,
        datetimeprop_usecond,
        datetimeprop_nsecond,
    };
}

size_t datetime_dtype::get_elwise_property_index(const std::string& property_name) const
{
    if (property_name == "struct") {
        // A read/write property for accessing a datetime as a struct
        return datetimeprop_struct;
    } else if (property_name == "year") {
        return datetimeprop_year;
    } else if (property_name == "month") {
        return datetimeprop_month;
    } else if (property_name == "day") {
        return datetimeprop_day;
    } else if (property_name == "hour") {
        return datetimeprop_hour;
    } else if (property_name == "minute") {
        return datetimeprop_minute;
    } else if (property_name == "second") {
        return datetimeprop_second;
    } else if (property_name == "msecond") {
        return datetimeprop_msecond;
    } else if (property_name == "usecond") {
        return datetimeprop_usecond;
    } else if (property_name == "nsecond") {
        return datetimeprop_nsecond;
    } else {
        stringstream ss;
        ss << "dynd type " << dtype(this, true) << " does not have a kernel for property " << property_name;
        throw runtime_error(ss.str());
    }
}

dtype datetime_dtype::get_elwise_property_dtype(size_t property_index,
            bool& out_readable, bool& out_writable) const
{
    switch (property_index) {
        case datetimeprop_struct:
            out_readable = true;
            out_writable = true;
            return get_default_struct_dtype();
        default:
            out_readable = true;
            out_writable = false;
            return make_dtype<int32_t>();
    }
}

size_t datetime_dtype::make_elwise_property_getter_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const char *DYND_UNUSED(dst_metadata),
                const char *DYND_UNUSED(src_metadata), size_t src_property_index,
                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx)) const
{
    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    kernel_data_prefix *e = out->get_at<kernel_data_prefix>(offset_out);
    switch (src_property_index) {
        case datetimeprop_struct:
            e->set_function<unary_single_operation_t>(&get_property_kernel_struct_single);
            return offset_out + sizeof(kernel_data_prefix);
        case datetimeprop_year:
            e->set_function<unary_single_operation_t>(&get_property_kernel_year_single);
            return offset_out + sizeof(kernel_data_prefix);
        case datetimeprop_month:
            e->set_function<unary_single_operation_t>(&get_property_kernel_month_single);
            return offset_out + sizeof(kernel_data_prefix);
        case datetimeprop_day:
            e->set_function<unary_single_operation_t>(&get_property_kernel_day_single);
            return offset_out + sizeof(kernel_data_prefix);
        case datetimeprop_hour:
            e->set_function<unary_single_operation_t>(&get_property_kernel_hour_single);
            return offset_out + sizeof(kernel_data_prefix);
        case datetimeprop_minute:
            e->set_function<unary_single_operation_t>(&get_property_kernel_minute_single);
            return offset_out + sizeof(kernel_data_prefix);
        case datetimeprop_second:
            e->set_function<unary_single_operation_t>(&get_property_kernel_second_single);
            return offset_out + sizeof(kernel_data_prefix);
        case datetimeprop_msecond:
            e->set_function<unary_single_operation_t>(&get_property_kernel_msecond_single);
            return offset_out + sizeof(kernel_data_prefix);
        case datetimeprop_usecond:
            e->set_function<unary_single_operation_t>(&get_property_kernel_usecond_single);
            return offset_out + sizeof(kernel_data_prefix);
        case datetimeprop_nsecond:
            e->set_function<unary_single_operation_t>(&get_property_kernel_nsecond_single);
            return offset_out + sizeof(kernel_data_prefix);
        default:
            stringstream ss;
            ss << "dynd date dtype given an invalid property index" << src_property_index;
            throw runtime_error(ss.str());
    }
}

size_t datetime_dtype::make_elwise_property_setter_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const char *DYND_UNUSED(dst_metadata), size_t dst_property_index,
                const char *DYND_UNUSED(src_metadata),
                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx)) const
{
    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    kernel_data_prefix *e = out->get_at<kernel_data_prefix>(offset_out);
    switch (dst_property_index) {
        case datetimeprop_struct:
            e->set_function<unary_single_operation_t>(&set_property_kernel_struct_single);
            return offset_out + sizeof(kernel_data_prefix);
        default:
            stringstream ss;
            ss << "dynd date dtype given an invalid property index" << dst_property_index;
            throw runtime_error(ss.str());
    }
}

