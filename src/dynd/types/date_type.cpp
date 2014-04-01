//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <time.h>

#include <cerrno>
#include <algorithm>

#include <dynd/types/date_type.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/unary_expr_type.hpp>
#include <dynd/kernels/date_assignment_kernels.hpp>
#include <dynd/kernels/date_expr_kernels.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/gfunc/make_callable.hpp>
#include <dynd/array_iter.hpp>

#include <datetime_strings.h>
#include <datetime_localtime.h>

using namespace std;
using namespace dynd;

date_type::date_type()
    : base_type(date_type_id, datetime_kind, 4, scalar_align_of<int32_t>::value,
                type_flag_scalar, 0, 0)
{
}

date_type::~date_type()
{
}

void date_type::set_ymd(const char *DYND_UNUSED(metadata), char *data,
                assign_error_mode errmode, int32_t year, int32_t month, int32_t day) const
{
    if (errmode != assign_error_none && !date_ymd::is_valid(year, month, day)) {
        stringstream ss;
        ss << "invalid input year/month/day " << year << "/" << month << "/" << day;
        throw runtime_error(ss.str());
    }

    *reinterpret_cast<int32_t *>(data) = date_ymd::to_days(year, month, day);
}

void date_type::set_utf8_string(const char *DYND_UNUSED(metadata),
                char *data, assign_error_mode DYND_UNUSED(errmode), const std::string& utf8_str) const
{
    date_ymd ymd;
    // TODO: Use errmode to adjust strictness of the parsing
    // TODO: properly distinguish "date" and "option[date]" with respect to NA support
    if (utf8_str == "NA") {
        ymd.set_to_na();
    } else {
        ymd.set_from_str(utf8_str);
    }
    *reinterpret_cast<int32_t *>(data) = ymd.to_days();
}


date_ymd date_type::get_ymd(const char *DYND_UNUSED(metadata), const char *data) const
{
    date_ymd ymd;
    ymd.set_from_days(*reinterpret_cast<const int32_t *>(data));
    return ymd;
}

void date_type::print_data(std::ostream& o, const char *DYND_UNUSED(metadata), const char *data) const
{
    date_ymd ymd;
    ymd.set_from_days(*reinterpret_cast<const int32_t *>(data));
    o << ymd.to_str();
}

void date_type::print_type(std::ostream& o) const
{
    o << "date";
}

bool date_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    if (dst_tp.extended() == this) {
        if (src_tp.extended() == this) {
            return true;
        } else if (src_tp.get_type_id() == date_type_id) {
            // There is only one possibility for the date type (TODO: timezones!)
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

bool date_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != date_type_id) {
        return false;
    } else {
        // There is only one possibility for the date type (TODO: timezones!)
        return true;
    }
}

size_t date_type::make_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_tp.extended()) {
        if (src_tp.get_type_id() == date_type_id) {
            return make_pod_typed_data_assignment_kernel(out, offset_out,
                            get_data_size(), get_data_alignment(), kernreq);
        } else if (src_tp.get_kind() == string_kind) {
            // Assignment from strings
            return make_string_to_date_assignment_kernel(out, offset_out,
                            src_tp, src_metadata,
                            kernreq, errmode, ectx);
        } else if (src_tp.get_kind() == struct_kind) {
            // Convert to struct using the "struct" property
            return ::make_assignment_kernel(out, offset_out,
                ndt::make_property(dst_tp, "struct"), dst_metadata,
                src_tp, src_metadata,
                kernreq, errmode, ectx);
        } else if (!src_tp.is_builtin()) {
            return src_tp.extended()->make_assignment_kernel(out, offset_out,
                            dst_tp, dst_metadata,
                            src_tp, src_metadata,
                            kernreq, errmode, ectx);
        }
    } else {
        if (dst_tp.get_kind() == string_kind) {
            // Assignment to strings
            return make_date_to_string_assignment_kernel(out, offset_out,
                            dst_tp, dst_metadata,
                            kernreq, errmode, ectx);
        } else if (dst_tp.get_kind() == struct_kind) {
            // Convert to struct using the "struct" property
            return ::make_assignment_kernel(out, offset_out,
                dst_tp, dst_metadata,
                ndt::make_property(src_tp, "struct"), src_metadata,
                kernreq, errmode, ectx);
        }
        // TODO
    }

    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw dynd::type_error(ss.str());
}

size_t date_type::make_comparison_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& src0_tp, const char *src0_metadata,
                const ndt::type& src1_tp, const char *src1_metadata,
                comparison_type_t comptype,
                const eval::eval_context *ectx) const
{
    if (this == src0_tp.extended()) {
        if (*this == *src1_tp.extended()) {
            return make_builtin_type_comparison_kernel(out, offset_out,
                            int32_type_id, int32_type_id, comptype);
        } else if (!src1_tp.is_builtin()) {
            return src1_tp.extended()->make_comparison_kernel(out, offset_out,
                            src0_tp, src0_metadata,
                            src1_tp, src1_metadata,
                            comptype, ectx);
        }
    }

    throw not_comparable_error(src0_tp, src1_tp, comptype);
}

///////// properties on the type

//static pair<string, gfunc::callable> date_type_properties[] = {
//};

void date_type::get_dynamic_type_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = NULL; //date_type_properties;
    *out_count = 0; //sizeof(date_type_properties) / sizeof(date_type_properties[0]);
}

///////// functions on the type

static nd::array function_type_today(const ndt::type& dt) {
    date_ymd ymd = date_ymd::get_current_local_date();
    nd::array result = nd::empty(dt);
    *reinterpret_cast<int32_t *>(result.get_readwrite_originptr()) = ymd.to_days();
    // Make the result immutable (we own the only reference to the data at this point)
    result.flag_as_immutable();
    return result;
}

static nd::array function_type_construct(const ndt::type& DYND_UNUSED(dt),
                const nd::array& year, const nd::array& month, const nd::array& day)
{
    // TODO proper buffering
    nd::array year_as_int = year.ucast(ndt::make_type<int32_t>()).eval();
    nd::array month_as_int = month.ucast(ndt::make_type<int32_t>()).eval();
    nd::array day_as_int = day.ucast(ndt::make_type<int32_t>()).eval();
    nd::array result;

    array_iter<1,3> iter(ndt::make_date(), result, year_as_int, month_as_int, day_as_int);
    if (!iter.empty()) {
        date_ymd ymd;
        do {
            ymd.year = *reinterpret_cast<const int32_t *>(iter.data<1>());
            ymd.month = *reinterpret_cast<const int32_t *>(iter.data<2>());
            ymd.day = *reinterpret_cast<const int32_t *>(iter.data<3>());
            if (!ymd.is_valid()) {
                stringstream ss;
                ss << "invalid year/month/day " << ymd.year << "/" << ymd.month << "/" << ymd.day;
                throw runtime_error(ss.str());
            }
            *reinterpret_cast<int32_t *>(iter.data<0>()) = ymd.to_days();
        } while (iter.next());
    }

    return result;
}

static pair<string, gfunc::callable> date_type_functions[] = {
    pair<string, gfunc::callable>("today", gfunc::make_callable(&function_type_today, "self")),
    pair<string, gfunc::callable>("__construct__", gfunc::make_callable(&function_type_construct, "self", "year", "month", "day"))
};

void date_type::get_dynamic_type_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const
{
    *out_functions = date_type_functions;
    *out_count = sizeof(date_type_functions) / sizeof(date_type_functions[0]);
}

///////// properties on the nd::array

static nd::array property_ndo_get_year(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "year"));
}

static nd::array property_ndo_get_month(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "month"));
}

static nd::array property_ndo_get_day(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "day"));
}

static pair<string, gfunc::callable> date_array_properties[] = {
    pair<string, gfunc::callable>("year", gfunc::make_callable(&property_ndo_get_year, "self")),
    pair<string, gfunc::callable>("month", gfunc::make_callable(&property_ndo_get_month, "self")),
    pair<string, gfunc::callable>("day", gfunc::make_callable(&property_ndo_get_day, "self"))
};

void date_type::get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = date_array_properties;
    *out_count = sizeof(date_array_properties) / sizeof(date_array_properties[0]);
}

///////// functions on the nd::array

static nd::array function_ndo_to_struct(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "struct"));
}

static nd::array function_ndo_strftime(const nd::array& n, const std::string& format) {
    // TODO: Allow 'format' itself to be an array, with broadcasting, etc.
    if (format.empty()) {
        throw runtime_error("format string for strftime should not be empty");
    }
    return n.replace_dtype(ndt::make_unary_expr(ndt::make_string(), n.get_dtype(),
                    make_strftime_kernelgen(format)));
}

static nd::array function_ndo_weekday(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "weekday"));
}

static nd::array function_ndo_replace(const nd::array& n, int32_t year, int32_t month, int32_t day) {
    // TODO: Allow 'year', 'month', and 'day' to be arrays, with broadcasting, etc.
    if (year == numeric_limits<int32_t>::max() && month == numeric_limits<int32_t>::max() &&
                    day == numeric_limits<int32_t>::max()) {
        throw std::runtime_error("no parameters provided to date.replace, should provide at least one");
    }
    return n.replace_dtype(ndt::make_unary_expr(ndt::make_date(), n.get_dtype(),
                    make_replace_kernelgen(year, month, day)));
}

static pair<string, gfunc::callable> date_array_functions[] = {
    pair<string, gfunc::callable>("to_struct", gfunc::make_callable(&function_ndo_to_struct, "self")),
    pair<string, gfunc::callable>("strftime", gfunc::make_callable(&function_ndo_strftime, "self", "format")),
    pair<string, gfunc::callable>("weekday", gfunc::make_callable(&function_ndo_weekday, "self")),
    pair<string, gfunc::callable>("replace", gfunc::make_callable_with_default(&function_ndo_replace, "self", "year", "month", "day",
                    numeric_limits<int32_t>::max(), numeric_limits<int32_t>::max(), numeric_limits<int32_t>::max()))
};

void date_type::get_dynamic_array_functions(
                const std::pair<std::string, gfunc::callable> **out_functions,
                size_t *out_count) const
{
    *out_functions = date_array_functions;
    *out_count = sizeof(date_array_functions) / sizeof(date_array_functions[0]);
}

///////// property accessor kernels (used by property_type)

namespace {
    void get_property_kernel_year_single(char *dst, const char *src,
                    ckernel_prefix *DYND_UNUSED(extra))
    {
        date_ymd ymd;
        ymd.set_from_days(*reinterpret_cast<const int32_t *>(src));
        *reinterpret_cast<int32_t *>(dst) = ymd.year;
    }

    void get_property_kernel_month_single(char *dst, const char *src,
                    ckernel_prefix *DYND_UNUSED(extra))
    {
        date_ymd ymd;
        ymd.set_from_days(*reinterpret_cast<const int32_t *>(src));
        *reinterpret_cast<int32_t *>(dst) = ymd.month;
    }

    void get_property_kernel_day_single(char *dst, const char *src,
                    ckernel_prefix *DYND_UNUSED(extra))
    {
        date_ymd ymd;
        ymd.set_from_days(*reinterpret_cast<const int32_t *>(src));
        *reinterpret_cast<int32_t *>(dst) = ymd.day;
    }

    void get_property_kernel_weekday_single(char *dst, const char *src,
                    ckernel_prefix *DYND_UNUSED(extra))
    {
        int32_t days = *reinterpret_cast<const int32_t *>(src);
        // 1970-01-05 is Monday
        int weekday = (int)((days - 4) % 7);
        if (weekday < 0) {
            weekday += 7;
        }
        *reinterpret_cast<int32_t *>(dst) = weekday;
    }

    void get_property_kernel_days_after_1970_int64_single(char *dst, const char *src,
                    ckernel_prefix *DYND_UNUSED(extra))
    {
        int32_t days = *reinterpret_cast<const int32_t *>(src);
        if (days == DYND_DATE_NA) {
            *reinterpret_cast<int64_t *>(dst) = numeric_limits<int64_t>::min();
        } else {
            *reinterpret_cast<int64_t *>(dst) = days;
        }
    }

    void set_property_kernel_days_after_1970_int64_single(char *dst, const char *src,
                    ckernel_prefix *DYND_UNUSED(extra))
    {
        int64_t days = *reinterpret_cast<const int64_t *>(src);
        if (days == numeric_limits<int64_t>::min()) {
            *reinterpret_cast<int32_t *>(dst) = DYND_DATE_NA;
        } else {
            *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>(days);
        }
    }

    void get_property_kernel_struct_single(char *dst, const char *src,
                    ckernel_prefix *DYND_UNUSED(extra))
    {
        date_ymd *dst_struct = reinterpret_cast<date_ymd *>(dst);
        dst_struct->set_from_days(*reinterpret_cast<const int32_t *>(src));
    }

    void set_property_kernel_struct_single(char *dst, const char *src,
                    ckernel_prefix *DYND_UNUSED(extra))
    {
        const date_ymd *src_struct = reinterpret_cast<const date_ymd *>(src);
        *reinterpret_cast<int32_t *>(dst) = src_struct->to_days();
    }
} // anonymous namespace

namespace {
    enum date_properties_t {
        dateprop_year,
        dateprop_month,
        dateprop_day,
        dateprop_weekday,
        dateprop_days_after_1970_int64,
        dateprop_struct
    };
}

size_t date_type::get_elwise_property_index(const std::string& property_name) const
{
    if (property_name == "year") {
        return dateprop_year;
    } else if (property_name == "month") {
        return dateprop_month;
    } else if (property_name == "day") {
        return dateprop_day;
    } else if (property_name == "weekday") {
        return dateprop_weekday;
    } else if (property_name == "days_after_1970_int64") {
        // A read/write property for NumPy datetime64[D] compatibility
        return dateprop_days_after_1970_int64;
    } else if (property_name == "struct") {
        // A read/write property for accessing a date as a struct
        return dateprop_struct;
    } else {
        stringstream ss;
        ss << "dynd date type does not have a kernel for property " << property_name;
        throw runtime_error(ss.str());
    }
}

ndt::type date_type::get_elwise_property_type(size_t property_index,
            bool& out_readable, bool& out_writable) const
{
    switch (property_index) {
        case dateprop_year:
        case dateprop_month:
        case dateprop_day:
        case dateprop_weekday:
            out_readable = true;
            out_writable = false;
            return ndt::make_type<int32_t>();
        case dateprop_days_after_1970_int64:
            out_readable = true;
            out_writable = true;
            return ndt::make_type<int64_t>();
        case dateprop_struct:
            out_readable = true;
            out_writable = true;
            return date_ymd::type();
        default:
            out_readable = false;
            out_writable = false;
            return ndt::make_type<void>();
    }
}

size_t date_type::make_elwise_property_getter_kernel(
                ckernel_builder *out, size_t offset_out,
                const char *DYND_UNUSED(dst_metadata),
                const char *DYND_UNUSED(src_metadata), size_t src_property_index,
                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx)) const
{
    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    ckernel_prefix *e = out->get_at<ckernel_prefix>(offset_out);
    switch (src_property_index) {
        case dateprop_year:
            e->set_function<unary_single_operation_t>(&get_property_kernel_year_single);
            return offset_out + sizeof(ckernel_prefix);
        case dateprop_month:
            e->set_function<unary_single_operation_t>(&get_property_kernel_month_single);
            return offset_out + sizeof(ckernel_prefix);
        case dateprop_day:
            e->set_function<unary_single_operation_t>(&get_property_kernel_day_single);
            return offset_out + sizeof(ckernel_prefix);
        case dateprop_weekday:
            e->set_function<unary_single_operation_t>(&get_property_kernel_weekday_single);
            return offset_out + sizeof(ckernel_prefix);
        case dateprop_days_after_1970_int64:
            e->set_function<unary_single_operation_t>(&get_property_kernel_days_after_1970_int64_single);
            return offset_out + sizeof(ckernel_prefix);
        case dateprop_struct:
            e->set_function<unary_single_operation_t>(&get_property_kernel_struct_single);
            return offset_out + sizeof(ckernel_prefix);
        default:
            stringstream ss;
            ss << "dynd date type given an invalid property index" << src_property_index;
            throw runtime_error(ss.str());
    }
}

size_t date_type::make_elwise_property_setter_kernel(
                ckernel_builder *out, size_t offset_out,
                const char *DYND_UNUSED(dst_metadata), size_t dst_property_index,
                const char *DYND_UNUSED(src_metadata),
                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx)) const
{
    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    ckernel_prefix *e = out->get_at<ckernel_prefix>(offset_out);
    switch (dst_property_index) {
        case dateprop_days_after_1970_int64:
            e->set_function<unary_single_operation_t>(&set_property_kernel_days_after_1970_int64_single);
            return offset_out + sizeof(ckernel_prefix);
        case dateprop_struct:
            e->set_function<unary_single_operation_t>(&set_property_kernel_struct_single);
            return offset_out + sizeof(ckernel_prefix);
        default:
            stringstream ss;
            ss << "dynd date type given an invalid property index" << dst_property_index;
            throw runtime_error(ss.str());
    }
}

