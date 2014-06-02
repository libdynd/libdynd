//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/time_assignment_kernels.hpp>
#include <dynd/func/make_callable.hpp>

using namespace std;
using namespace dynd;

time_type::time_type(datetime_tz_t timezone)
    : base_type(time_type_id, datetime_kind, 8,
                scalar_align_of<int64_t>::value, type_flag_scalar, 0, 0),
      m_timezone(timezone)
{
}

time_type::~time_type()
{
}

void time_type::set_time(const char *DYND_UNUSED(metadata), char *data,
                         assign_error_mode errmode, int32_t hour,
                         int32_t minute, int32_t second, int32_t tick) const
{
    if (errmode != assign_error_none &&
            !time_hmst::is_valid(hour, minute, second, tick)) {
        stringstream ss;
        ss << "invalid input time " << hour << ":" << minute << ":" << second << ", ticks: " << tick;
        throw runtime_error(ss.str());
    }

    *reinterpret_cast<int64_t *>(data) = time_hmst::to_ticks(hour, minute, second, tick);
}

void time_type::set_utf8_string(const char *DYND_UNUSED(metadata),
                char *data, assign_error_mode DYND_UNUSED(errmode), const std::string& utf8_str) const
{
    time_hmst hmst;
    // TODO: Use errmode to adjust strictness of the parsing
    // TODO: properly distinguish "time" and "option[time]" with respect to NA support
    if (utf8_str == "NA") {
        hmst.set_to_na();
    } else {
        hmst.set_from_str(utf8_str);
    }
    *reinterpret_cast<int64_t *>(data) = hmst.to_ticks();
}

time_hmst time_type::get_time(const char *DYND_UNUSED(metadata), const char *data) const
{
    time_hmst hmst;
    hmst.set_from_ticks(*reinterpret_cast<const int64_t *>(data));
    return hmst;
}

void time_type::print_data(std::ostream& o, const char *DYND_UNUSED(metadata), const char *data) const
{
    time_hmst hmst;
    hmst.set_from_ticks(*reinterpret_cast<const int64_t *>(data));
    string s = hmst.to_str();
    if (s.empty()) {
        o << "NA";
    } else {
        o << s;
    }
}

void time_type::print_type(std::ostream& o) const
{
    if (m_timezone == tz_abstract) {
        o << "time";
    } else {
        o << "time[tz='";
        switch (m_timezone) {
            case tz_utc:
                o << "UTC";
                break;
            default:
                o << "(invalid " << (int32_t)m_timezone << ")";
                break;
        }
        o << "']";
    }
}

bool time_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    if (dst_tp.extended() == this) {
        if (src_tp.extended() == this) {
            return true;
        } else if (src_tp.get_type_id() == time_type_id) {
            // There is only one possibility for the time type (TODO: timezones!)
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

bool time_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != time_type_id) {
        return false;
    } else {
        const time_type& r = static_cast<const time_type &>(rhs);
        // TODO: When "other" timezone data is supported, need to compare them too
        return m_timezone == r.m_timezone;
    }
}

size_t time_type::make_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_tp.extended()) {
        if (src_tp.get_type_id() == time_type_id) {
            return make_pod_typed_data_assignment_kernel(out, offset_out,
                            get_data_size(), get_data_alignment(), kernreq);
        } else if (src_tp.get_kind() == string_kind) {
            // Assignment from strings
            return make_string_to_time_assignment_kernel(out, offset_out,
                            dst_tp,
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
            return make_time_to_string_assignment_kernel(out, offset_out,
                            dst_tp, dst_metadata,
                            src_tp,
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

size_t time_type::make_comparison_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& src0_tp, const char *src0_metadata,
                const ndt::type& src1_tp, const char *src1_metadata,
                comparison_type_t comptype,
                const eval::eval_context *ectx) const
{
    if (this == src0_tp.extended()) {
        if (*this == *src1_tp.extended()) {
            return make_builtin_type_comparison_kernel(out, offset_out,
                            int64_type_id, int64_type_id, comptype);
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

void time_type::get_dynamic_type_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = NULL;
    *out_count = 0;
}

///////// functions on the type

void time_type::get_dynamic_type_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const
{
    *out_functions = NULL;
    *out_count = 0;
}

///////// properties on the nd::array

static nd::array property_ndo_get_hour(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "hour"));
}

static nd::array property_ndo_get_minute(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "minute"));
}

static nd::array property_ndo_get_second(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "second"));
}

static nd::array property_ndo_get_microsecond(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "microsecond"));
}

static nd::array property_ndo_get_tick(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "tick"));
}

void time_type::get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    static pair<string, gfunc::callable> time_array_properties[] = {
        pair<string, gfunc::callable>(
            "hour", gfunc::make_callable(&property_ndo_get_hour, "self")),
        pair<string, gfunc::callable>(
            "minute", gfunc::make_callable(&property_ndo_get_minute, "self")),
        pair<string, gfunc::callable>(
            "second", gfunc::make_callable(&property_ndo_get_second, "self")),
        pair<string, gfunc::callable>(
            "microsecond",
            gfunc::make_callable(&property_ndo_get_microsecond, "self")),
        pair<string, gfunc::callable>(
            "tick", gfunc::make_callable(&property_ndo_get_tick, "self"))};

    *out_properties = time_array_properties;
    *out_count = sizeof(time_array_properties) / sizeof(time_array_properties[0]);
}

///////// functions on the nd::array

static nd::array function_ndo_to_struct(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "struct"));
}

void time_type::get_dynamic_array_functions(
                const std::pair<std::string, gfunc::callable> **out_functions,
                size_t *out_count) const
{
    static pair<string, gfunc::callable> time_array_functions[] = {
        pair<string, gfunc::callable>(
            "to_struct",
            gfunc::make_callable(&function_ndo_to_struct, "self")), };

    *out_functions = time_array_functions;
    *out_count = sizeof(time_array_functions) / sizeof(time_array_functions[0]);
}

///////// property accessor kernels (used by property_type)

namespace {
    void get_property_kernel_hour_single(char *dst, const char *src,
                    ckernel_prefix *DYND_UNUSED(extra))
    {
        int64_t ticks = *reinterpret_cast<const int64_t *>(src);
        *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>(ticks / DYND_TICKS_PER_HOUR);
    }

    void get_property_kernel_minute_single(char *dst, const char *src,
                    ckernel_prefix *DYND_UNUSED(extra))
    {
        int64_t ticks = *reinterpret_cast<const int64_t *>(src);
        *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>((ticks / DYND_TICKS_PER_MINUTE) % 60);
    }

    void get_property_kernel_second_single(char *dst, const char *src,
                    ckernel_prefix *DYND_UNUSED(extra))
    {
        int64_t ticks = *reinterpret_cast<const int64_t *>(src);
        *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>((ticks / DYND_TICKS_PER_SECOND) % 60);
    }

    void get_property_kernel_microsecond_single(char *dst, const char *src,
                    ckernel_prefix *DYND_UNUSED(extra))
    {
        int64_t ticks = *reinterpret_cast<const int64_t *>(src);
        *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>((ticks / DYND_TICKS_PER_MICROSECOND) % 1000000);
    }

    void get_property_kernel_tick_single(char *dst, const char *src,
                    ckernel_prefix *DYND_UNUSED(extra))
    {
        int64_t ticks = *reinterpret_cast<const int64_t *>(src);
        *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>(ticks % 10000000);
    }

    void get_property_kernel_struct_single(char *dst, const char *src,
                    ckernel_prefix *DYND_UNUSED(extra))
    {
        time_hmst *dst_struct = reinterpret_cast<time_hmst *>(dst);
        dst_struct->set_from_ticks(*reinterpret_cast<const int64_t *>(src));
    }

    void set_property_kernel_struct_single(char *dst, const char *src,
                    ckernel_prefix *DYND_UNUSED(extra))
    {
        const time_hmst *src_struct = reinterpret_cast<const time_hmst *>(src);
        *reinterpret_cast<int64_t *>(dst) = src_struct->to_ticks();
    }
} // anonymous namespace

namespace {
    enum time_properties_t {
        timeprop_hour,
        timeprop_minute,
        timeprop_second,
        timeprop_microsecond,
        timeprop_tick,
        timeprop_struct
    };
}

size_t time_type::get_elwise_property_index(const std::string& property_name) const
{
    if (property_name == "hour") {
        return timeprop_hour;
    } else if (property_name == "minute") {
        return timeprop_minute;
    } else if (property_name == "second") {
        return timeprop_second;
    } else if (property_name == "microsecond") {
        return timeprop_microsecond;
    } else if (property_name == "tick") {
        return timeprop_tick;
    } else if (property_name == "struct") {
        // A read/write property for accessing a time as a struct
        return timeprop_struct;
    } else {
        stringstream ss;
        ss << "dynd time type does not have a kernel for property " << property_name;
        throw runtime_error(ss.str());
    }
}

ndt::type time_type::get_elwise_property_type(size_t property_index,
            bool& out_readable, bool& out_writable) const
{
    switch (property_index) {
        case timeprop_hour:
        case timeprop_minute:
        case timeprop_second:
        case timeprop_microsecond:
        case timeprop_tick:
            out_readable = true;
            out_writable = false;
            return ndt::make_type<int32_t>();
        case timeprop_struct:
            out_readable = true;
            out_writable = true;
            return time_hmst::type();
        default:
            out_readable = false;
            out_writable = false;
            return ndt::make_type<void>();
    }
}

size_t time_type::make_elwise_property_getter_kernel(
                ckernel_builder *out, size_t offset_out,
                const char *DYND_UNUSED(dst_metadata),
                const char *DYND_UNUSED(src_metadata), size_t src_property_index,
                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx)) const
{
    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    ckernel_prefix *e = out->get_at<ckernel_prefix>(offset_out);
    switch (src_property_index) {
        case timeprop_hour:
            e->set_function<unary_single_operation_t>(&get_property_kernel_hour_single);
            return offset_out + sizeof(ckernel_prefix);
        case timeprop_minute:
            e->set_function<unary_single_operation_t>(&get_property_kernel_minute_single);
            return offset_out + sizeof(ckernel_prefix);
        case timeprop_second:
            e->set_function<unary_single_operation_t>(&get_property_kernel_second_single);
            return offset_out + sizeof(ckernel_prefix);
        case timeprop_microsecond:
            e->set_function<unary_single_operation_t>(&get_property_kernel_microsecond_single);
            return offset_out + sizeof(ckernel_prefix);
        case timeprop_tick:
            e->set_function<unary_single_operation_t>(&get_property_kernel_tick_single);
            return offset_out + sizeof(ckernel_prefix);
        case timeprop_struct:
            e->set_function<unary_single_operation_t>(&get_property_kernel_struct_single);
            return offset_out + sizeof(ckernel_prefix);
        default:
            stringstream ss;
            ss << "dynd time type given an invalid property index" << src_property_index;
            throw runtime_error(ss.str());
    }
}

size_t time_type::make_elwise_property_setter_kernel(
                ckernel_builder *out, size_t offset_out,
                const char *DYND_UNUSED(dst_metadata), size_t dst_property_index,
                const char *DYND_UNUSED(src_metadata),
                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx)) const
{
    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    ckernel_prefix *e = out->get_at<ckernel_prefix>(offset_out);
    switch (dst_property_index) {
        case timeprop_struct:
            e->set_function<unary_single_operation_t>(&set_property_kernel_struct_single);
            return offset_out + sizeof(ckernel_prefix);
        default:
            stringstream ss;
            ss << "dynd time type given an invalid property index" << dst_property_index;
            throw runtime_error(ss.str());
    }
}

const ndt::type& ndt::make_time()
{
    // Static instance of the type, which has a reference count > 0 for the
    // lifetime of the program. This static construction is inside a
    // function to ensure correct creation order during startup.
    static time_type tt(tz_abstract);
    static const ndt::type static_instance(&tt, true);
    return static_instance;
}
