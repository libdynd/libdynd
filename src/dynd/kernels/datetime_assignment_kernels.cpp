//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/datetime_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <datetime_strings.h>

using namespace std;
using namespace dynd;

static datetime::datetime_unit_t dynd_unit_to_datetime_unit(datetime_unit_t unit) {
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


/////////////////////////////////////////
// string to datetime assignment

namespace {
    struct string_to_datetime_kernel_extra {
        typedef string_to_datetime_kernel_extra extra_type;

        ckernel_data_prefix base;
        const datetime_type *dst_datetime_dt;
        const base_string_type *src_string_dt;
        const char *src_metadata;
        assign_error_mode errmode;
        datetime::datetime_unit_t unit;
        datetime::datetime_conversion_rule_t casting;

        static void single(char *dst, const char *src, ckernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const string& s = e->src_string_dt->get_utf8_string(e->src_metadata, src, e->errmode);
            *reinterpret_cast<int64_t *>(dst) = datetime::parse_iso_8601_datetime(s, e->unit,
                                    e->dst_datetime_dt->get_timezone() == tz_abstract,
                                    e->casting);
        }

        static void destruct(ckernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            base_type_xdecref(e->dst_datetime_dt);
            base_type_xdecref(e->src_string_dt);
        }
    };
} // anonymous namespace

size_t dynd::make_string_to_datetime_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const ndt::type& dst_datetime_dt, const char *DYND_UNUSED(dst_metadata),
                const ndt::type& src_string_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    if (src_string_dt.get_kind() != string_kind) {
        stringstream ss;
        ss << "make_string_to_datetime_assignment_kernel: source type " << src_string_dt << " is not a string type";
        throw runtime_error(ss.str());
    }

    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    out->ensure_capacity(offset_out + sizeof(string_to_datetime_kernel_extra));
    string_to_datetime_kernel_extra *e = out->get_at<string_to_datetime_kernel_extra>(offset_out);
    e->base.set_function<unary_single_operation_t>(&string_to_datetime_kernel_extra::single);
    e->base.destructor = &string_to_datetime_kernel_extra::destruct;
    // The kernel data owns a reference to this type
    e->dst_datetime_dt = static_cast<const datetime_type *>(ndt::type(dst_datetime_dt).release());
    // The kernel data owns a reference to this type
    e->src_string_dt = static_cast<const base_string_type *>(ndt::type(src_string_dt).release());
    e->src_metadata = src_metadata;
    e->errmode = errmode;
    e->unit = dynd_unit_to_datetime_unit(e->dst_datetime_dt->get_unit());
    switch (errmode) {
        case assign_error_fractional:
        case assign_error_inexact:
            e->casting = datetime::datetime_conversion_strict;
            break;
        default:
            e->casting = datetime::datetime_conversion_relaxed;
    }
    return offset_out + sizeof(string_to_datetime_kernel_extra);
}

/////////////////////////////////////////
// datetime to string assignment

namespace {
    struct datetime_to_string_kernel_extra {
        typedef datetime_to_string_kernel_extra extra_type;

        ckernel_data_prefix base;
        const base_string_type *dst_string_dt;
        const datetime_type *src_datetime_dt;
        const char *dst_metadata;
        assign_error_mode errmode;
        datetime::datetime_unit_t unit;

        static void single(char *dst, const char *src, ckernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            int64_t datetime = *reinterpret_cast<const int64_t *>(src);
            e->dst_string_dt->set_utf8_string(e->dst_metadata, dst, e->errmode,
                            datetime::make_iso_8601_datetime(datetime, e->unit,
                                            e->src_datetime_dt->get_timezone() == tz_abstract));
        }

        static void destruct(ckernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            base_type_xdecref(e->dst_string_dt);
            base_type_xdecref(e->src_datetime_dt);
        }
    };
} // anonymous namespace

size_t dynd::make_datetime_to_string_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const ndt::type& dst_string_dt, const char *dst_metadata,
                const ndt::type& src_datetime_dt, const char *DYND_UNUSED(src_metadata),
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    if (dst_string_dt.get_kind() != string_kind) {
        stringstream ss;
        ss << "get_datetime_to_string_assignment_kernel: dest type " << dst_string_dt << " is not a string type";
        throw runtime_error(ss.str());
    }

    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    out->ensure_capacity(offset_out + sizeof(datetime_to_string_kernel_extra));
    datetime_to_string_kernel_extra *e = out->get_at<datetime_to_string_kernel_extra>(offset_out);
    e->base.set_function<unary_single_operation_t>(&datetime_to_string_kernel_extra::single);
    e->base.destructor = &datetime_to_string_kernel_extra::destruct;
    // The kernel data owns a reference to this type
    e->dst_string_dt = static_cast<const base_string_type *>(ndt::type(dst_string_dt).release());
    // The kernel data owns a reference to this type
    e->src_datetime_dt = static_cast<const datetime_type *>(ndt::type(src_datetime_dt).release());
    e->dst_metadata = dst_metadata;
    e->unit = dynd_unit_to_datetime_unit(e->src_datetime_dt->get_unit());
    e->errmode = errmode;
    return offset_out + sizeof(datetime_to_string_kernel_extra);
}

