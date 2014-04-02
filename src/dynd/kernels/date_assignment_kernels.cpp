//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/date_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <datetime_strings.h>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// string to date assignment

namespace {
    struct string_to_date_kernel_extra {
        typedef string_to_date_kernel_extra extra_type;

        ckernel_prefix base;
        const base_string_type *src_string_dt;
        const char *src_metadata;
        assign_error_mode errmode;
        datetime::datetime_conversion_rule_t casting;

        static void single(char *dst, const char *src, ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const string& s = e->src_string_dt->get_utf8_string(e->src_metadata, src, e->errmode);
            date_ymd ymd;
            // TODO: properly distinguish "date" and "option[date]" with respect to NA support
            if (s == "NA") {
                ymd.set_to_na();
            } else {
                ymd.set_from_str(s);
            }
            *reinterpret_cast<int32_t *>(dst) = ymd.to_days();
        }

        static void destruct(ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            base_type_xdecref(e->src_string_dt);
        }
    };
} // anonymous namespace

size_t dynd::make_string_to_date_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& src_string_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    if (src_string_dt.get_kind() != string_kind) {
        stringstream ss;
        ss << "make_string_to_date_assignment_kernel: source type " << src_string_dt << " is not a string type";
        throw runtime_error(ss.str());
    }

    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    out->ensure_capacity(offset_out + sizeof(string_to_date_kernel_extra));
    string_to_date_kernel_extra *e = out->get_at<string_to_date_kernel_extra>(offset_out);
    e->base.set_function<unary_single_operation_t>(&string_to_date_kernel_extra::single);
    e->base.destructor = &string_to_date_kernel_extra::destruct;
    // The kernel data owns a reference to this type
    e->src_string_dt = static_cast<const base_string_type *>(ndt::type(src_string_dt).release());
    e->src_metadata = src_metadata;
    e->errmode = errmode;
    switch (errmode) {
        case assign_error_fractional:
        case assign_error_inexact:
            e->casting = datetime::datetime_conversion_strict;
            break;
        default:
            e->casting = datetime::datetime_conversion_relaxed;
    }
    return offset_out + sizeof(string_to_date_kernel_extra);
}

/////////////////////////////////////////
// date to string assignment

namespace {
    struct date_to_string_kernel_extra {
        typedef date_to_string_kernel_extra extra_type;

        ckernel_prefix base;
        const base_string_type *dst_string_dt;
        const char *dst_metadata;
        assign_error_mode errmode;

        static void single(char *dst, const char *src, ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            date_ymd ymd;
            ymd.set_from_days(*reinterpret_cast<const int32_t *>(src));
            e->dst_string_dt->set_utf8_string(e->dst_metadata, dst, e->errmode, ymd.to_str());
        }

        static void destruct(ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            base_type_xdecref(e->dst_string_dt);
        }
    };
} // anonymous namespace

size_t dynd::make_date_to_string_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_string_dt, const char *dst_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    if (dst_string_dt.get_kind() != string_kind) {
        stringstream ss;
        ss << "get_date_to_string_assignment_kernel: dest type " << dst_string_dt << " is not a string type";
        throw runtime_error(ss.str());
    }

    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    out->ensure_capacity(offset_out + sizeof(date_to_string_kernel_extra));
    date_to_string_kernel_extra *e = out->get_at<date_to_string_kernel_extra>(offset_out);
    e->base.set_function<unary_single_operation_t>(&date_to_string_kernel_extra::single);
    e->base.destructor = &date_to_string_kernel_extra::destruct;
    // The kernel data owns a reference to this type
    e->dst_string_dt = static_cast<const base_string_type *>(ndt::type(dst_string_dt).release());
    e->dst_metadata = dst_metadata;
    e->errmode = errmode;
    return offset_out + sizeof(date_to_string_kernel_extra);
}

