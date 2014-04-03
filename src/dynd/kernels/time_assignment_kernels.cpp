//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/time_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/cstruct_type.hpp>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// string to time assignment

namespace {
    struct string_to_time_kernel_extra {
        typedef string_to_time_kernel_extra extra_type;

        ckernel_prefix base;
        const base_string_type *src_string_tp;
        const char *src_metadata;
        assign_error_mode errmode;

        static void single(char *dst, const char *src, ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const string& s = e->src_string_tp->get_utf8_string(e->src_metadata, src, e->errmode);
            time_hmst hmst;
            // TODO: properly distinguish "time" and "option[time]" with respect to NA support
            if (s == "NA") {
                hmst.set_to_na();
            } else {
                hmst.set_from_str(s);
            }
            *reinterpret_cast<int64_t *>(dst) = hmst.to_ticks();
        }

        static void destruct(ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            base_type_xdecref(e->src_string_tp);
        }
    };
} // anonymous namespace

size_t dynd::make_string_to_time_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& DYND_UNUSED(dst_time_tp),
                const ndt::type& src_string_tp, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    // TODO: Use dst_time_tp when time zone is developed more.
    if (src_string_tp.get_kind() != string_kind) {
        stringstream ss;
        ss << "make_string_to_time_assignment_kernel: source type " << src_string_tp << " is not a string type";
        throw runtime_error(ss.str());
    }

    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    out->ensure_capacity(offset_out + sizeof(string_to_time_kernel_extra));
    string_to_time_kernel_extra *e = out->get_at<string_to_time_kernel_extra>(offset_out);
    e->base.set_function<unary_single_operation_t>(&string_to_time_kernel_extra::single);
    e->base.destructor = &string_to_time_kernel_extra::destruct;
    // The kernel data owns a reference to this type
    e->src_string_tp = static_cast<const base_string_type *>(ndt::type(src_string_tp).release());
    e->src_metadata = src_metadata;
    e->errmode = errmode;
    return offset_out + sizeof(string_to_time_kernel_extra);
}

/////////////////////////////////////////
// time to string assignment

namespace {
    struct time_to_string_kernel_extra {
        typedef time_to_string_kernel_extra extra_type;

        ckernel_prefix base;
        const base_string_type *dst_string_tp;
        const char *dst_metadata;
        assign_error_mode errmode;

        static void single(char *dst, const char *src, ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            time_hmst hmst;
            hmst.set_from_ticks(*reinterpret_cast<const int64_t *>(src));
            string s = hmst.to_str();
            if (s.empty()) {
                s = "NA";
            }
            e->dst_string_tp->set_utf8_string(e->dst_metadata, dst, e->errmode, s);
        }

        static void destruct(ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            base_type_xdecref(e->dst_string_tp);
        }
    };
} // anonymous namespace

size_t dynd::make_time_to_string_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_string_tp, const char *dst_metadata,
                const ndt::type& DYND_UNUSED(src_time_tp),
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    // TODO: Use src_time_tp when time zone is developed more.
    if (dst_string_tp.get_kind() != string_kind) {
        stringstream ss;
        ss << "get_time_to_string_assignment_kernel: dest type " << dst_string_tp << " is not a string type";
        throw runtime_error(ss.str());
    }

    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    out->ensure_capacity(offset_out + sizeof(time_to_string_kernel_extra));
    time_to_string_kernel_extra *e = out->get_at<time_to_string_kernel_extra>(offset_out);
    e->base.set_function<unary_single_operation_t>(&time_to_string_kernel_extra::single);
    e->base.destructor = &time_to_string_kernel_extra::destruct;
    // The kernel data owns a reference to this type
    e->dst_string_tp = static_cast<const base_string_type *>(ndt::type(dst_string_tp).release());
    e->dst_metadata = dst_metadata;
    e->errmode = errmode;
    return offset_out + sizeof(time_to_string_kernel_extra);
}

