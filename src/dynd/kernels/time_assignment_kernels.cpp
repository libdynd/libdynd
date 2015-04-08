//
// Copyright (C) 2011-15 DyND Developers
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
    struct string_to_time_ck : public kernels::unary_ck<string_to_time_ck> {
        ndt::type m_src_string_tp;
        const char *m_src_arrmeta;
        assign_error_mode m_errmode;

        inline void single(char *dst, const char *src)
        {
            const base_string_type *bst = static_cast<const base_string_type *>(m_src_string_tp.extended());
            const string& s = bst->get_utf8_string(m_src_arrmeta, src, m_errmode);
            time_hmst hmst;
            // TODO: properly distinguish "time" and "option[time]" with respect to NA support
            if (s == "NA") {
                hmst.set_to_na();
            } else {
                hmst.set_from_str(s);
            }
            *reinterpret_cast<int64_t *>(dst) = hmst.to_ticks();
        }
    };
} // anonymous namespace

size_t dynd::make_string_to_time_assignment_kernel(
    void *ckb, intptr_t ckb_offset,
    const ndt::type &DYND_UNUSED(dst_time_tp), const ndt::type &src_string_tp,
    const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
    typedef string_to_time_ck self_type;
    // TODO: Use dst_time_tp when time zone is developed more.
    if (src_string_tp.get_kind() != string_kind) {
        stringstream ss;
        ss << "make_string_to_time_assignment_kernel: source type " << src_string_tp << " is not a string type";
        throw runtime_error(ss.str());
    }

    self_type *self = self_type::make(ckb, kernreq, ckb_offset);
    self->m_src_string_tp = src_string_tp;
    self->m_src_arrmeta = src_arrmeta;
    self->m_errmode = ectx->errmode;
    return ckb_offset;
}

/////////////////////////////////////////
// time to string assignment

namespace {
    struct time_to_string_ck : public kernels::unary_ck<time_to_string_ck> {
        ndt::type m_dst_string_tp;
        const char *m_dst_arrmeta;
        eval::eval_context m_ectx;

        inline void single(char *dst, const char *src)
        {
            time_hmst hmst;
            hmst.set_from_ticks(*reinterpret_cast<const int64_t *>(src));
            string s = hmst.to_str();
            if (s.empty()) {
                s = "NA";
            }
            const base_string_type *bst = static_cast<const base_string_type *>(m_dst_string_tp.extended());
            bst->set_from_utf8_string(m_dst_arrmeta, dst, s, &m_ectx);
        }
    };
} // anonymous namespace

size_t dynd::make_time_to_string_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_string_tp,
    const char *dst_arrmeta, const ndt::type &DYND_UNUSED(src_time_tp),
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
    typedef time_to_string_ck self_type;
    // TODO: Use src_time_tp when time zone is developed more.
    if (dst_string_tp.get_kind() != string_kind) {
        stringstream ss;
        ss << "get_time_to_string_assignment_kernel: dest type " << dst_string_tp << " is not a string type";
        throw runtime_error(ss.str());
    }

    self_type *self = self_type::make(ckb, kernreq, ckb_offset);
    self->m_dst_string_tp = dst_string_tp;
    self->m_dst_arrmeta = dst_arrmeta;
    self->m_ectx = *ectx;
    return ckb_offset;
}

