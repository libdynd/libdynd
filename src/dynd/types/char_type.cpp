//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/char_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/string_comparison_kernels.hpp>
#include <dynd/kernels/string_numeric_assignment_kernels.hpp>
#include <dynd/exceptions.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

char_type::char_type(string_encoding_t encoding)
: base_type(char_type_id, char_kind, string_encoding_char_size_table[encoding],
        string_encoding_char_size_table[encoding], type_flag_scalar, 0, 0, 0),
m_encoding(encoding)
{
    switch (encoding) {
        case string_encoding_ascii:
        case string_encoding_latin1:
        case string_encoding_ucs_2:
        case string_encoding_utf_32:
            break;
        default: {
            stringstream ss;
            ss << "dynd char type requires fixed-size encoding, " << encoding << " is not supported";
            throw runtime_error(ss.str());
        }
    }
}

char_type::~char_type()
{
}

uint32_t char_type::get_code_point(const char *data) const
{
    next_unicode_codepoint_t next_fn;
    next_fn = get_next_unicode_codepoint_function(m_encoding, assign_error_nocheck);
    return next_fn(data, data + get_data_size());
}

void char_type::set_code_point(char *out_data, uint32_t cp)
{
    append_unicode_codepoint_t append_fn;
    append_fn = get_append_unicode_codepoint_function(m_encoding, assign_error_nocheck);
    append_fn(cp, out_data, out_data + get_data_size());
}

void char_type::print_data(std::ostream& o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
    // Print as an escaped string
    o << "\"";
    print_escaped_unicode_codepoint(o, get_code_point(data), false);
    o << "\"";
}

void char_type::print_type(std::ostream& o) const {

    o << "char";
    if (m_encoding != string_encoding_utf_32) {
        o << "['" << m_encoding << "']";
    }
}

ndt::type char_type::get_canonical_type() const
{
    // The canonical char type is UTF-32
    if (m_encoding == string_encoding_utf_32) {
        return ndt::type(this, true);
    }
    else {
        return ndt::type(new char_type(string_encoding_utf_32), false);
    }
}

bool char_type::is_lossless_assignment(
    const ndt::type& DYND_UNUSED(dst_tp),
    const ndt::type& DYND_UNUSED(src_tp)) const
{
    // Don't shortcut anything to 'nocheck' error checking, so that
    // decoding errors get caught appropriately.
    return false;
}

bool char_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    }
    else if (rhs.get_type_id() != char_type_id) {
        return false;
    }
    else {
        const char_type *dt = static_cast<const char_type*>(&rhs);
        return m_encoding == dt->m_encoding;
    }
}

size_t char_type::make_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    if (this == dst_tp.extended()) {
        if (dst_tp == src_tp) {
            // If the types are the same, it's a POD assignment
            return make_pod_typed_data_assignment_kernel(ckb, ckb_offset,
                        m_members.data_size, m_members.data_alignment,
                        kernreq);
        }
        switch (src_tp.get_type_id()) {
            case char_type_id: {
                // Use the fixedstring assignment to do this conversion
                const char_type *src_fs = src_tp.tcast<char_type>();
                return make_fixedstring_assignment_kernel(
                    ckb, ckb_offset, get_data_size(), m_encoding,
                    src_fs->get_data_size(), src_fs->m_encoding, kernreq, ectx);
            }
            case fixedstring_type_id: {
                // Use the fixedstring assignment to do this conversion
                const base_string_type *src_fs = src_tp.tcast<base_string_type>();
                return make_fixedstring_assignment_kernel(
                    ckb, ckb_offset, get_data_size(), m_encoding,
                    src_fs->get_data_size(), src_fs->get_encoding(), kernreq,
                    ectx);
            }
            case string_type_id: {
                const base_string_type *src_fs = src_tp.tcast<base_string_type>();
                return make_blockref_string_to_fixedstring_assignment_kernel(
                    ckb, ckb_offset, get_data_size(), m_encoding,
                    src_fs->get_encoding(), kernreq, ectx);
            }
            default: {
                if (!src_tp.is_builtin()) {
                    return src_tp.extended()->make_assignment_kernel(
                        ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
                        src_arrmeta, kernreq, ectx);
                }
                break;
            }
        }
    }
    else {
        switch (dst_tp.get_type_id()) {
            case fixedstring_type_id: {
                // Use the fixedstring assignment to do this conversion
                const base_string_type *dst_fs = dst_tp.tcast<base_string_type>();
                return make_fixedstring_assignment_kernel(
                    ckb, ckb_offset, dst_fs->get_data_size(),
                    dst_fs->get_encoding(), get_data_size(), m_encoding,
                    kernreq, ectx);
            }
            case string_type_id: {
                const base_string_type *dst_fs = dst_tp.tcast<base_string_type>();
                return make_fixedstring_to_blockref_string_assignment_kernel(
                    ckb, ckb_offset, dst_arrmeta, dst_fs->get_encoding(),
                    get_data_size(), m_encoding, kernreq, ectx);
            }
            default: {
                break;
            }
        }
    }

    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw dynd::type_error(ss.str());
}

size_t char_type::make_comparison_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset,
    const ndt::type& src0_dt, const char *src0_arrmeta,
    const ndt::type& src1_dt, const char *src1_arrmeta,
    comparison_type_t comptype,
    const eval::eval_context *ectx) const
{
    if (this == src0_dt.extended()) {
        if (*this == *src1_dt.extended()) {
            return make_string_comparison_kernel(ckb, ckb_offset,
                m_encoding,
                comptype, ectx);
        }
        else if (src1_dt.get_kind() == string_kind) {
            return make_general_string_comparison_kernel(ckb, ckb_offset,
                src0_dt, src0_arrmeta,
                src1_dt, src1_arrmeta,
                comptype, ectx);
        }
        else if (!src1_dt.is_builtin()) {
            return src1_dt.extended()->make_comparison_kernel(ckb, ckb_offset,
                src0_dt, src0_arrmeta,
                src1_dt, src1_arrmeta,
                comptype, ectx);
        }
    }

    throw not_comparable_error(src0_dt, src1_dt, comptype);
}
