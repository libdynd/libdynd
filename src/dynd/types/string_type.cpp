//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/string_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/string_comparison_kernels.hpp>
#include <dynd/kernels/string_numeric_assignment_kernels.hpp>
#include <dynd/types/fixedstring_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/iter/string_iter.hpp>
#include <dynd/exceptions.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

string_type::string_type(string_encoding_t encoding)
    : base_string_type(string_type_id, sizeof(string_type_data),
                    sizeof(const char *), type_flag_scalar|type_flag_zeroinit|type_flag_blockref,
                    sizeof(string_type_arrmeta)),
            m_encoding(encoding)
{
    switch (encoding) {
        case string_encoding_ascii:
        case string_encoding_ucs_2:
        case string_encoding_utf_8:
        case string_encoding_utf_16:
        case string_encoding_utf_32:
            break;
        default:
            throw runtime_error("Unrecognized string encoding in string type constructor");
    }
}

string_type::~string_type()
{
}

void string_type::get_string_range(const char **out_begin, const char**out_end,
                const char *DYND_UNUSED(arrmeta), const char *data) const
{
    *out_begin = reinterpret_cast<const string_type_data *>(data)->begin;
    *out_end = reinterpret_cast<const string_type_data *>(data)->end;
}

void string_type::set_from_utf8_string(const char *arrmeta, char *dst,
                                       const char *utf8_begin,
                                       const char *utf8_end,
                                       const eval::eval_context *ectx) const
{
    const string_type_arrmeta *data_md =
        reinterpret_cast<const string_type_arrmeta *>(arrmeta);
    assign_error_mode errmode = ectx->errmode;
    const intptr_t src_charsize = 1;
    intptr_t dst_charsize = string_encoding_char_size_table[m_encoding];
    char *dst_begin = NULL, *dst_current, *dst_end = NULL;
    next_unicode_codepoint_t next_fn =
        get_next_unicode_codepoint_function(string_encoding_utf_8, errmode);
    append_unicode_codepoint_t append_fn =
        get_append_unicode_codepoint_function(m_encoding, errmode);
    uint32_t cp;

    memory_block_pod_allocator_api *allocator =
        get_memory_block_pod_allocator_api(data_md->blockref);

    // Allocate the initial output as the src number of characters + some padding
    // TODO: Don't add padding if the output is not a multi-character encoding
    allocator->allocate(data_md->blockref,
                        ((utf8_end - utf8_begin) / src_charsize + 16) *
                            dst_charsize * 1124 / 1024,
                        dst_charsize, &dst_begin, &dst_end);

    dst_current = dst_begin;
    while (utf8_begin < utf8_end) {
        cp = next_fn(utf8_begin, utf8_end);
        // Append the codepoint, or increase the allocated memory as necessary
        if (dst_end - dst_current >= 8) {
            append_fn(cp, dst_current, dst_end);
        } else {
            char *dst_begin_saved = dst_begin;
            allocator->resize(data_md->blockref, 2 * (dst_end - dst_begin),
                              &dst_begin, &dst_end);
            dst_current = dst_begin + (dst_current - dst_begin_saved);

            append_fn(cp, dst_current, dst_end);
        }
    }

    // Shrink-wrap the memory to just fit the string
    allocator->resize(data_md->blockref, dst_current - dst_begin, &dst_begin,
                      &dst_end);

    // Set the output
    reinterpret_cast<string_type_data *>(dst)->begin = dst_begin;
    reinterpret_cast<string_type_data*>(dst)->end = dst_end;
}

void string_type::print_data(std::ostream& o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
    uint32_t cp;
    next_unicode_codepoint_t next_fn;
    next_fn = get_next_unicode_codepoint_function(m_encoding, assign_error_nocheck);
    const char *begin = reinterpret_cast<const string_type_data *>(data)->begin;
    const char *end = reinterpret_cast<const string_type_data *>(data)->end;

    // Print as an escaped string
    o << "\"";
    while (begin < end) {
        cp = next_fn(begin, end);
        print_escaped_unicode_codepoint(o, cp, false);
    }
    o << "\"";
}

void string_type::print_type(std::ostream& o) const {

    o << "string";
    if (m_encoding != string_encoding_utf_8) {
        o << "['" << m_encoding << "']";
    }
}

bool string_type::is_unique_data_owner(const char *arrmeta) const
{
    const string_type_arrmeta *md = reinterpret_cast<const string_type_arrmeta *>(arrmeta);
    if (md->blockref != NULL &&
            (md->blockref->m_use_count != 1 ||
             md->blockref->m_type != pod_memory_block_type)) {
        return false;
    }
    return true;
}

ndt::type string_type::get_canonical_type() const
{
    return ndt::type(this, true);
}

void string_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                const char *DYND_UNUSED(arrmeta), const char *DYND_UNUSED(data)) const
{
    out_shape[i] = -1;
    if (i+1 < ndim) {
        stringstream ss;
        ss << "requested too many dimensions from type " << ndt::type(this, true);
        throw runtime_error(ss.str());
    }
}

bool string_type::is_lossless_assignment(
                const ndt::type& DYND_UNUSED(dst_tp),
                const ndt::type& DYND_UNUSED(src_tp)) const
{
    // Don't shortcut anything to 'nocheck' error checking, so that
    // decoding errors get caught appropriately.
    return false;
}

bool string_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != string_type_id) {
        return false;
    } else {
        const string_type *dt = static_cast<const string_type*>(&rhs);
        return m_encoding == dt->m_encoding;
    }
}

void string_type::arrmeta_default_construct(char *arrmeta, intptr_t DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const
{
    // Simply allocate a POD memory block
    string_type_arrmeta *md = reinterpret_cast<string_type_arrmeta *>(arrmeta);
    md->blockref = make_pod_memory_block().release();
}

void string_type::arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta, memory_block_data *embedded_reference) const
{
    // Copy the blockref, switching it to the embedded_reference if necessary
    const string_type_arrmeta *src_md = reinterpret_cast<const string_type_arrmeta *>(src_arrmeta);
    string_type_arrmeta *dst_md = reinterpret_cast<string_type_arrmeta *>(dst_arrmeta);
    dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
    if (dst_md->blockref) {
        memory_block_incref(dst_md->blockref);
    }
}

void string_type::arrmeta_reset_buffers(char *arrmeta) const
{
    const string_type_arrmeta *md = reinterpret_cast<const string_type_arrmeta *>(arrmeta);
    if (md->blockref != NULL && md->blockref->m_type == pod_memory_block_type) {
        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        allocator->reset(md->blockref);
    } else {
        throw runtime_error("can only reset the buffers of a dynd string "
                        "type if the memory block reference was constructed by default");
    }
}

void string_type::arrmeta_finalize_buffers(char *arrmeta) const
{
    string_type_arrmeta *md = reinterpret_cast<string_type_arrmeta *>(arrmeta);
    if (md->blockref != NULL) {
        // Finalize the memory block
        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        if (allocator != NULL) {
            allocator->finalize(md->blockref);
        }
    }
}

void string_type::arrmeta_destruct(char *arrmeta) const
{
    string_type_arrmeta *md = reinterpret_cast<string_type_arrmeta *>(arrmeta);
    if (md->blockref) {
        memory_block_decref(md->blockref);
    }
}

void string_type::arrmeta_debug_print(const char *arrmeta, std::ostream& o, const std::string& indent) const
{
    const string_type_arrmeta *md = reinterpret_cast<const string_type_arrmeta *>(arrmeta);
    o << indent << "string arrmeta\n";
    memory_block_debug_print(md->blockref, o, indent + " ");
}

size_t string_type::make_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    if (this == dst_tp.extended()) {
        switch (src_tp.get_type_id()) {
            case string_type_id: {
                return make_blockref_string_assignment_kernel(
                    ckb, ckb_offset, dst_arrmeta, get_encoding(), src_arrmeta,
                    src_tp.tcast<base_string_type>()->get_encoding(), kernreq,
                    ectx);
            }
            case fixedstring_type_id: {
                return make_fixedstring_to_blockref_string_assignment_kernel(
                    ckb, ckb_offset, dst_arrmeta, get_encoding(),
                    src_tp.get_data_size(),
                    src_tp.tcast<base_string_type>()->get_encoding(), kernreq,
                    ectx);
            }
            default: {
                if (!src_tp.is_builtin()) {
                    return src_tp.extended()->make_assignment_kernel(
                        ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
                        src_arrmeta, kernreq, ectx);
                } else {
                    return make_builtin_to_string_assignment_kernel(
                        ckb, ckb_offset, dst_tp, dst_arrmeta,
                        src_tp.get_type_id(), kernreq, ectx);
                }
            }
        }
    } else {
        if (dst_tp.is_builtin()) {
            return make_string_to_builtin_assignment_kernel(
                ckb, ckb_offset, dst_tp.get_type_id(), src_tp, src_arrmeta,
                kernreq, ectx);
        } else {
            stringstream ss;
            ss << "Cannot assign from " << src_tp << " to " << dst_tp;
            throw dynd::type_error(ss.str());
        }
    }
}

size_t string_type::make_comparison_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &src0_dt,
    const char *src0_arrmeta, const ndt::type &src1_dt,
    const char *src1_arrmeta, comparison_type_t comptype,
    const eval::eval_context *ectx) const
{
    if (this == src0_dt.extended()) {
        if (*this == *src1_dt.extended()) {
            return make_string_comparison_kernel(ckb, ckb_offset,
                            m_encoding,
                            comptype, ectx);
        } else if (src1_dt.get_kind() == string_kind) {
            return make_general_string_comparison_kernel(ckb, ckb_offset,
                            src0_dt, src0_arrmeta,
                            src1_dt, src1_arrmeta,
                            comptype, ectx);
        } else if (!src1_dt.is_builtin()) {
            return src1_dt.extended()->make_comparison_kernel(ckb, ckb_offset,
                            src0_dt, src0_arrmeta,
                            src1_dt, src1_arrmeta,
                            comptype, ectx);
        }
    }

    throw not_comparable_error(src0_dt, src1_dt, comptype);
}

void string_type::make_string_iter(dim_iter *out_di, string_encoding_t encoding,
            const char *arrmeta, const char *data,
            const memory_block_ptr& ref,
            intptr_t buffer_max_mem,
            const eval::eval_context *ectx) const
{
    const string_type_data *d = reinterpret_cast<const string_type_data *>(data);
    memory_block_ptr dataref = ref;
    const string_type_arrmeta *md = reinterpret_cast<const string_type_arrmeta *>(arrmeta);
    if (md->blockref != NULL) {
        dataref = memory_block_ptr(md->blockref);
    }
    iter::make_string_iter(out_di, encoding,
            m_encoding, d->begin, d->end, dataref, buffer_max_mem, ectx);
}

namespace {
struct string_is_avail_ck {
    static void single(char *dst, const char *const *src,
                       ckernel_prefix *DYND_UNUSED(self))
    {
        const string_type_data *std =
            *reinterpret_cast<const string_type_data *const *>(src);
        *dst = std->begin != NULL;
    }

    static void strided(char *dst, intptr_t dst_stride, const char *const *src,
                        const intptr_t *src_stride, size_t count,
                        ckernel_prefix *DYND_UNUSED(self))
    {
        const char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
            const string_type_data *std =
                reinterpret_cast<const string_type_data *>(src0);
            *dst = std->begin != NULL;
            dst += dst_stride;
            src0 += src0_stride;
        }
    }

    static intptr_t instantiate(const arrfunc_type_data *DYND_UNUSED(self),
                                dynd::ckernel_builder *ckb, intptr_t ckb_offset,
                                const ndt::type &dst_tp,
                                const char *DYND_UNUSED(dst_arrmeta),
                                const ndt::type *src_tp,
                                const char *const *DYND_UNUSED(src_arrmeta),
                                kernel_request_t kernreq,
                                const eval::eval_context *DYND_UNUSED(ectx))
    {
        if (src_tp[0].get_type_id() != option_type_id ||
                src_tp[0].tcast<option_type>()->get_value_type().get_type_id() !=
                    string_type_id) {
            stringstream ss;
            ss << "Expected source type ?string, got " << src_tp[0];
            throw type_error(ss.str());
        }
        if (dst_tp.get_type_id() != bool_type_id) {
            stringstream ss;
            ss << "Expected destination type bool, got " << dst_tp;
            throw type_error(ss.str());
        }
        ckernel_prefix *ckp = ckb->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
        ckp->set_expr_function<string_is_avail_ck>(kernreq);
        return ckb_offset;
    }
};

struct string_assign_na_ck {
    static void single(char *dst, const char *const *DYND_UNUSED(src),
                       ckernel_prefix *DYND_UNUSED(self))
    {
        const string_type_data *std =
            reinterpret_cast<const string_type_data *>(dst);
        if (std->begin != NULL) {
            throw invalid_argument("Cannot assign an NA to a dynd string after "
                                   "it has been allocated");
        }
    }

    static void strided(char *dst, intptr_t dst_stride,
                        const char *const *DYND_UNUSED(src),
                        const intptr_t *DYND_UNUSED(src_stride), size_t count,
                        ckernel_prefix *DYND_UNUSED(self))
    {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
            const string_type_data *std =
                reinterpret_cast<const string_type_data *>(dst);
            if (std->begin != NULL) {
                throw invalid_argument(
                    "Cannot assign an NA to a dynd string after "
                    "it has been allocated");
            }
        }
    }

    static intptr_t instantiate(const arrfunc_type_data *DYND_UNUSED(self),
                                dynd::ckernel_builder *ckb, intptr_t ckb_offset,
                                const ndt::type &dst_tp,
                                const char *DYND_UNUSED(dst_arrmeta),
                                const ndt::type *DYND_UNUSED(src_tp),
                                const char *const *DYND_UNUSED(src_arrmeta),
                                kernel_request_t kernreq,
                                const eval::eval_context *DYND_UNUSED(ectx))
    {
        if (dst_tp.get_type_id() != option_type_id ||
                dst_tp.tcast<option_type>()->get_value_type().get_type_id() !=
                    string_type_id) {
            stringstream ss;
            ss << "Expected destination type ?string, got " << dst_tp;
            throw type_error(ss.str());
        }
        ckernel_prefix *ckp = ckb->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
        ckp->set_expr_function<string_assign_na_ck>(kernreq);
        return ckb_offset;
    }
};
} // anonymous namespace

nd::array string_type::get_option_nafunc() const
{
    nd::array naf = nd::empty(option_type::make_nafunc_type());
    arrfunc_type_data *is_avail =
        reinterpret_cast<arrfunc_type_data *>(naf.get_ndo()->m_data_pointer);
    arrfunc_type_data *assign_na = is_avail + 1;

    // Use a typevar instead of option[T] to avoid a circular dependency
    is_avail->func_proto = ndt::make_funcproto(ndt::make_typevar("T"),
                                               ndt::make_type<dynd_bool>());
    is_avail->instantiate = &string_is_avail_ck::instantiate;
    assign_na->func_proto =
        ndt::make_funcproto(0, NULL, ndt::make_typevar("T"));
    assign_na->instantiate = &string_assign_na_ck::instantiate;
    naf.flag_as_immutable();
    return naf;
}
