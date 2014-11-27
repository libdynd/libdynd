//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/json_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/string_comparison_kernels.hpp>
#include <dynd/kernels/string_numeric_assignment_kernels.hpp>
#include <dynd/types/fixedstring_type.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/iter/string_iter.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

json_type::json_type()
    : base_string_type(
          json_type_id, sizeof(json_type_data), sizeof(const char *),
          type_flag_scalar | type_flag_zeroinit | type_flag_blockref,
          sizeof(json_type_arrmeta))
{
    // While stored as a string, JSON data can hold many types of data
    m_members.kind = dynamic_kind;
}

json_type::~json_type()
{
}

void json_type::get_string_range(const char **out_begin, const char **out_end,
                                 const char *DYND_UNUSED(arrmeta),
                                 const char *data) const
{
    *out_begin = reinterpret_cast<const json_type_data *>(data)->begin;
    *out_end = reinterpret_cast<const json_type_data *>(data)->end;
}

void json_type::set_from_utf8_string(const char *arrmeta, char *dst,
                                     const char *utf8_begin,
                                     const char *utf8_end,
                                     const eval::eval_context *ectx) const
{
    // Validate that the input is JSON
    if (ectx->errmode != assign_error_nocheck) {
        validate_json(utf8_begin, utf8_end);
    }

    const json_type_arrmeta *data_md =
        reinterpret_cast<const json_type_arrmeta *>(arrmeta);

    memory_block_pod_allocator_api *allocator =
        get_memory_block_pod_allocator_api(data_md->blockref);

    allocator->allocate(data_md->blockref, utf8_end - utf8_begin, 1,
                        &reinterpret_cast<json_type_data *>(dst)->begin,
                        &reinterpret_cast<json_type_data *>(dst)->end);

    memcpy(reinterpret_cast<json_type_data *>(dst)->begin,
                    utf8_begin, utf8_end - utf8_begin);
}

void json_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta),
                           const char *data) const
{
    uint32_t cp;
    next_unicode_codepoint_t next_fn;
    next_fn = get_next_unicode_codepoint_function(string_encoding_utf_8,
                                                  assign_error_nocheck);
    const char *begin = reinterpret_cast<const json_type_data *>(data)->begin;
    const char *end = reinterpret_cast<const json_type_data *>(data)->end;

    // Print as an escaped string
    o << "\"";
    while (begin < end) {
        cp = next_fn(begin, end);
        print_escaped_unicode_codepoint(o, cp, false);
    }
    o << "\"";
}

void json_type::print_type(std::ostream& o) const {

    o << "json";
}

bool json_type::is_unique_data_owner(const char *arrmeta) const
{
    const json_type_arrmeta *md =
        reinterpret_cast<const json_type_arrmeta *>(arrmeta);
    if (md->blockref != NULL &&
        (md->blockref->m_use_count != 1 ||
         md->blockref->m_type != pod_memory_block_type)) {
        return false;
    }
    return true;
}

ndt::type json_type::get_canonical_type() const
{
    return ndt::type(this, true);
}

bool json_type::is_lossless_assignment(const ndt::type &dst_tp,
                                       const ndt::type &src_tp) const
{
    if (dst_tp.extended() == this) {
        if (src_tp.get_type_id() == json_type_id) {
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

bool json_type::operator==(const base_type &rhs) const
{
    if (this == &rhs) {
        return true;
    } else {
        return rhs.get_type_id() == json_type_id;
    }
}

void json_type::arrmeta_default_construct(char *arrmeta,
                                          bool blockref_alloc) const
{
  // Simply allocate a POD memory block
  if (blockref_alloc) {
    json_type_arrmeta *md = reinterpret_cast<json_type_arrmeta *>(arrmeta);
    md->blockref = make_pod_memory_block().release();
  }
}

void json_type::arrmeta_copy_construct(char *dst_arrmeta,
                                       const char *src_arrmeta,
                                       memory_block_data *embedded_reference)
    const
{
    // Copy the blockref, switching it to the embedded_reference if necessary
    const json_type_arrmeta *src_md =
        reinterpret_cast<const json_type_arrmeta *>(src_arrmeta);
    json_type_arrmeta *dst_md =
        reinterpret_cast<json_type_arrmeta *>(dst_arrmeta);
    dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
    if (dst_md->blockref) {
        memory_block_incref(dst_md->blockref);
    }
}

void json_type::arrmeta_reset_buffers(char *arrmeta) const
{
    const json_type_arrmeta *md =
        reinterpret_cast<const json_type_arrmeta *>(arrmeta);
    if (md->blockref != NULL && md->blockref->m_type == pod_memory_block_type) {
        memory_block_pod_allocator_api *allocator =
            get_memory_block_pod_allocator_api(md->blockref);
        allocator->reset(md->blockref);
    } else {
        throw runtime_error(
            "can only reset the buffers of a dynd json string "
            "type if the memory block reference was constructed by default");
    }
}

void json_type::arrmeta_finalize_buffers(char *arrmeta) const
{
    json_type_arrmeta *md = reinterpret_cast<json_type_arrmeta *>(arrmeta);
    if (md->blockref != NULL) {
        // Finalize the memory block
        memory_block_pod_allocator_api *allocator =
            get_memory_block_pod_allocator_api(md->blockref);
        if (allocator != NULL) {
            allocator->finalize(md->blockref);
        }
    }
}

void json_type::arrmeta_destruct(char *arrmeta) const
{
    json_type_arrmeta *md = reinterpret_cast<json_type_arrmeta *>(arrmeta);
    if (md->blockref) {
        memory_block_decref(md->blockref);
    }
}

void json_type::arrmeta_debug_print(const char *arrmeta, std::ostream &o,
                                    const std::string &indent) const
{
    const json_type_arrmeta *md =
        reinterpret_cast<const json_type_arrmeta *>(arrmeta);
    o << indent << "json arrmeta\n";
    memory_block_debug_print(md->blockref, o, indent + " ");
}

namespace {
struct string_to_json_ck
  : public kernels::unary_ck<string_to_json_ck> {
    const char *m_dst_arrmeta;
    bool m_validate;

    inline void single(char *dst, char *src)
    {
        const json_type_arrmeta *md =
            reinterpret_cast<const json_type_arrmeta *>(m_dst_arrmeta);
        json_type_data *out_d = reinterpret_cast<json_type_data *>(dst);
        // First copy it as a string
        ckernel_prefix *child = get_child_ckernel();
        expr_single_t child_fn = child->get_function<expr_single_t>();
        child_fn(dst, &src, child);
        // Then validate that it's correct JSON
        if (m_validate) {
            try { validate_json(out_d->begin, out_d->end); }
            catch (const std::exception &)
            {
                // Free the memory allocated for the output json data
                memory_block_pod_allocator_api *allocator =
                    get_memory_block_pod_allocator_api(md->blockref);
                allocator->allocate(md->blockref, 0, 1, &out_d->begin,
                                    &out_d->end);
                out_d->begin = NULL;
                out_d->end = NULL;
                throw;
            }
        }
    }

    inline void destruct_children()
    {
        // Destroy the child ckernel
        get_child_ckernel()->destroy();
    }
};
} // anonymous namespace

size_t json_type::make_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    if (this == dst_tp.extended()) {
        switch (src_tp.get_type_id()) {
            case json_type_id: {
                // Assume the input is valid JSON when copying from json to json types
                return make_blockref_string_assignment_kernel(
                    ckb, ckb_offset, dst_arrmeta, string_encoding_utf_8,
                    src_arrmeta, string_encoding_utf_8, kernreq, ectx);
            }
            case string_type_id:
            case fixedstring_type_id: {
                string_to_json_ck *self =
                    string_to_json_ck::create(ckb, kernreq, ckb_offset);
                self->m_dst_arrmeta = dst_arrmeta;
                self->m_validate = (ectx->errmode != assign_error_nocheck);
                if (src_tp.get_type_id() == string_type_id) {
                    return make_blockref_string_assignment_kernel(
                        ckb, ckb_offset, dst_arrmeta, string_encoding_utf_8,
                        src_arrmeta,
                        src_tp.extended<base_string_type>()->get_encoding(),
                        kernel_request_single, ectx);
                } else {
                    return make_fixedstring_to_blockref_string_assignment_kernel(
                        ckb, ckb_offset, dst_arrmeta, string_encoding_utf_8,
                        src_tp.get_data_size(),
                        src_tp.extended<base_string_type>()->get_encoding(),
                        kernel_request_single, ectx);
                }
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
        } else if(dst_tp.get_type_id() == string_type_id) {
            return make_blockref_string_assignment_kernel(
                ckb, ckb_offset, dst_arrmeta,
                dst_tp.extended<base_string_type>()->get_encoding(), src_arrmeta,
                string_encoding_utf_8, kernreq, ectx);
        } else if(dst_tp.get_type_id() == fixedstring_type_id) {
            return make_blockref_string_to_fixedstring_assignment_kernel(
                ckb, ckb_offset, dst_tp.get_data_size(),
                dst_tp.extended<base_string_type>()->get_encoding(),
                string_encoding_utf_8, kernreq, ectx);
        } else {
            stringstream ss;
            ss << "Cannot assign from " << src_tp << " to " << dst_tp;
            throw dynd::type_error(ss.str());
        }
    }
}

void json_type::make_string_iter(dim_iter *out_di, string_encoding_t encoding,
                                 const char *arrmeta, const char *data,
                                 const memory_block_ptr &ref,
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
            string_encoding_utf_8, d->begin, d->end, dataref, buffer_max_mem, ectx);
}
