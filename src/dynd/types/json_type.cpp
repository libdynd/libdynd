//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
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
          sizeof(json_type_metadata))
{
    // While stored as a string, JSON data can hold many types of data
    m_members.kind = dynamic_kind;
}

json_type::~json_type()
{
}

void json_type::get_string_range(const char **out_begin, const char**out_end,
                const char *DYND_UNUSED(metadata), const char *data) const
{
    *out_begin = reinterpret_cast<const json_type_data *>(data)->begin;
    *out_end = reinterpret_cast<const json_type_data *>(data)->end;
}

void json_type::set_utf8_string(const char *data_metadata, char *data,
                assign_error_mode errmode, const char* utf8_begin, const char *utf8_end) const
{
    // Validate that the input is JSON
    if (errmode != assign_error_none) {
        validate_json(utf8_begin, utf8_end);
    }

    const json_type_metadata *data_md = reinterpret_cast<const json_type_metadata *>(data_metadata);

    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(data_md->blockref);

    allocator->allocate(data_md->blockref, utf8_end - utf8_begin, 1,
                    &reinterpret_cast<json_type_data *>(data)->begin,
                    &reinterpret_cast<json_type_data*>(data)->end);

    memcpy(reinterpret_cast<json_type_data *>(data)->begin,
                    utf8_begin, utf8_end - utf8_begin);
}

void json_type::print_data(std::ostream& o, const char *DYND_UNUSED(metadata), const char *data) const
{
    uint32_t cp;
    next_unicode_codepoint_t next_fn;
    next_fn = get_next_unicode_codepoint_function(string_encoding_utf_8, assign_error_none);
    const char *begin = reinterpret_cast<const json_type_data *>(data)->begin;
    const char *end = reinterpret_cast<const json_type_data *>(data)->end;

    // Print as an escaped string
    o << "\"";
    while (begin < end) {
        cp = next_fn(begin, end);
        print_escaped_unicode_codepoint(o, cp);
    }
    o << "\"";
}

void json_type::print_type(std::ostream& o) const {

    o << "json";
}

bool json_type::is_unique_data_owner(const char *metadata) const
{
    const json_type_metadata *md = reinterpret_cast<const json_type_metadata *>(metadata);
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


bool json_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
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

bool json_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else {
        return rhs.get_type_id() == json_type_id;
    }
}

void json_type::metadata_default_construct(char *metadata, intptr_t DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const
{
    // Simply allocate a POD memory block
    json_type_metadata *md = reinterpret_cast<json_type_metadata *>(metadata);
    md->blockref = make_pod_memory_block().release();
}

void json_type::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    // Copy the blockref, switching it to the embedded_reference if necessary
    const json_type_metadata *src_md = reinterpret_cast<const json_type_metadata *>(src_metadata);
    json_type_metadata *dst_md = reinterpret_cast<json_type_metadata *>(dst_metadata);
    dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
    if (dst_md->blockref) {
        memory_block_incref(dst_md->blockref);
    }
}

void json_type::metadata_reset_buffers(char *metadata) const
{
    const json_type_metadata *md = reinterpret_cast<const json_type_metadata *>(metadata);
    if (md->blockref != NULL && md->blockref->m_type == pod_memory_block_type) {
        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        allocator->reset(md->blockref);
    } else {
        throw runtime_error("can only reset the buffers of a dynd json string "
                        "type if the memory block reference was constructed by default");
    }
}

void json_type::metadata_finalize_buffers(char *metadata) const
{
    json_type_metadata *md = reinterpret_cast<json_type_metadata *>(metadata);
    if (md->blockref != NULL) {
        // Finalize the memory block
        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        if (allocator != NULL) {
            allocator->finalize(md->blockref);
        }
    }
}

void json_type::metadata_destruct(char *metadata) const
{
    json_type_metadata *md = reinterpret_cast<json_type_metadata *>(metadata);
    if (md->blockref) {
        memory_block_decref(md->blockref);
    }
}

void json_type::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    const json_type_metadata *md = reinterpret_cast<const json_type_metadata *>(metadata);
    o << indent << "json metadata\n";
    memory_block_debug_print(md->blockref, o, indent + " ");
}

namespace {
   struct string_to_json_kernel_extra {
        typedef string_to_json_kernel_extra extra_type;

        ckernel_prefix base;
        const char *dst_metadata;
        bool validate;

        static void single(char *dst, const char *src, ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const json_type_metadata *md = reinterpret_cast<const json_type_metadata *>(e->dst_metadata);
            json_type_data *out_d = reinterpret_cast<json_type_data *>(dst);
            // First copy it as a string
            (e+1)->base.get_function<unary_single_operation_t>()(dst, src, &(e+1)->base);
            // Then validate that it's correct JSON
            if (e->validate) {
                try {
                    validate_json(out_d->begin, out_d->end);
                } catch(const std::exception&) {
                    // Free the memory allocated for the output json data
                    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
                    allocator->allocate(md->blockref, 0, 1, &out_d->begin, &out_d->end);
                    out_d->begin = NULL;
                    out_d->end = NULL;
                    throw;
                }
            }
        }

        static void destruct(ckernel_prefix *self)
        {
            self->destroy_child_ckernel(sizeof(extra_type));
        }
    };
} // anonymous namespace

size_t json_type::make_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_tp.extended()) {
        switch (src_tp.get_type_id()) {
            case json_type_id: {
                // Assume the input is valid JSON when copying from json to json types
                return make_blockref_string_assignment_kernel(out, offset_out,
                                dst_metadata, string_encoding_utf_8,
                                src_metadata, string_encoding_utf_8,
                                kernreq, errmode, ectx);
            }
            case string_type_id:
            case fixedstring_type_id: {
                offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
                out->ensure_capacity(offset_out + sizeof(string_to_json_kernel_extra));
                string_to_json_kernel_extra *e = out->get_at<string_to_json_kernel_extra>(offset_out);
                e->base.set_function<unary_single_operation_t>(&string_to_json_kernel_extra::single);
                e->base.destructor = &string_to_json_kernel_extra::destruct;
                e->dst_metadata = dst_metadata;
                e->validate = (errmode != assign_error_none);
                if (src_tp.get_type_id() == string_type_id) {
                    return make_blockref_string_assignment_kernel(
                                    out, offset_out + sizeof(string_to_json_kernel_extra),
                                    dst_metadata, string_encoding_utf_8,
                                    src_metadata,
                                    src_tp.tcast<base_string_type>()->get_encoding(),
                                    kernel_request_single, errmode, ectx);
                } else {
                    return make_fixedstring_to_blockref_string_assignment_kernel(
                                    out, offset_out + sizeof(string_to_json_kernel_extra),
                                    dst_metadata, string_encoding_utf_8,
                                    src_tp.get_data_size(),
                                    src_tp.tcast<base_string_type>()->get_encoding(),
                                    kernel_request_single, errmode, ectx);
                }
            }
            default: {
                if (!src_tp.is_builtin()) {
                    return src_tp.extended()->make_assignment_kernel(out, offset_out,
                                    dst_tp, dst_metadata,
                                    src_tp, src_metadata,
                                    kernreq, errmode, ectx);
                } else {
                    return make_builtin_to_string_assignment_kernel(out, offset_out,
                                dst_tp, dst_metadata,
                                src_tp.get_type_id(),
                                kernreq, errmode, ectx);
                }
            }
        }
    } else {
        if (dst_tp.is_builtin()) {
            return make_string_to_builtin_assignment_kernel(out, offset_out,
                            dst_tp.get_type_id(),
                            src_tp, src_metadata,
                            kernreq, errmode, ectx);
        } else if(dst_tp.get_type_id() == string_type_id) {
           return make_blockref_string_assignment_kernel(out, offset_out,
                            dst_metadata, dst_tp.tcast<base_string_type>()->get_encoding(),
                            src_metadata, string_encoding_utf_8,
                            kernreq, errmode, ectx);
        } else if(dst_tp.get_type_id() == fixedstring_type_id) {
            return make_blockref_string_to_fixedstring_assignment_kernel(
                out, offset_out, dst_tp.get_data_size(),
                dst_tp.tcast<base_string_type>()->get_encoding(),
                string_encoding_utf_8, kernreq, errmode, ectx);
        } else {
            stringstream ss;
            ss << "Cannot assign from " << src_tp << " to " << dst_tp;
            throw dynd::type_error(ss.str());
        }
    }
}

void json_type::make_string_iter(dim_iter *out_di, string_encoding_t encoding,
            const char *metadata, const char *data,
            const memory_block_ptr& ref,
            intptr_t buffer_max_mem,
            const eval::eval_context *ectx) const
{
    const string_type_data *d = reinterpret_cast<const string_type_data *>(data);
    memory_block_ptr dataref = ref;
    const string_type_metadata *md = reinterpret_cast<const string_type_metadata *>(metadata);
    if (md->blockref != NULL) {
        dataref = memory_block_ptr(md->blockref);
    }
    iter::make_string_iter(out_di, encoding,
            string_encoding_utf_8, d->begin, d->end, dataref, buffer_max_mem, ectx);
}
