//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/json_dtype.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/string_comparison_kernels.hpp>
#include <dynd/kernels/string_numeric_assignment_kernels.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/exceptions.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

json_dtype::json_dtype()
    : base_string_dtype(json_type_id, sizeof(json_dtype_data), sizeof(const char *), dtype_flag_scalar|dtype_flag_zeroinit)
{
}

json_dtype::~json_dtype()
{
}

void json_dtype::get_string_range(const char **out_begin, const char**out_end,
                const char *DYND_UNUSED(metadata), const char *data) const
{
    *out_begin = reinterpret_cast<const json_dtype_data *>(data)->begin;
    *out_end = reinterpret_cast<const json_dtype_data *>(data)->end;
}

void json_dtype::set_utf8_string(const char *data_metadata, char *data,
                assign_error_mode errmode, const char* utf8_begin, const char *utf8_end) const
{
    // Validate that the input is JSON
    if (errmode != assign_error_none) {
        validate_json(utf8_begin, utf8_end);
    }

    const json_dtype_metadata *data_md = reinterpret_cast<const json_dtype_metadata *>(data_metadata);

    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(data_md->blockref);

    allocator->allocate(data_md->blockref, utf8_end - utf8_begin, 1,
                    &reinterpret_cast<json_dtype_data *>(data)->begin,
                    &reinterpret_cast<json_dtype_data*>(data)->end);

    memcpy(reinterpret_cast<json_dtype_data *>(data)->begin,
                    utf8_begin, utf8_end - utf8_begin);
}

void json_dtype::print_data(std::ostream& o, const char *DYND_UNUSED(metadata), const char *data) const
{
    uint32_t cp;
    next_unicode_codepoint_t next_fn;
    next_fn = get_next_unicode_codepoint_function(string_encoding_utf_8, assign_error_none);
    const char *begin = reinterpret_cast<const json_dtype_data *>(data)->begin;
    const char *end = reinterpret_cast<const json_dtype_data *>(data)->end;

    // Print as an escaped string
    o << "\"";
    while (begin < end) {
        cp = next_fn(begin, end);
        print_escaped_unicode_codepoint(o, cp);
    }
    o << "\"";
}

void json_dtype::print_dtype(std::ostream& o) const {

    o << "json";
}

bool json_dtype::is_unique_data_owner(const char *metadata) const
{
    const json_dtype_metadata *md = reinterpret_cast<const json_dtype_metadata *>(metadata);
    if (md->blockref != NULL &&
            (md->blockref->m_use_count != 1 ||
             md->blockref->m_type != pod_memory_block_type)) {
        return false;
    }
    return true;
}

dtype json_dtype::get_canonical_dtype() const
{
    return dtype(this, true);
}


void json_dtype::get_shape(size_t DYND_UNUSED(i),
                intptr_t *DYND_UNUSED(out_shape)) const
{
}

void json_dtype::get_shape(size_t DYND_UNUSED(i),
                intptr_t *DYND_UNUSED(out_shape), const char *DYND_UNUSED(metadata)) const
{
}

bool json_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.get_type_id() == json_type_id) {
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

void json_dtype::get_single_compare_kernel(kernel_instance<compare_operations_t>& out_kernel) const
{
    get_string_comparison_kernel(string_encoding_utf_8, out_kernel);
}

namespace {
   struct string_to_json_assign {
        // Assign from a categorical dtype to some other dtype
        struct auxdata_storage {
            kernel_instance<unary_operation_pair_t> skernel;
        };

        static void single_kernel(char *dst, const char *src, unary_kernel_static_data *extra)
        {
            const json_dtype_metadata *md = reinterpret_cast<const json_dtype_metadata *>(extra->src_metadata);
            json_dtype_data *out_d = reinterpret_cast<json_dtype_data *>(dst);
            auxdata_storage& ad = get_auxiliary_data<auxdata_storage>(extra->auxdata);
            ad.skernel.extra.dst_metadata = extra->dst_metadata;
            ad.skernel.extra.src_metadata = extra->src_metadata;
            // First copy it as a string
            ad.skernel.kernel.single(dst, src, &ad.skernel.extra);
            // Then validate that it's correct JSON
            try {
                validate_json(out_d->begin, out_d->end);
            } catch(const std::exception&) {
                // Free the memory allocated for the output json data
                memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
                allocator->allocate(md->blockref, 0, 1, &out_d->begin, &out_d->end);
            }
        }
    };

   struct json_to_string_assign {
        // Assign from a categorical dtype to some other dtype
        struct auxdata_storage {
            dtype dst_string_dtype;
            assign_error_mode errmode;
        };

        static void single_kernel(char *dst, const char *src, unary_kernel_static_data *extra)
        {
            const json_dtype_data *src_d = reinterpret_cast<const json_dtype_data *>(src);
            auxdata_storage& ad = get_auxiliary_data<auxdata_storage>(extra->auxdata);
            const base_string_dtype *bsd = static_cast<const base_string_dtype *>(ad.dst_string_dtype.extended());
            bsd->set_utf8_string(extra->dst_metadata, dst, ad.errmode, src_d->begin, src_d->end);
        }
    };
} // anonymous namespace



void json_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel) const
{
    if (this == dst_dt.extended()) {
        switch (src_dt.get_type_id()) {
            case json_type_id: {
                // Assume the input is valid JSON when copying from json to json types
                get_blockref_string_assignment_kernel(string_encoding_utf_8, string_encoding_utf_8,
                                errmode, out_kernel);
                break;
            }
            case string_type_id: {
                kernel_instance<unary_operation_pair_t> *out_kernel_ptr = NULL;
                if (errmode == assign_error_none) {
                    out_kernel_ptr = &out_kernel;
                } else {
                    out_kernel.kernel = unary_operation_pair_t(string_to_json_assign::single_kernel,
                                    NULL);
                    make_auxiliary_data<string_to_json_assign::auxdata_storage>(out_kernel.extra.auxdata);
                    string_to_json_assign::auxdata_storage& ad =
                                out_kernel.extra.auxdata.get<string_to_json_assign::auxdata_storage>();
                    out_kernel_ptr = &ad.skernel;
                }
                const base_string_dtype *src_fs = static_cast<const base_string_dtype *>(src_dt.extended());
                get_blockref_string_assignment_kernel(string_encoding_utf_8,
                                src_fs->get_encoding(), errmode, *out_kernel_ptr);
                break;
            }
            case fixedstring_type_id: {
                kernel_instance<unary_operation_pair_t> *out_kernel_ptr = NULL;
                if (errmode == assign_error_none) {
                    out_kernel_ptr = &out_kernel;
                } else {
                    out_kernel.kernel = unary_operation_pair_t(string_to_json_assign::single_kernel,
                                    NULL);
                    make_auxiliary_data<string_to_json_assign::auxdata_storage>(out_kernel.extra.auxdata);
                    string_to_json_assign::auxdata_storage& ad =
                                out_kernel.extra.auxdata.get<string_to_json_assign::auxdata_storage>();
                    out_kernel_ptr = &ad.skernel;
                }
                const base_string_dtype *src_fs = static_cast<const base_string_dtype *>(src_dt.extended());
                get_fixedstring_to_blockref_string_assignment_kernel(string_encoding_utf_8,
                                        src_fs->get_data_size(), src_fs->get_encoding(),
                                        errmode, *out_kernel_ptr);
                break;
            }
            default: {
                if (!src_dt.is_builtin()) {
                    src_dt.extended()->get_dtype_assignment_kernel(dst_dt, src_dt, errmode, out_kernel);
                } else {
                    get_builtin_to_string_assignment_kernel(dst_dt, src_dt.get_type_id(), errmode, out_kernel);
                }
                break;
            }
        }
    } else {
        if (dst_dt.is_builtin()) {
            get_string_to_builtin_assignment_kernel(dst_dt.get_type_id(), src_dt, errmode, out_kernel);
        } else if (dst_dt.get_kind() == string_kind) {
            out_kernel.kernel = unary_operation_pair_t(json_to_string_assign::single_kernel,
                            NULL);
            make_auxiliary_data<json_to_string_assign::auxdata_storage>(out_kernel.extra.auxdata);
            json_to_string_assign::auxdata_storage& ad =
                        out_kernel.extra.auxdata.get<json_to_string_assign::auxdata_storage>();
            ad.dst_string_dtype = dst_dt;
            ad.errmode = errmode;
        } else {
            stringstream ss;
            ss << "assignment from " << src_dt << " to " << dst_dt << " is not implemented yet";
            throw runtime_error(ss.str());
        }
    }
}


bool json_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else {
        return rhs.get_type_id() == json_type_id;
    }
}

size_t json_dtype::get_metadata_size() const
{
    return sizeof(json_dtype_metadata);
}

void json_dtype::metadata_default_construct(char *metadata, int DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const
{
    // Simply allocate a POD memory block
    json_dtype_metadata *md = reinterpret_cast<json_dtype_metadata *>(metadata);
    md->blockref = make_pod_memory_block().release();
}

void json_dtype::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    // Copy the blockref, switching it to the embedded_reference if necessary
    const json_dtype_metadata *src_md = reinterpret_cast<const json_dtype_metadata *>(src_metadata);
    json_dtype_metadata *dst_md = reinterpret_cast<json_dtype_metadata *>(dst_metadata);
    dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
    memory_block_incref(dst_md->blockref);
}

void json_dtype::metadata_reset_buffers(char *DYND_UNUSED(metadata)) const
{
    throw runtime_error("TODO implement json_dtype::metadata_reset_buffers");
}

void json_dtype::metadata_finalize_buffers(char *metadata) const
{
    json_dtype_metadata *md = reinterpret_cast<json_dtype_metadata *>(metadata);
    if (md->blockref != NULL) {
        // Finalize the memory block
        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        if (allocator != NULL) {
            allocator->finalize(md->blockref);
        }
    }
}

void json_dtype::metadata_destruct(char *metadata) const
{
    json_dtype_metadata *md = reinterpret_cast<json_dtype_metadata *>(metadata);
    if (md->blockref) {
        memory_block_decref(md->blockref);
    }
}

void json_dtype::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    const json_dtype_metadata *md = reinterpret_cast<const json_dtype_metadata *>(metadata);
    o << indent << "json metadata\n";
    memory_block_debug_print(md->blockref, o, indent + " ");
}
