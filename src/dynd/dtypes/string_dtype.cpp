//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/single_compare_kernel_instance.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/exceptions.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

string_dtype::string_dtype(string_encoding_t encoding)
    : m_encoding(encoding)
{
    switch (encoding) {
        case string_encoding_ascii:
        case string_encoding_ucs_2:
        case string_encoding_utf_8:
        case string_encoding_utf_16:
        case string_encoding_utf_32:
            break;
        default:
            throw runtime_error("Unrecognized string encoding in string dtype constructor");
    }
}

void string_dtype::get_string_range(const char **out_begin, const char**out_end,
                const char *data, const char *DYND_UNUSED(metadata)) const
{
    *out_begin = reinterpret_cast<const char * const *>(data)[0];
    *out_end = reinterpret_cast<const char * const *>(data)[1];
}

void string_dtype::print_element(std::ostream& o, const char *DYND_UNUSED(metadata), const char *data) const
{
    uint32_t cp;
    next_unicode_codepoint_t next_fn;
    next_fn = get_next_unicode_codepoint_function(m_encoding, assign_error_none);
    const char *begin = reinterpret_cast<const char * const *>(data)[0];
    const char *end = reinterpret_cast<const char * const *>(data)[1];

    // Print as an escaped string
    o << "\"";
    while (begin < end) {
        cp = next_fn(begin, end);
        print_escaped_unicode_codepoint(o, cp);
    }
    o << "\"";
}

void string_dtype::print_dtype(std::ostream& o) const {

    o << "string<" << m_encoding << ">";

}

dtype string_dtype::apply_linear_index(int nindices, const irange *DYND_UNUSED(indices),
                int current_i, const dtype& DYND_UNUSED(root_dt)) const
{
    if (nindices == 0) {
        return dtype(this, true);
        /* TODO:
    } else if (nindices == 1) {
        if (indices->step() == 0) {
            // Return a fixedstring dtype, since it's always one character.
            // If the string encoding is variable-length switch to UTF32 so that the result can always
            // store a single character.
            return make_fixedstring_dtype(is_variable_length_string_encoding(m_encoding) ? string_encoding_utf_32 : m_encoding, 1);
        } else {
            return dtype(this, true);
        }
        */
    } else {
        throw too_many_indices(nindices, current_i + 1);
    }
}

intptr_t string_dtype::apply_linear_index(int DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
                char *DYND_UNUSED(data), const char *metadata,
                const dtype& DYND_UNUSED(result_dtype), char *out_metadata,
                int DYND_UNUSED(current_i), const dtype& DYND_UNUSED(root_dt)) const
{
    const string_dtype_metadata *md = reinterpret_cast<const string_dtype_metadata *>(metadata);
    string_dtype_metadata *out_md = reinterpret_cast<string_dtype_metadata *>(out_metadata);
    // Just copy the blockref
    out_md->blockref = md->blockref;
    memory_block_incref(out_md->blockref);
    return 0;
}

dtype string_dtype::get_canonical_dtype() const
{
    return dtype(this, true);
}


void string_dtype::get_shape(int DYND_UNUSED(i), std::vector<intptr_t>& DYND_UNUSED(out_shape)) const
{
}

bool string_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.get_kind() == string_kind) {
            // If the source is a string, only the encoding matters because the dest is variable sized
            const extended_string_dtype *src_esd = static_cast<const extended_string_dtype*>(src_dt.extended());
            string_encoding_t src_encoding = src_esd->get_encoding();
            switch (m_encoding) {
                case string_encoding_ascii:
                    return src_encoding == string_encoding_ascii;
                case string_encoding_ucs_2:
                    return src_encoding == string_encoding_ascii ||
                            src_encoding == string_encoding_ucs_2;
                case string_encoding_utf_8:
                case string_encoding_utf_16:
                case string_encoding_utf_32:
                    return true;
                default:
                    return false;
            }
        } else if (src_dt.extended() != NULL) {
            return src_dt.extended()->is_lossless_assignment(dst_dt, src_dt);
        } else {
            return false;
        }
    } else {
        return false;
    }
}

void string_dtype::get_single_compare_kernel(single_compare_kernel_instance& DYND_UNUSED(out_kernel)) const {
    throw std::runtime_error("string_dtype::get_single_compare_kernel not supported yet");
}

void string_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel) const
{
    if (this == dst_dt.extended()) {
        switch (src_dt.get_type_id()) {
            case string_type_id: {
                const string_dtype *src_fs = static_cast<const string_dtype *>(src_dt.extended());
                get_blockref_string_assignment_kernel(m_encoding, src_fs->m_encoding, errmode, out_kernel);
                break;
            }
            case fixedstring_type_id: {
                const extended_string_dtype *src_fs = static_cast<const extended_string_dtype *>(src_dt.extended());
                get_fixedstring_to_blockref_string_assignment_kernel(m_encoding,
                                        src_fs->get_element_size(), src_fs->get_encoding(),
                                        errmode, out_kernel);
                break;
            }
            default: {
                if (src_dt.extended()) {
                    src_dt.extended()->get_dtype_assignment_kernel(dst_dt, src_dt, errmode, out_kernel);
                } else {
                    stringstream ss;
                    ss << "assignment from " << src_dt << " to " << dst_dt << " is not implemented yet";
                    throw runtime_error(ss.str());
                }
                break;
            }
        }
    } else {
        stringstream ss;
        ss << "assignment from " << src_dt << " to " << dst_dt << " is not implemented yet";
        throw runtime_error(ss.str());
    }
}


bool string_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != string_type_id) {
        return false;
    } else {
        const string_dtype *dt = static_cast<const string_dtype*>(&rhs);
        return m_encoding == dt->m_encoding;
    }
}

void string_dtype::prepare_kernel_auxdata(const char *metadata, AuxDataBase *auxdata) const
{
    const string_dtype_metadata *md = reinterpret_cast<const string_dtype_metadata *>(metadata);
    auxdata = reinterpret_cast<AuxDataBase *>(reinterpret_cast<uintptr_t>(auxdata)&~1);
    if (auxdata->kernel_api) {
        auxdata->kernel_api->set_dst_memory_block(auxdata, md->blockref);
    }
}

size_t string_dtype::get_metadata_size() const
{
    return sizeof(string_dtype_metadata);
}

void string_dtype::metadata_default_construct(char *metadata, int DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const
{
    // Simply allocate a POD memory block
    string_dtype_metadata *md = reinterpret_cast<string_dtype_metadata *>(metadata);
    md->blockref = make_pod_memory_block().release();
}

void string_dtype::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    // Copy the blockref, switching it to the embedded_reference if necessary
    const string_dtype_metadata *src_md = reinterpret_cast<const string_dtype_metadata *>(src_metadata);
    string_dtype_metadata *dst_md = reinterpret_cast<string_dtype_metadata *>(dst_metadata);
    dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
    memory_block_incref(dst_md->blockref);
}

void string_dtype::metadata_destruct(char *metadata) const
{
    string_dtype_metadata *md = reinterpret_cast<string_dtype_metadata *>(metadata);
    if (md->blockref) {
        memory_block_decref(md->blockref);
    }
}

void string_dtype::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    const string_dtype_metadata *md = reinterpret_cast<const string_dtype_metadata *>(metadata);
    o << indent << "string metadata\n";
    memory_block_debug_print(md->blockref, o, indent + " ");
}
