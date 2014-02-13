//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/json_formatter.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/json_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

struct output_data {
    char *out_begin, *out_end, *out_capacity_end;
    memory_block_pod_allocator_api *api;
    memory_block_data *blockref;

    void ensure_capacity(intptr_t added_capacity) {
        // If there's not enough space, double the capacity
        if (out_capacity_end - out_end < added_capacity) {
            intptr_t current_size = out_end - out_begin;
            intptr_t new_capacity = 2 * (out_capacity_end - out_begin);
            // Make sure this adds the requested additional capacity
            if (new_capacity < current_size + added_capacity) {
                new_capacity = current_size + added_capacity;
            }
            api->resize(blockref, new_capacity, &out_begin, &out_capacity_end);
            out_end = out_begin + current_size;
        }
    }

    inline void write(char c) {
        ensure_capacity(1);
        *out_end++ = c;
    }

    // Write a literal string
    template<int N>
    inline void write(const char (&str)[N]) {
        ensure_capacity(N - 1);
        memcpy(out_end, str, N - 1);
        out_end += N - 1;
    }

    // Write a std::string
    inline void write(const std::string& s) {
        ensure_capacity(s.size());
        memcpy(out_end, s.data(), s.size());
        out_end += s.size();
    }

    // Write a string-range
    inline void write(const char *begin, const char *end) {
        ensure_capacity(end - begin);
        memcpy(out_end, begin, end - begin);
        out_end += (end - begin);
    }
};

static void format_json(output_data& out, const ndt::type& dt, const char *metadata, const char *data);

static void format_json_bool(output_data& out, const ndt::type& dt, const char *metadata, const char *data)
{
    dynd_bool value = false;
    if (dt.get_type_id() == bool_type_id) {
        value = (*data != 0);
    } else {
        typed_data_assign(ndt::make_type<dynd_bool>(), NULL, reinterpret_cast<char *>(&value), dt, metadata, data);
    }
    if (value) {
        out.write("true");
    } else {
        out.write("false");
    }
}

static void format_json_number(output_data& out, const ndt::type& dt, const char *metadata, const char *data)
{
    stringstream ss;
    dt.print_data(ss, metadata, data);
    out.write(ss.str());
}

static void print_escaped_unicode_codepoint(output_data& out, uint32_t cp, append_unicode_codepoint_t append_fn)
{
    if (cp < 0x80) {
        switch (cp) {
            case '\b':
                out.write("\\b");
                break;
            case '\f':
                out.write("\\f");
                break;
            case '\n':
                out.write("\\n");
                break;
            case '\r':
                out.write("\\r");
                break;
            case '\t':
                out.write("\\t");
                break;
            case '\\':
                out.write("\\\\");
                break;
            case '/':
                out.write("\\/");
                break;
            case '\"':
                out.write("\\\"");
                break;
            default:
                if (cp < 0x20 || cp == 0x7f) {
                    stringstream ss;
                    ss << "\\u";
                    hexadecimal_print(ss, static_cast<uint16_t>(cp));
                    out.write(ss.str());
                } else {
                    out.write(static_cast<char>(cp));
                }
                break;
        }
    } else {
        out.ensure_capacity(16);
        append_fn(cp, out.out_end, out.out_capacity_end);
    }
    // TODO: Could have an ASCII output mode where unicode is always escaped
    /*
    else if (cp < 0x10000) {
        stringstream ss;
        ss << "\\u";
        hexadecimal_print(ss, static_cast<uint16_t>(cp));
        out.write(ss.str());
    } else {
        stringstream ss;
        ss << "\\U";
        hexadecimal_print(ss, static_cast<uint32_t>(cp));
        out.write(ss.str());
    }
    */
}

static void format_json_encoded_string(output_data& out, const char *begin, const char *end, string_encoding_t encoding)
{
    uint32_t cp;
    next_unicode_codepoint_t next_fn;
    append_unicode_codepoint_t append_fn;
    next_fn = get_next_unicode_codepoint_function(encoding, assign_error_none);
    append_fn = get_append_unicode_codepoint_function(string_encoding_utf_8, assign_error_none);
    out.write('\"');
    while (begin < end) {
        cp = next_fn(begin, end);
        print_escaped_unicode_codepoint(out, cp, append_fn);
    }
    out.write('\"');
}

static void format_json_string(output_data& out, const ndt::type& dt, const char *metadata, const char *data)
{
    if (dt.get_type_id() == json_type_id) {
        // Copy the JSON data directly
        const json_type_data *d = reinterpret_cast<const json_type_data *>(data);
        out.write(d->begin, d->end);
    } else {
        const base_string_type *bsd = static_cast<const base_string_type *>(dt.extended());
        string_encoding_t encoding = bsd->get_encoding();
        const char *begin = NULL, *end = NULL;
        bsd->get_string_range(&begin, &end, metadata, data);
        format_json_encoded_string(out, begin, end, encoding);
    }
}

static void format_json_datetime(output_data& out, const ndt::type& dt, const char *metadata, const char *data)
{
    switch (dt.get_type_id()) {
        case date_type_id: {
            stringstream ss;
            dt.print_data(ss, metadata, data);
            string s = ss.str();
            format_json_encoded_string(out, s.data(), s.data() + s.size(), string_encoding_ascii);
            break;
        }
        default: {
            stringstream ss;
            ss << "Formatting dynd type " << dt << " as JSON is not implemented yet";
            throw runtime_error(ss.str());
        }
    }
}

static void format_json_struct(output_data& out, const ndt::type& dt, const char *metadata, const char *data)
{
    const base_struct_type *bsd = static_cast<const base_struct_type *>(dt.extended());
    size_t field_count = bsd->get_field_count();
    const string *field_names = bsd->get_field_names();
    const ndt::type *field_types = bsd->get_field_types();
    const size_t *data_offsets = bsd->get_data_offsets(metadata);
    const size_t *metadata_offsets = bsd->get_metadata_offsets();

    out.write('{');
    for (size_t i = 0; i < field_count; ++i) {
        const string& fname = field_names[i];
        format_json_encoded_string(out, fname.data(), fname.data() + fname.size(), string_encoding_utf_8);
        out.write(':');
        ::format_json(out, field_types[i], metadata + metadata_offsets[i], data + data_offsets[i]);
        if (i != field_count - 1) {
            out.write(',');
        }
    }
    out.write('}');
}

static void format_json_uniform_dim(output_data& out, const ndt::type& dt, const char *metadata, const char *data)
{
    out.write('[');
    switch (dt.get_type_id()) {
        case strided_dim_type_id: {
            const strided_dim_type *sad = static_cast<const strided_dim_type *>(dt.extended());
            const strided_dim_type_metadata *md = reinterpret_cast<const strided_dim_type_metadata *>(metadata);
            ndt::type element_tp = sad->get_element_type();
            intptr_t size = md->size, stride = md->stride;
            metadata += sizeof(strided_dim_type_metadata);
            for (intptr_t i = 0; i < size; ++i) {
                ::format_json(out, element_tp, metadata, data + i * stride);
                if (i != size - 1) {
                    out.write(',');
                }
            }
            break;
        }
        case fixed_dim_type_id: {
            const fixed_dim_type *fad = static_cast<const fixed_dim_type *>(dt.extended());
            ndt::type element_tp = fad->get_element_type();
            intptr_t size = (intptr_t)fad->get_fixed_dim_size(), stride = fad->get_fixed_stride();
            for (intptr_t i = 0; i < size; ++i) {
                ::format_json(out, element_tp, metadata, data + i * stride);
                if (i != size - 1) {
                    out.write(',');
                }
            }
            break;
        }
        case var_dim_type_id: {
            const var_dim_type *vad = static_cast<const var_dim_type *>(dt.extended());
            const var_dim_type_metadata *md = reinterpret_cast<const var_dim_type_metadata *>(metadata);
            const var_dim_type_data *d = reinterpret_cast<const var_dim_type_data *>(data);
            ndt::type element_tp = vad->get_element_type();
            intptr_t size = d->size, stride = md->stride;
            const char *begin = d->begin + md->offset;
            metadata += sizeof(var_dim_type_metadata);
            for (intptr_t i = 0; i < size; ++i) {
                ::format_json(out, element_tp, metadata, begin + i * stride);
                if (i != size - 1) {
                    out.write(',');
                }
            }
            break;
        }
        default: {
            stringstream ss;
            ss << "Formatting dynd type " << dt << " as JSON is not implemented yet";
            throw runtime_error(ss.str());
        }
    }
    out.write(']');
}

static void format_json(output_data& out, const ndt::type& dt, const char *metadata, const char *data)
{
    switch (dt.get_kind()) {
        case bool_kind:
            format_json_bool(out, dt, metadata, data);
            break;
        case int_kind:
        case uint_kind:
        case real_kind:
        case complex_kind:
            format_json_number(out, dt, metadata, data);
            break;
        case string_kind:
            format_json_string(out, dt, metadata, data);
            break;
        case datetime_kind:
            format_json_datetime(out, dt, metadata, data);
            break;
        case struct_kind:
            format_json_struct(out, dt, metadata, data);
            break;
        case uniform_dim_kind:
            format_json_uniform_dim(out, dt, metadata, data);
            break;
        default: {
            stringstream ss;
            ss << "Formatting dynd type " << dt << " as JSON is not implemented yet";
            throw runtime_error(ss.str());
        }
    }
}

nd::array dynd::format_json(const nd::array& n)
{
    // Create a UTF-8 string
    nd::array result = nd::empty(ndt::make_string());

    // Initialize the output with some memory
    output_data out;
    out.blockref = reinterpret_cast<const string_type_metadata *>(result.get_ndo_meta())->blockref;
    out.api = get_memory_block_pod_allocator_api(out.blockref);
    out.api->allocate(out.blockref, 1024, 1, &out.out_begin, &out.out_capacity_end);
    out.out_end = out.out_begin;

    if (!n.get_type().is_expression()) {
        ::format_json(out, n.get_type(), n.get_ndo_meta(), n.get_readonly_originptr());
    } else {
        nd::array tmp = n.eval();
        ::format_json(out, tmp.get_type(), tmp.get_ndo_meta(), tmp.get_readonly_originptr());
    }

    // Shrink the memory to fit, and set the pointers in the output
    string_type_data *d = reinterpret_cast<string_type_data *>(result.get_readwrite_originptr());
    d->begin = out.out_begin;
    d->end = out.out_capacity_end;
    out.api->resize(out.blockref, out.out_end - out.out_begin, &d->begin, &d->end);

    // Finalize processing and mark the result as immutable
    result.get_type().extended()->metadata_finalize_buffers(result.get_ndo_meta());
    result.flag_as_immutable();

    return result;
}
