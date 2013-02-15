//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>
#include <cctype>

#include <dynd/dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/string_numeric_assignment_kernels.hpp>
#include "single_assigner_builtin.hpp"

using namespace std;
using namespace dynd;

// Trim taken from boost string algorithms library
template< typename ForwardIteratorT>
inline ForwardIteratorT trim_begin( 
    ForwardIteratorT InBegin, 
    ForwardIteratorT InEnd )
{
    ForwardIteratorT It=InBegin;
    for(; It!=InEnd; ++It )
    {
        if (!isspace(*It))
            return It;
    }

    return It;
}
template< typename ForwardIteratorT>
inline ForwardIteratorT trim_end( 
    ForwardIteratorT InBegin, 
    ForwardIteratorT InEnd )
{
    for( ForwardIteratorT It=InEnd; It!=InBegin;  )
    {
        if ( !isspace(*(--It)) )
            return ++It;
    }

    return InBegin;
}
template<typename SequenceT>
inline void trim_left_if(SequenceT& Input)
{
    Input.erase( 
        Input.begin(),
        trim_begin( 
            Input.begin(), 
            Input.end() )
        );
}
template<typename SequenceT>
inline void trim_right_if(SequenceT& Input)
{
    Input.erase(
        trim_end( 
            Input.begin(), 
            Input.end() ),
        Input.end()
        );
}
template<typename SequenceT>
inline void trim(SequenceT& Input)
{
    trim_right_if( Input );
    trim_left_if( Input );
}
// End trim taken from boost string algorithms
void to_lower(std::string& s)
{
    for (size_t i = 0, i_end = s.size(); i != i_end; ++i) {
        s[i] = tolower(s[i]);
    }
}

namespace {
    struct string_to_builtin_auxdata {
        dtype src_string_dt;
        assign_error_mode errmode;
    };

    struct string_to_builtin_kernel_extra {
        typedef string_to_builtin_kernel_extra extra_type;

        kernel_data_prefix base;
        const base_string_dtype *src_string_dt;
        assign_error_mode errmode;
        const char *src_metadata;

        static void destruct(kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            if (e->src_string_dt) {
                base_dtype_decref(e->src_string_dt);
            }
        }
    };
} // anonymous namespace

/////////////////////////////////////////
// builtin to string assignment

static void raise_string_cast_error(const dtype& dst_dt, const dtype& string_dt, const char *metadata, const char *data)
{
    stringstream ss;
    ss << "cannot cast string ";
    string_dt.print_data(ss, metadata, data);
    ss << " to " << dst_dt;
    throw runtime_error(ss.str());
}

static void raise_string_cast_overflow_error(const dtype& dst_dt, const dtype& string_dt, const char *metadata, const char *data)
{
    stringstream ss;
    ss << "overflow converting string ";
    string_dt.print_data(ss, metadata, data);
    ss << " to " << dst_dt;
    throw runtime_error(ss.str());
}

static void string_to_bool_single(char *dst, const char *src,
                        kernel_data_prefix *extra)
{
    string_to_builtin_kernel_extra *e = reinterpret_cast<string_to_builtin_kernel_extra *>(extra);
    // Get the string from the source
    string s = e->src_string_dt->get_utf8_string(e->src_metadata, src, e->errmode);
    trim(s);
    to_lower(s);
    if (e->errmode == assign_error_none) {
        if (s.empty() || s == "0" || s == "false" || s == "no" || s == "off" || s == "f" || s == "n") {
            *dst = 0;
        } else {
            *dst = 1;
        }
    } else {
        if (s == "0" || s == "false" || s == "no" || s == "off" || s == "f" || s == "n") {
            *dst = 0;
        } else if (s == "1" || s == "true" || s == "yes" || s == "on" || s == "t" || s == "y") {
            *dst = 1;
        } else {
            raise_string_cast_error(make_dtype<dynd_bool>(), dtype(e->src_string_dt, true), e->src_metadata, src);
        }
    }
}

static uint64_t parse_uint64_noerror(const std::string& s)
{
    uint64_t result = 0;
    size_t pos = 0, end = s.size();
    while (pos < end) {
        char c = s[pos];
        if ('0' <= c && c <= '9') {
            result = (result * 10) + (c - '0');
        } else {
            break;
        }
        ++pos;
    }
    return result;
}

static uint64_t parse_uint64(const std::string& s, bool& out_overflow, bool& out_badparse)
{
    uint64_t result = 0, prev_result = 0;
    size_t pos = 0, end = s.size();
    while (pos < end) {
        char c = s[pos];
        if ('0' <= c && c <= '9') {
            result = (result * 10) + (c - '0');
            if (result < prev_result) {
                out_overflow = true;
            }
        } else {
            out_badparse = true;
            break;
        }
        ++pos;
        prev_result = result;
    }
    return result;
}

template <class T> struct overflow_check;
template <> struct overflow_check<int8_t> { inline static bool is_overflow(uint64_t value, bool negative) {
    return (value&~0x7fULL) != 0 && !(negative && value == 0x80ULL);
}};
template <> struct overflow_check<int16_t> { inline static bool is_overflow(uint64_t value, bool negative) {
    return (value&~0x7fffULL) != 0 && !(negative && value == 0x8000ULL);
}};
template <> struct overflow_check<int32_t> { inline static bool is_overflow(uint64_t value, bool negative) {
    return (value&~0x7fffffffULL) != 0 && !(negative && value == 0x80000000ULL);
}};
template <> struct overflow_check<int64_t> { inline static bool is_overflow(uint64_t value, bool negative) {
    return (value&~0x7fffffffffffffffULL) != 0 && !(negative && value == 0x8000000000000000ULL);
}};
template <> struct overflow_check<uint8_t> { inline static bool is_overflow(uint64_t value) {
    return (value&~0xffULL) != 0;
}};
template <> struct overflow_check<uint16_t> { inline static bool is_overflow(uint64_t value) {
    return (value&~0xffffULL) != 0;
}};
template <> struct overflow_check<uint32_t> { inline static bool is_overflow(uint64_t value) {
    return (value&~0xffffffffULL) != 0;
}};
template <> struct overflow_check<uint64_t> { inline static bool is_overflow(uint64_t DYND_UNUSED(value)) {
    return false;
}};

namespace { template<typename T> struct string_to_int {
    static void single(char *dst, const char *src, 
                        kernel_data_prefix *extra)
    {
        string_to_builtin_kernel_extra *e = reinterpret_cast<string_to_builtin_kernel_extra *>(extra);
        string s = e->src_string_dt->get_utf8_string(e->src_metadata, src, e->errmode);
        trim(s);
        bool negative = false;
        if (!s.empty() && s[0] == '-') {
            s.erase(0, 1);
            negative = true;
        }
        T result;
        if (e->errmode == assign_error_none) {
            uint64_t value = parse_uint64_noerror(s);
            result = negative ? static_cast<T>(-static_cast<int64_t>(value)) : static_cast<T>(value);
        } else {
            bool overflow = false, badparse = false;
            uint64_t value = parse_uint64(s, overflow, badparse);
            if (badparse) {
                raise_string_cast_error(make_dtype<T>(), dtype(e->src_string_dt, true), e->src_metadata, src);
            } else if (overflow || overflow_check<T>::is_overflow(value, negative)) {
                raise_string_cast_overflow_error(make_dtype<T>(), dtype(e->src_string_dt, true), e->src_metadata, src);
            }
            result = negative ? static_cast<T>(-static_cast<int64_t>(value)) : static_cast<T>(value);
        }
        *reinterpret_cast<T *>(dst) = result;
    }
};}

namespace { template<typename T> struct string_to_uint {
    static void single(char *dst, const char *src, 
                        kernel_data_prefix *extra)
    {
        string_to_builtin_kernel_extra *e = reinterpret_cast<string_to_builtin_kernel_extra *>(extra);
        string s = e->src_string_dt->get_utf8_string(e->src_metadata, src, e->errmode);
        trim(s);
        bool negative = false;
        if (!s.empty() && s[0] == '-') {
            s.erase(0, 1);
            negative = true;
        }
        T result;
        if (e->errmode == assign_error_none) {
            uint64_t value = parse_uint64_noerror(s);
            result = negative ? static_cast<T>(0) : static_cast<T>(value);
        } else {
            bool overflow = false, badparse = false;
            uint64_t value = parse_uint64(s, overflow, badparse);
            if (badparse) {
                raise_string_cast_error(make_dtype<T>(), dtype(e->src_string_dt, true), e->src_metadata, src);
            } else if (negative || overflow || overflow_check<T>::is_overflow(value)) {
                raise_string_cast_overflow_error(make_dtype<T>(), dtype(e->src_string_dt, true), e->src_metadata, src);
            }
            result = static_cast<T>(value);
        }
        *reinterpret_cast<T *>(dst) = result;
    }
};}

static void string_to_float32_single(char *dst, const char *src,
                        kernel_data_prefix *extra)
{
    string_to_builtin_kernel_extra *e = reinterpret_cast<string_to_builtin_kernel_extra *>(extra);
    // Get the string from the source
    string s = e->src_string_dt->get_utf8_string(e->src_metadata, src,e->errmode);
    trim(s);
    to_lower(s);
    // Handle special values
    if (s == "nan" || s == "1.#qnan") {
        *reinterpret_cast<uint32_t *>(dst) = 0x7fc00000;
        return;
    } else if (s == "-nan" || s == "-1.#ind") {
        *reinterpret_cast<uint32_t *>(dst) = 0xffc00000;
        return;
    } else if (s == "inf" || s == "infinity" || s == "1.#inf") {
        *reinterpret_cast<uint32_t *>(dst) = 0x7f800000;
        return;
    } else if (s == "-inf" || s == "-infinity" || s == "-1.#inf") {
        *reinterpret_cast<uint32_t *>(dst) = 0xff800000;
        return;
    } else if (s == "na") {
        // A 32-bit version of R's special NA NaN
        *reinterpret_cast<uint32_t *>(dst) = 0x7f8007a2;
        return;
    }
    char *end_ptr;
    // TODO: use a different parsing code that's guaranteed to round correctly in a cross-platform fashion
    double value = strtod(s.c_str(), &end_ptr);
    if (e->errmode != assign_error_none && (size_t)(end_ptr - s.c_str()) != s.size()) {
        raise_string_cast_error(make_dtype<float>(), dtype(e->src_string_dt, true), e->src_metadata, src);
    } else {
        // Assign double -> float according to the error mode
        switch (e->errmode) {
            case assign_error_none:
                single_assigner_builtin<float, double, assign_error_none>::assign(
                                reinterpret_cast<float *>(dst), &value, NULL);
                break;
            case assign_error_overflow:
                single_assigner_builtin<float, double, assign_error_overflow>::assign(
                                reinterpret_cast<float *>(dst), &value, NULL);
                break;
            case assign_error_fractional:
                single_assigner_builtin<float, double, assign_error_fractional>::assign(
                                reinterpret_cast<float *>(dst), &value, NULL);
                break;
            case assign_error_inexact:
                single_assigner_builtin<float, double, assign_error_inexact>::assign(
                                reinterpret_cast<float *>(dst), &value, NULL);
                break;
            default:
                single_assigner_builtin<float, double, assign_error_fractional>::assign(
                                reinterpret_cast<float *>(dst), &value, NULL);
                break;
        }
    }
}

static void string_to_float64_single(char *dst, const char *src,
                        kernel_data_prefix *extra)
{
    string_to_builtin_kernel_extra *e = reinterpret_cast<string_to_builtin_kernel_extra *>(extra);
    // Get the string from the source
    string s = e->src_string_dt->get_utf8_string(e->src_metadata, src, e->errmode);
    trim(s);
    to_lower(s);
    // Handle special values
    if (s == "nan" || s == "1.#qnan") {
        *reinterpret_cast<uint64_t *>(dst) = 0x7ff8000000000000ULL;
        return;
    } else if (s == "-nan" || s == "-1.#ind") {
        *reinterpret_cast<uint64_t *>(dst) = 0xfff8000000000000ULL;
        return;
    } else if (s == "inf" || s == "infinity" || s == "1.#inf") {
        *reinterpret_cast<uint64_t *>(dst) = 0x7ff0000000000000ULL;
        return;
    } else if (s == "-inf" || s == "-infinity" || s == "-1.#inf") {
        *reinterpret_cast<uint64_t *>(dst) = 0xfff0000000000000ULL;
        return;
    } else if (s == "na") {
        // R's special NA NaN
        *reinterpret_cast<uint64_t *>(dst) = 0x7ff00000000007a2ULL;
        return;
    }
    char *end_ptr;
    // TODO: use a different parsing code that's guaranteed to round correctly in a cross-platform fashion
    double value = strtod(s.c_str(), &end_ptr);
    if (e->errmode != assign_error_none && (size_t)(end_ptr - s.c_str()) != s.size()) {
        raise_string_cast_error(make_dtype<double>(), dtype(e->src_string_dt, true), e->src_metadata, src);
    } else {
        *reinterpret_cast<double *>(dst) = value;
    }
}

static void string_to_complex_float32_single(char *DYND_UNUSED(dst), const char *DYND_UNUSED(src),
                kernel_data_prefix *DYND_UNUSED(extra))
{
    throw std::runtime_error("TODO: implement string_to_complex_float32_single");
}

static void string_to_complex_float64_single(char *DYND_UNUSED(dst), const char *DYND_UNUSED(src),
                kernel_data_prefix *DYND_UNUSED(extra))
{
    throw std::runtime_error("TODO: implement string_to_complex_float64_single");
}

static unary_single_operation_t static_string_to_builtin_kernels[builtin_type_id_count-2] = {
        &string_to_bool_single,
        &string_to_int<int8_t>::single,
        &string_to_int<int16_t>::single,
        &string_to_int<int32_t>::single,
        &string_to_int<int64_t>::single,
        &string_to_uint<uint8_t>::single,
        &string_to_uint<uint16_t>::single,
        &string_to_uint<uint32_t>::single,
        &string_to_uint<uint64_t>::single,
        &string_to_float32_single,
        &string_to_float64_single,
        &string_to_complex_float32_single,
        &string_to_complex_float64_single
    };

size_t dynd::make_string_to_builtin_assignment_kernel(
                assignment_kernel *out,
                size_t offset_out,
                type_id_t dst_type_id,
                const dtype& src_string_dt, const char *src_metadata,
                assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    if (src_string_dt.get_kind() != string_kind) {
        stringstream ss;
        ss << "make_string_to_builtin_assignment_kernel: source dtype " << src_string_dt << " is not a string dtype";
        throw runtime_error(ss.str());
    }

    if (dst_type_id >= bool_type_id && dst_type_id <= complex_float64_type_id) {
        out->ensure_capacity_leaf(offset_out + sizeof(string_to_builtin_kernel_extra));
        string_to_builtin_kernel_extra *e = out->get_at<string_to_builtin_kernel_extra>(offset_out);
        e->base.set_function<unary_single_operation_t>(static_string_to_builtin_kernels[dst_type_id-bool_type_id]);
        e->base.destructor = string_to_builtin_kernel_extra::destruct;
        // The kernel data owns this reference
        e->src_string_dt = static_cast<const base_string_dtype *>(dtype(src_string_dt).release());
        e->errmode = errmode;
        e->src_metadata = src_metadata;
        return offset_out + sizeof(string_to_builtin_kernel_extra);
    } else {
        stringstream ss;
        ss << "make_string_to_builtin_assignment_kernel: destination type id " << dst_type_id << " is not builtin";
        throw runtime_error(ss.str());
    }
}

/////////////////////////////////////////
// string to builtin assignment

namespace {
    struct builtin_to_string_kernel_extra {
        typedef builtin_to_string_kernel_extra extra_type;

        kernel_data_prefix base;
        const base_string_dtype *dst_string_dt;
        type_id_t src_type_id;
        assign_error_mode errmode;
        const char *dst_metadata;

        static void single(char *dst, const char *src,
                            kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);

            // TODO: There are much faster ways to do this, but it's very generic!
            //       Also, for floating point values, a printing scheme like Python's,
            //       where it prints the shortest string that's guaranteed to parse to
            //       the same float number, would be better.
            stringstream ss;
            dtype(e->src_type_id).print_data(ss, NULL, src);
            e->dst_string_dt->set_utf8_string(e->dst_metadata, dst, e->errmode, ss.str());
        }

        static void destruct(kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            if (e->dst_string_dt) {
                base_dtype_decref(e->dst_string_dt);
            }
        }
    };
} // anonymous namespace

size_t dynd::make_builtin_to_string_assignment_kernel(
                assignment_kernel *out,
                size_t offset_out,
                const dtype& dst_string_dt, const char *dst_metadata,
                type_id_t src_type_id,
                assign_error_mode errmode,
                const eval::eval_context *DYND_UNUSED(ectx))
{
    if (dst_string_dt.get_kind() != string_kind) {
        stringstream ss;
        ss << "make_builtin_to_string_assignment_kernel: destination dtype " << dst_string_dt << " is not a string dtype";
        throw runtime_error(ss.str());
    }

    if (src_type_id >= 0 && src_type_id < builtin_type_id_count) {
        out->ensure_capacity_leaf(offset_out + sizeof(builtin_to_string_kernel_extra));
        builtin_to_string_kernel_extra *e = out->get_at<builtin_to_string_kernel_extra>(offset_out);
        e->base.set_function<unary_single_operation_t>(builtin_to_string_kernel_extra::single);
        e->base.destructor = builtin_to_string_kernel_extra::destruct;
        // The kernel data owns this reference
        e->dst_string_dt = static_cast<const base_string_dtype *>(dtype(dst_string_dt).release());
        e->src_type_id = src_type_id;
        e->errmode = errmode;
        e->dst_metadata = dst_metadata;
        return offset_out + sizeof(builtin_to_string_kernel_extra);
    } else {
        stringstream ss;
        ss << "make_builtin_to_string_assignment_kernel: source type id " << src_type_id << " is not builtin";
        throw runtime_error(ss.str());
    }
}


void dynd::assign_utf8_string_to_builtin(type_id_t dst_type_id, char *dst,
                const char *str_begin, const char *str_end, assign_error_mode errmode)
{
    dtype dt = make_string_dtype(string_encoding_utf_8);
    string_dtype_data d;
    string_dtype_metadata md;
    d.begin = const_cast<char *>(str_begin);
    d.end = const_cast<char *>(str_end);
    md.blockref = NULL;

    assignment_kernel k;
    make_string_to_builtin_assignment_kernel(&k, 0,
                    dst_type_id,
                    dt, reinterpret_cast<const char *>(&md),
                    errmode, &eval::default_eval_context);
    k(dst, reinterpret_cast<const char *>(&d));
}
