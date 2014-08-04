//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>
#include <cctype>

#include <dynd/type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/string_numeric_assignment_kernels.hpp>
#include <dynd/parser_util.hpp>
#include "single_assigner_builtin.hpp"

using namespace std;
using namespace dynd;

// Trim taken from boost string algorithms library
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
        ndt::type src_string_tp;
        assign_error_mode errmode;
    };

    struct string_to_builtin_kernel_extra {
        typedef string_to_builtin_kernel_extra extra_type;

        ckernel_prefix base;
        const base_string_type *src_string_tp;
        assign_error_mode errmode;
        const char *src_arrmeta;

        static void destruct(ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            if (e->src_string_tp) {
                base_type_decref(e->src_string_tp);
            }
        }
    };
} // anonymous namespace

/////////////////////////////////////////
// builtin to string assignment

static void raise_string_cast_error(const ndt::type &dst_tp,
                                    const ndt::type &string_tp,
                                    const char *arrmeta, const char *data)
{
    stringstream ss;
    ss << "cannot cast string ";
    string_tp.print_data(ss, arrmeta, data);
    ss << " to " << dst_tp;
    throw invalid_argument(ss.str());
}

static void raise_string_cast_overflow_error(const ndt::type &dst_tp,
                                             const ndt::type &string_tp,
                                             const char *arrmeta,
                                             const char *data)
{
    stringstream ss;
    ss << "overflow converting string ";
    string_tp.print_data(ss, arrmeta, data);
    ss << " to " << dst_tp;
    throw overflow_error(ss.str());
}

static void string_to_bool_single(char *dst, const char *const *src,
                                  ckernel_prefix *extra)
{
    string_to_builtin_kernel_extra *e =
        reinterpret_cast<string_to_builtin_kernel_extra *>(extra);
    // Get the string from the source
    string s =
        e->src_string_tp->get_utf8_string(e->src_arrmeta, src[0], e->errmode);
    trim(s);
    parse::string_to_bool(dst, s.data(), s.data() + s.size(), false, e->errmode);
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
template <> struct overflow_check<dynd_int128> { inline static bool is_overflow(dynd_uint128 value, bool negative) {
        return (value.m_hi & ~0x7fffffffffffffffULL) != 0 &&
               !(negative && value.m_hi == 0x8000000000000000ULL &&
                 value.m_lo == 0ULL);
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
    static void single(char *dst, const char *const *src, 
                        ckernel_prefix *extra)
    {
        string_to_builtin_kernel_extra *e = reinterpret_cast<string_to_builtin_kernel_extra *>(extra);
        string s = e->src_string_tp->get_utf8_string(e->src_arrmeta, src[0], e->errmode);
        trim(s);
        bool negative = false;
        if (!s.empty() && s[0] == '-') {
            s.erase(0, 1);
            negative = true;
        }
        T result;
        if (e->errmode == assign_error_nocheck) {
            uint64_t value = parse::unchecked_string_to_uint64(
                s.data(), s.data() + s.size());
            result = negative ? static_cast<T>(-static_cast<int64_t>(value))
                              : static_cast<T>(value);
        } else {
            bool overflow = false, badparse = false;
            uint64_t value = parse::checked_string_to_uint64(
                s.data(), s.data() + s.size(), overflow, badparse);
            if (badparse) {
                raise_string_cast_error(ndt::make_type<T>(),
                                        ndt::type(e->src_string_tp, true),
                                        e->src_arrmeta, src[0]);
            } else if (overflow ||
                       overflow_check<T>::is_overflow(value, negative)) {
                raise_string_cast_overflow_error(
                    ndt::make_type<T>(), ndt::type(e->src_string_tp, true),
                    e->src_arrmeta, src[0]);
            }
            result = negative ? static_cast<T>(-static_cast<int64_t>(value))
                              : static_cast<T>(value);
        }
        *reinterpret_cast<T *>(dst) = result;
    }
};}

namespace { template<typename T> struct string_to_uint {
    static void single(char *dst, const char *const *src, 
                        ckernel_prefix *extra)
    {
        string_to_builtin_kernel_extra *e = reinterpret_cast<string_to_builtin_kernel_extra *>(extra);
        string s = e->src_string_tp->get_utf8_string(e->src_arrmeta, src[0], e->errmode);
        trim(s);
        bool negative = false;
        if (!s.empty() && s[0] == '-') {
            s.erase(0, 1);
            negative = true;
        }
        T result;
        if (e->errmode == assign_error_nocheck) {
            uint64_t value = parse::unchecked_string_to_uint64(
                s.data(), s.data() + s.size());
            result = negative ? static_cast<T>(0) : static_cast<T>(value);
        } else {
            bool overflow = false, badparse = false;
            uint64_t value = parse::checked_string_to_uint64(
                s.data(), s.data() + s.size(), overflow, badparse);
            if (badparse) {
                raise_string_cast_error(ndt::make_type<T>(),
                                        ndt::type(e->src_string_tp, true),
                                        e->src_arrmeta, src[0]);
            } else if (overflow || (negative && value != 0) ||
                       overflow_check<T>::is_overflow(value)) {
                raise_string_cast_overflow_error(
                    ndt::make_type<T>(), ndt::type(e->src_string_tp, true),
                    e->src_arrmeta, src[0]);
            }
            result = static_cast<T>(value);
        }
        *reinterpret_cast<T *>(dst) = result;
    }
};}

static void string_to_int128_single(char *dst, const char *const *src,
                                    ckernel_prefix *extra)
{
    string_to_builtin_kernel_extra *e =
        reinterpret_cast<string_to_builtin_kernel_extra *>(extra);
    string s =
        e->src_string_tp->get_utf8_string(e->src_arrmeta, src[0], e->errmode);
    trim(s);
    bool negative = false;
    if (!s.empty() && s[0] == '-') {
        s.erase(0, 1);
        negative = true;
    }
    dynd_int128 result;
    if (e->errmode == assign_error_nocheck) {
        dynd_uint128 value =
            parse::unchecked_string_to_uint128(s.data(), s.data() + s.size());
        result = negative ? static_cast<dynd_int128>(0) : static_cast<dynd_int128>(value);
    } else {
        bool overflow = false, badparse = false;
        dynd_uint128 value = parse::checked_string_to_uint128(
            s.data(), s.data() + s.size(), overflow, badparse);
        if (badparse) {
            raise_string_cast_error(ndt::make_type<dynd_int128>(),
                                    ndt::type(e->src_string_tp, true),
                                    e->src_arrmeta, src[0]);
        } else if (overflow ||
                   overflow_check<dynd_int128>::is_overflow(value, negative)) {
            raise_string_cast_overflow_error(ndt::make_type<dynd_int128>(),
                                             ndt::type(e->src_string_tp, true),
                                             e->src_arrmeta, src[0]);
        }
        result = negative ? -static_cast<dynd_int128>(value)
                          : static_cast<dynd_int128>(value);
    }
    *reinterpret_cast<dynd_int128 *>(dst) = result;
}

static void string_to_uint128_single(char *dst, const char *const *src,
                                     ckernel_prefix *extra)
{
    string_to_builtin_kernel_extra *e =
        reinterpret_cast<string_to_builtin_kernel_extra *>(extra);
    string s =
        e->src_string_tp->get_utf8_string(e->src_arrmeta, src[0], e->errmode);
    trim(s);
    bool negative = false;
    if (!s.empty() && s[0] == '-') {
        s.erase(0, 1);
        negative = true;
    }
    dynd_int128 result;
    if (e->errmode == assign_error_nocheck) {
        result =
            parse::unchecked_string_to_uint128(s.data(), s.data() + s.size());
    } else {
        bool overflow = false, badparse = false;
        result = parse::checked_string_to_uint128(
            s.data(), s.data() + s.size(), overflow, badparse);
        if (badparse) {
            raise_string_cast_error(ndt::make_type<dynd_int128>(),
                                    ndt::type(e->src_string_tp, true),
                                    e->src_arrmeta, src[0]);
        } else if (overflow || (negative && result != 0)) {
            raise_string_cast_overflow_error(ndt::make_type<dynd_uint128>(),
                                             ndt::type(e->src_string_tp, true),
                                             e->src_arrmeta, src[0]);
        }
    }
    *reinterpret_cast<dynd_uint128 *>(dst) = result;
}

static void string_to_float32_single(char *dst, const char *const *src,
                                     ckernel_prefix *extra)
{
    string_to_builtin_kernel_extra *e =
        reinterpret_cast<string_to_builtin_kernel_extra *>(extra);
    // Get the string from the source
    string s =
        e->src_string_tp->get_utf8_string(e->src_arrmeta, src[0], e->errmode);
    trim(s);
    double value = parse::checked_string_to_float64(
        s.data(), s.data() + s.size(), e->errmode);
    // Assign double -> float according to the error mode
    switch (e->errmode) {
        case assign_error_nocheck:
            single_assigner_builtin<float, double, assign_error_nocheck>::assign(
                reinterpret_cast<float *>(dst), &value);
            break;
        case assign_error_overflow:
            single_assigner_builtin<float, double, assign_error_overflow>::
                assign(reinterpret_cast<float *>(dst), &value);
            break;
        case assign_error_fractional:
            single_assigner_builtin<float, double, assign_error_fractional>::
                assign(reinterpret_cast<float *>(dst), &value);
            break;
        case assign_error_inexact:
            single_assigner_builtin<float, double, assign_error_inexact>::
                assign(reinterpret_cast<float *>(dst), &value);
            break;
        default:
            single_assigner_builtin<float, double, assign_error_fractional>::
                assign(reinterpret_cast<float *>(dst), &value);
            break;
    }
}

static void string_to_float64_single(char *dst, const char *const *src,
                        ckernel_prefix *extra)
{
    string_to_builtin_kernel_extra *e = reinterpret_cast<string_to_builtin_kernel_extra *>(extra);
    // Get the string from the source
    string s =
        e->src_string_tp->get_utf8_string(e->src_arrmeta, src[0], e->errmode);
    trim(s);
    double value = parse::checked_string_to_float64(
        s.data(), s.data() + s.size(), e->errmode);
    *reinterpret_cast<double *>(dst) = value;
}

static void string_to_float16_single(char *dst, const char *const *src,
                        ckernel_prefix *extra)
{
    string_to_builtin_kernel_extra *e =
        reinterpret_cast<string_to_builtin_kernel_extra *>(extra);
    double tmp;
    string_to_float64_single(reinterpret_cast<char *>(&tmp), src, extra);
    *reinterpret_cast<dynd_float16 *>(dst) = dynd_float16(tmp, e->errmode);
}

static void string_to_float128_single(char *DYND_UNUSED(dst),
                                      const char *const *DYND_UNUSED(src),
                                      ckernel_prefix *DYND_UNUSED(self))
{
    throw std::runtime_error("TODO: implement string_to_float128_single");
}

static void
string_to_complex_float32_single(char *DYND_UNUSED(dst),
                                 const char *const *DYND_UNUSED(src),
                                 ckernel_prefix *DYND_UNUSED(self))
{
    throw std::runtime_error("TODO: implement string_to_complex_float32_single");
}

static void
string_to_complex_float64_single(char *DYND_UNUSED(dst),
                                 const char *const *DYND_UNUSED(src),
                                 ckernel_prefix *DYND_UNUSED(self))
{
    throw std::runtime_error("TODO: implement string_to_complex_float64_single");
}

static expr_single_t static_string_to_builtin_kernels[builtin_type_id_count-2] = {
        &string_to_bool_single,
        &string_to_int<int8_t>::single,
        &string_to_int<int16_t>::single,
        &string_to_int<int32_t>::single,
        &string_to_int<int64_t>::single,
        &string_to_int128_single,
        &string_to_uint<uint8_t>::single,
        &string_to_uint<uint16_t>::single,
        &string_to_uint<uint32_t>::single,
        &string_to_uint<uint64_t>::single,
        &string_to_uint128_single,
        &string_to_float16_single,
        &string_to_float32_single,
        &string_to_float64_single,
        &string_to_float128_single,
        &string_to_complex_float32_single,
        &string_to_complex_float64_single
    };

size_t dynd::make_string_to_builtin_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, type_id_t dst_type_id,
    const ndt::type &src_string_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
    if (src_string_tp.get_kind() != string_kind) {
        stringstream ss;
        ss << "make_string_to_builtin_assignment_kernel: source type "
           << src_string_tp << " is not a string type";
        throw runtime_error(ss.str());
    }

    if (dst_type_id >= bool_type_id && dst_type_id <= complex_float64_type_id) {
        ckb_offset =
            make_kernreq_to_single_kernel_adapter(ckb, ckb_offset, 1, kernreq);
        string_to_builtin_kernel_extra *e =
            ckb->alloc_ck<string_to_builtin_kernel_extra>(ckb_offset);
        e->base.set_function<expr_single_t>(
            static_string_to_builtin_kernels[dst_type_id - bool_type_id]);
        e->base.destructor = &string_to_builtin_kernel_extra::destruct;
        // The kernel data owns this reference
        e->src_string_tp = static_cast<const base_string_type *>(
            ndt::type(src_string_tp).release());
        e->errmode = ectx->errmode;
        e->src_arrmeta = src_arrmeta;
        return ckb_offset;
    } else {
        stringstream ss;
        ss << "make_string_to_builtin_assignment_kernel: destination type id "
           << dst_type_id << " is not builtin";
        throw runtime_error(ss.str());
    }
}

/////////////////////////////////////////
// string to builtin assignment

namespace {
    struct builtin_to_string_kernel_extra {
        typedef builtin_to_string_kernel_extra extra_type;

        ckernel_prefix base;
        const base_string_type *dst_string_tp;
        type_id_t src_type_id;
        eval::eval_context ectx;
        const char *dst_arrmeta;

        static void single(char *dst, const char *const *src,
                            ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);

            // TODO: There are much faster ways to do this, but it's very generic!
            //       Also, for floating point values, a printing scheme like
            //       Python's, where it prints the shortest string that's
            //       guaranteed to parse to the same float number, would be
            //       better.
            stringstream ss;
            ndt::type(e->src_type_id).print_data(ss, NULL, src[0]);
            e->dst_string_tp->set_from_utf8_string(e->dst_arrmeta, dst, ss.str(), &e->ectx);
        }

        static void destruct(ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            if (e->dst_string_tp) {
                base_type_decref(e->dst_string_tp);
            }
        }
    };
} // anonymous namespace

size_t dynd::make_builtin_to_string_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_string_tp,
    const char *dst_arrmeta, type_id_t src_type_id, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
    if (dst_string_tp.get_kind() != string_kind) {
        stringstream ss;
        ss << "make_builtin_to_string_assignment_kernel: destination type " << dst_string_tp << " is not a string type";
        throw runtime_error(ss.str());
    }

    if (src_type_id >= 0 && src_type_id < builtin_type_id_count) {
        ckb_offset = make_kernreq_to_single_kernel_adapter(ckb, ckb_offset, 1, kernreq);
        builtin_to_string_kernel_extra *e =
            ckb->alloc_ck_leaf<builtin_to_string_kernel_extra>(ckb_offset);
        e->base.set_function<expr_single_t>(builtin_to_string_kernel_extra::single);
        e->base.destructor = builtin_to_string_kernel_extra::destruct;
        // The kernel data owns this reference
        e->dst_string_tp = static_cast<const base_string_type *>(ndt::type(dst_string_tp).release());
        e->src_type_id = src_type_id;
        e->ectx = *ectx;
        e->dst_arrmeta = dst_arrmeta;
        return ckb_offset;
    } else {
        stringstream ss;
        ss << "make_builtin_to_string_assignment_kernel: source type id " << src_type_id << " is not builtin";
        throw runtime_error(ss.str());
    }
}

void dynd::assign_utf8_string_to_builtin(type_id_t dst_type_id, char *dst,
                                         const char *str_begin,
                                         const char *str_end,
                                         const eval::eval_context *ectx)
{
    ndt::type dt = ndt::make_string();
    string_type_data d;
    string_type_arrmeta md;
    d.begin = const_cast<char *>(str_begin);
    d.end = const_cast<char *>(str_end);
    md.blockref = NULL;

    unary_ckernel_builder k;
    make_string_to_builtin_assignment_kernel(
        &k, 0, dst_type_id, dt, reinterpret_cast<const char *>(&md),
        kernel_request_single, ectx);
    k(dst, reinterpret_cast<const char *>(&d));
}
