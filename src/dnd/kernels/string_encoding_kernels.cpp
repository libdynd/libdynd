//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>

#include <dnd/diagnostics.hpp>
#include <dnd/kernels/string_encoding_kernels.hpp>

#include <utf8.h>

using namespace std;
using namespace dnd;

namespace {
    // Typedefs for reading and writing the next unicode characters for
    // a given encoding.
    typedef uint32_t (*next_codepoint_t)(const char *&it, const char *end);
    typedef void (*append_codepoint_t)(uint32_t cp, char *&it, char *end);
    // Typedef for copying a single string
    typedef void (*string_copy_t)(char *dst, const char *src, const AuxDataBase *auxdata);

    // The unicode byte order marker
    const uint32_t UNICODE_BOM_CODEPOINT = 0xFEFF;
    // The substitute code point when things are invalid
    const uint32_t ERROR_SUBSTITUTE_CODEPOINT = (uint32_t)'?';

    // The next_* functions advance an iterator pair and return
    // the code point that was processed.
    static uint32_t next_ascii(const char *&it, const char *DND_UNUSED(end))
    {
        uint32_t result = *reinterpret_cast<const uint8_t *>(it);
        if (result&0x80) {
            throw std::runtime_error("Ascii input string had an invalid character with the highest bit set.");
        }
        ++it;
        return result;
    }

    static uint32_t noerror_next_ascii(const char *&it, const char *DND_UNUSED(end))
    {
        uint32_t result = *reinterpret_cast<const uint8_t *>(it);
        ++it;
        return ((result&0x80) == 0) ? result : ERROR_SUBSTITUTE_CODEPOINT;
    }

    static void append_ascii(uint32_t cp, char *&it, char *DND_UNUSED(end))
    {
        if ((cp&~0x7f) != 0) {
            std::stringstream ss;
            ss << "Cannot encode input code point U+";
            hexadecimal_print(ss, cp);
            ss << " as ascii.";
            throw std::runtime_error(ss.str());
        }
        *it = static_cast<char>(cp);
        ++it;
    }

    static void noerror_append_ascii(uint32_t cp, char *&it, char *DND_UNUSED(end))
    {
        if ((cp&~0x7f) != 0) {
            cp = ERROR_SUBSTITUTE_CODEPOINT;
        }
        *it = static_cast<char>(cp);
        ++it;
    }

    static uint32_t next_utf8(const char *&it, const char *end)
    {
        return utf8::next(reinterpret_cast<const uint8_t *&>(it), reinterpret_cast<const uint8_t *>(end));
    }

    static uint32_t noerror_next_utf8(const char *&it, const char *end)
    {
        uint32_t cp = 0;
        // Determine the sequence length based on the lead octet
        std::size_t length = utf8::internal::sequence_length(it);

        // Get trail octets and calculate the code point
        utf8::internal::utf_error err = utf8::internal::UTF8_OK;
        switch (length) {
            case 0:
                return ERROR_SUBSTITUTE_CODEPOINT;
            case 1:
                err = utf8::internal::get_sequence_1(it, end, cp);
                break;
            case 2:
                err = utf8::internal::get_sequence_2(it, end, cp);
                break;
            case 3:
                err = utf8::internal::get_sequence_3(it, end, cp);
                break;
            case 4:
                err = utf8::internal::get_sequence_4(it, end, cp);
                break;
        }

        if (err == utf8::internal::UTF8_OK) {
            // Decoding succeeded. Now, security checks...
            if (utf8::internal::is_code_point_valid(cp)) {
                if (!utf8::internal::is_overlong_sequence(cp, length)){
                    // Passed! Return here.
                    ++it;
                    return cp;
                }
                else {
                    return ERROR_SUBSTITUTE_CODEPOINT;
                }
            }
            else {
                return ERROR_SUBSTITUTE_CODEPOINT;
            }
        } else {
            return ERROR_SUBSTITUTE_CODEPOINT;
        }

        return cp;
    }

    static void append_utf8(uint32_t cp, char *&it, char *end)
    {
        if (end - it >= 4) {
            it = utf8::append(cp, it);
        } else {
            char tmp[4];
            char *tmp_ptr = tmp;
            tmp_ptr = utf8::append(cp, tmp_ptr);
            if (tmp_ptr - tmp <= end - it) {
                memcpy(it, tmp, tmp_ptr - tmp);
                it += (tmp_ptr - tmp);
            } else {
                throw std::runtime_error("Input too large to convert to destination string");
            }
        }
    }

    static void noerror_append_utf8(uint32_t cp, char *&it, char *end)
    {
        if (end - it >= 4) {
            it = utf8::append(cp, it);
        } else {
            char tmp[4];
            char *tmp_ptr = tmp;
            tmp_ptr = utf8::append(cp, tmp_ptr);
            if (tmp_ptr - tmp <= end - it) {
                memcpy(it, tmp, tmp_ptr - tmp);
                it += (tmp_ptr - tmp);
            } else {
                // If it didn't fit, null-terminate
                memset(it, 0, end - it);
                it = end;
            }
        }
    }

    static uint32_t next_utf16(const char *&it_raw, const char *end_raw)
    {
        const uint16_t *&it = reinterpret_cast<const uint16_t *&>(it_raw);
        const uint16_t *end = reinterpret_cast<const uint16_t *>(end_raw);
        uint32_t cp = utf8::internal::mask16(*it++);
        // Take care of surrogate pairs first
        if (utf8::internal::is_lead_surrogate(cp)) {
            if (it != end) {
                uint32_t trail_surrogate = utf8::internal::mask16(*it++);
                if (utf8::internal::is_trail_surrogate(trail_surrogate)) {
                    cp = (cp << 10) + trail_surrogate + utf8::internal::SURROGATE_OFFSET;
                } else {
                    throw utf8::invalid_utf16(static_cast<uint16_t>(trail_surrogate));
                }
            }
            else {
                throw utf8::invalid_utf16(static_cast<uint16_t>(cp));
            }

        } else if (utf8::internal::is_trail_surrogate(cp)) {
            // Lone trail surrogate
            throw utf8::invalid_utf16(static_cast<uint16_t>(cp));
        }
        ++it;
        return cp;
    }

    static uint32_t noerror_next_utf16(const char *&it_raw, const char *end_raw)
    {
        const uint16_t *&it = reinterpret_cast<const uint16_t *&>(it_raw);
        const uint16_t *end = reinterpret_cast<const uint16_t *>(end_raw);
        uint32_t cp = utf8::internal::mask16(*it++);
        // Take care of surrogate pairs first
        if (utf8::internal::is_lead_surrogate(cp)) {
            if (it != end) {
                uint32_t trail_surrogate = utf8::internal::mask16(*it++);
                if (utf8::internal::is_trail_surrogate(trail_surrogate)) {
                    cp = (cp << 10) + trail_surrogate + utf8::internal::SURROGATE_OFFSET;
                } else {
                    return ERROR_SUBSTITUTE_CODEPOINT;
                }
            }
            else {
                return ERROR_SUBSTITUTE_CODEPOINT;
            }

        } else if (utf8::internal::is_trail_surrogate(cp)) {
            // Lone trail surrogate
            return ERROR_SUBSTITUTE_CODEPOINT;
        }
        ++it;
        return cp;
    }

    static void append_utf16(uint32_t cp, char *&it_raw, char *end_raw)
    {
        uint16_t *&it = reinterpret_cast<uint16_t *&>(it_raw);
        uint16_t *end = reinterpret_cast<uint16_t *>(end_raw);
        if (cp > 0xffff) { //make a surrogate pair
            *it = static_cast<uint16_t>((cp >> 10)   + utf8::internal::LEAD_OFFSET);
            if (++it >= end) {
                throw std::runtime_error("Input too large to convert to destination string");
            }
            *it = static_cast<uint16_t>((cp & 0x3ff) + utf8::internal::TRAIL_SURROGATE_MIN);
            ++it;
        }
        else {
            *it = static_cast<uint16_t>(cp);
            ++it;
        }
    }

    static void noerror_append_utf16(uint32_t cp, char *&it_raw, char *end_raw)
    {
        uint16_t *&it = reinterpret_cast<uint16_t *&>(it_raw);
        uint16_t *end = reinterpret_cast<uint16_t *>(end_raw);
        if (cp > 0xffff) { //make a surrogate pair
            if (it + 1 < end) {
                *it = static_cast<uint16_t>((cp >> 10)   + utf8::internal::LEAD_OFFSET);
                ++it;
                *it = static_cast<uint16_t>((cp & 0x3ff) + utf8::internal::TRAIL_SURROGATE_MIN);
                ++it;
            } else {
                // Null-terminate
                memset(it_raw, 0, end_raw - it_raw);
                it_raw = end_raw;
            }
        }
        else {
            *it = static_cast<uint16_t>(cp);
            ++it;
        }
    }

    static uint32_t next_utf32(const char *&it_raw, const char *end_raw)
    {
        const uint32_t *&it = reinterpret_cast<const uint32_t *&>(it_raw);
        const uint32_t *end = reinterpret_cast<const uint32_t *>(end_raw);
        uint32_t result = *it;
        if (!utf8::internal::is_code_point_valid(result)) {
            throw std::runtime_error("UTF32 input string had an invalid code point.");
        }
        ++it;
        return result;
    }

    static uint32_t noerror_next_utf32(const char *&it_raw, const char *end_raw)
    {
        const uint32_t *&it = reinterpret_cast<const uint32_t *&>(it_raw);
        const uint32_t *end = reinterpret_cast<const uint32_t *>(end_raw);
        uint32_t result = *it;
        ++it;
        if (!utf8::internal::is_code_point_valid(result)) {
            return ERROR_SUBSTITUTE_CODEPOINT;
        }
        return result;
    }

    static void append_utf32(uint32_t cp, char *&it_raw, char *DND_UNUSED(end_raw))
    {
        uint32_t *&it = reinterpret_cast<uint32_t *&>(it_raw);
        //uint32_t *end = reinterpret_cast<uint32_t *>(end);
        *it = cp;
        ++it;
    }

    static void noerror_append_utf32(uint32_t cp, char *&it_raw, char *DND_UNUSED(end_raw))
    {
        uint32_t *&it = reinterpret_cast<uint32_t *&>(it_raw);
        //uint32_t *end = reinterpret_cast<uint32_t *>(end);
        *it = cp;
        ++it;
    }

    struct fixedstring_encoder_kernel_auxdata {
        next_codepoint_t next_fn;
        append_codepoint_t append_fn;
        intptr_t dst_element_size, src_element_size;
        bool overflow_check;
    };

    /** Does a single fixed-string copy */
    static void fixedstring_copy(char *dst, const char *src,
            const fixedstring_encoder_kernel_auxdata& ad)
    {
        char *dst_end = dst + ad.dst_element_size;
        const char *src_end = src + ad.src_element_size;
        next_codepoint_t next_fn = ad.next_fn;
        append_codepoint_t append_fn = ad.append_fn;
        uint32_t cp;

        while (src < src_end && dst < dst_end) {
            cp = next_fn(src, src_end);
            // The fixedstring dtype uses null-terminated strings
            if (cp == 0) {
                // Null-terminate the destination string, and we're done
                memset(dst, 0, dst_end - dst);
                return;
            } else if (cp != UNICODE_BOM_CODEPOINT) {
                append_fn(cp, dst, dst_end);
            }
        }
        if (src < src_end && ad.overflow_check) {
            throw std::runtime_error("Input string is too large to convert to destination fixed-size string");
        }
   }

    struct fixedstring_copy_kernel {
        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const fixedstring_encoder_kernel_auxdata& ad = get_auxiliary_data<fixedstring_encoder_kernel_auxdata>(auxdata);
            for (intptr_t i = 0; i < count; ++i) {
                fixedstring_copy(dst, src, ad);

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t, const AuxDataBase *auxdata)
        {
            const fixedstring_encoder_kernel_auxdata& ad = get_auxiliary_data<fixedstring_encoder_kernel_auxdata>(auxdata);
            fixedstring_copy(dst, src, ad);
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const fixedstring_encoder_kernel_auxdata& ad = get_auxiliary_data<fixedstring_encoder_kernel_auxdata>(auxdata);
            intptr_t dst_element_size = ad.dst_element_size;

            // Convert the encoding once, then use memcpy calls for the rest.
            fixedstring_copy(dst, src, ad);
            const char *dst_first = dst;

            for (intptr_t i = 0; i < count; ++i) {
                memcpy(dst, dst_first, dst_element_size);

                dst += dst_stride;
            }
        }
    };
} // anonymous namespace

void dnd::get_fixedstring_encoding_kernel(intptr_t dst_element_size, string_encoding_t dst_encoding,
                intptr_t src_element_size, string_encoding_t src_encoding,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel)
{
    static specialized_unary_operation_table_t optable = {
        fixedstring_copy_kernel::general_kernel,
        fixedstring_copy_kernel::scalar_kernel,
        fixedstring_copy_kernel::general_kernel,
        fixedstring_copy_kernel::scalar_to_contiguous_kernel};
    out_kernel.specializations = optable;

    make_auxiliary_data<fixedstring_encoder_kernel_auxdata>(out_kernel.auxdata);
    fixedstring_encoder_kernel_auxdata& ad = out_kernel.auxdata.get<fixedstring_encoder_kernel_auxdata>();
    ad.dst_element_size = dst_element_size;
    ad.src_element_size = src_element_size;
    ad.overflow_check = (errmode != assign_error_none);
    switch (dst_encoding) {
        case string_encoding_ascii:
            ad.append_fn = (errmode != assign_error_none) ? append_ascii : noerror_append_ascii;
            break;
        case string_encoding_utf8:
            ad.append_fn = (errmode != assign_error_none) ? append_utf8 : noerror_append_utf8;
            break;
        case string_encoding_utf16:
            ad.append_fn = (errmode != assign_error_none) ? append_utf16 : noerror_append_utf16;
            break;
        case string_encoding_utf32:
            ad.append_fn = (errmode != assign_error_none) ? append_utf32 : noerror_append_utf32;
            break;
        default:
            throw runtime_error("Unrecognized destination string encoding");
    }
    switch (src_encoding) {
        case string_encoding_ascii:
            ad.next_fn = (errmode != assign_error_none) ? next_ascii : noerror_next_ascii;
            break;
        case string_encoding_utf8:
            ad.next_fn = (errmode != assign_error_none) ? next_utf8 : noerror_next_utf8;
            break;
        case string_encoding_utf16:
            ad.next_fn = (errmode != assign_error_none) ? next_utf16 : noerror_next_utf16;
            break;
        case string_encoding_utf32:
            ad.next_fn = (errmode != assign_error_none) ? next_utf32 : noerror_next_utf32;
            break;
        default:
            throw runtime_error("Unrecognized destination string encoding");
    }
}
