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
    // The next_* functions advance an iterator pair and return
    // the code point that was processed.
    inline uint32_t next_ascii(const char *&it, const char *end)
    {
        uint32_t result = *reinterpret_cast<const uint8_t *>(it);
        if (result&0x80) {
            throw std::runtime_error("ascii input string had an invalid character with the highest bit set.");
        }
        ++it;
        return result;
    }

    inline void append_ascii(uint32_t cp, char *&it, char *end)
    {
        if ((cp&~0x7f) != 0) {
            throw std::runtime_error("cannot encode input code point as ascii.");
        }
        *it = static_cast<char>(cp);
        ++it;
    }

    inline uint32_t next_utf8(const char *&it, const char *end)
    {
        return utf8::next(reinterpret_cast<const uint8_t *&>(it), reinterpret_cast<const uint8_t *>(end));
    }

    inline void append_utf8(uint32_t cp, char *&it, char *end)
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
                throw std::runtime_error("Input too large to convert to destination fixed-size string");
            }
        }
    }

    inline uint32_t next_utf16(const char *&it_raw, const char *end_raw)
    {
        const uint16_t *&it = reinterpret_cast<const uint16_t *&>(it_raw);
        const uint16_t *end = reinterpret_cast<const uint16_t *>(end);
        uint32_t cp = utf8::internal::mask16(*it++);
        // Take care of surrogate pairs first
        if (utf8::internal::is_lead_surrogate(cp)) {
            if (it != end) {
                uint32_t trail_surrogate = utf8::internal::mask16(*it++);
                if (utf8::internal::is_trail_surrogate(trail_surrogate))
                    cp = (cp << 10) + trail_surrogate + utf8::internal::SURROGATE_OFFSET;
                else
                    throw utf8::invalid_utf16(static_cast<uint16_t>(trail_surrogate));
            }
            else
                throw utf8::invalid_utf16(static_cast<uint16_t>(cp));

        }
        // Lone trail surrogate
        else if (utf8::internal::is_trail_surrogate(cp))
            throw utf8::invalid_utf16(static_cast<uint16_t>(cp));
        ++it;
        return static_cast<uint32_t>(cp);
    }

    inline void append_utf16(uint32_t cp, char *&it_raw, char *end_raw)
    {
        uint16_t *&it = reinterpret_cast<uint16_t *&>(it_raw);
        uint16_t *end = reinterpret_cast<uint16_t *>(end);
        if (cp > 0xffff) { //make a surrogate pair
            *it = static_cast<uint16_t>((cp >> 10)   + utf8::internal::LEAD_OFFSET);
            if (++it >= end) {
                throw std::runtime_error("Input too large to convert to destination fixed-size string");
            }
            *it = static_cast<uint16_t>((cp & 0x3ff) + utf8::internal::TRAIL_SURROGATE_MIN);
            ++it;
        }
        else {
            *it = static_cast<uint16_t>(cp);
            ++it;
        }
    }

    inline uint32_t next_utf32(const char *&it_raw, const char *end_raw)
    {
        const uint32_t *&it = reinterpret_cast<const uint32_t *&>(it_raw);
        const uint32_t *end = reinterpret_cast<const uint32_t *>(end);
        uint32_t result = *it;
        if (!utf8::internal::is_code_point_valid(result)) {
            throw std::runtime_error("UTF32 input string had an invalid code point.");
        }
        ++it;
        return result;
    }

    inline void append_utf32(uint32_t cp, char *&it_raw, char *end_raw)
    {
        uint32_t *&it = reinterpret_cast<uint32_t *&>(it_raw);
        //uint32_t *end = reinterpret_cast<uint32_t *>(end);
        *it = cp;
        ++it;
    }

    struct fixedstring_encoder_kernel {

    };
} // anonymous namespace


void dnd::get_fixedstring_encoding_kernel(intptr_t dst_element_size, string_encoding_t dst_encoding,
                intptr_t src_element_size, string_encoding_t src_encoding,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel)
{
}
