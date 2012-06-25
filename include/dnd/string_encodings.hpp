//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__STRING_ENCODINGS_HPP_
#define _DND__STRING_ENCODINGS_HPP_

#include <iostream>

namespace dnd {

enum string_encoding_t {
    string_encoding_ascii,
    string_encoding_utf8,
    string_encoding_utf16,
    string_encoding_utf32,

    string_encoding_invalid
};

inline std::ostream& operator<<(std::ostream& o, string_encoding_t encoding)
{
    switch (encoding) {
        case string_encoding_ascii:
            o << "ascii";
            break;
        case string_encoding_utf8:
            o << "utf8";
            break;
        case string_encoding_utf16:
            o << "utf16";
            break;
        case string_encoding_utf32:
            o << "utf32";
            break;
        default:
            o << "unknown codec";
            break;
    }

    return o;
}


} // namespace dnd

#endif // _DND__STRING_ENCODINGS_HPP_
