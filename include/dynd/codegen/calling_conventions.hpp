//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>

namespace dynd {

enum calling_convention_t {
    cdecl_callconv,
    win32_stdcall_callconv
};

inline std::ostream& operator <<(std::ostream& o, calling_convention_t cc) {
    switch (cc) {
        case cdecl_callconv:
            o << "cdecl";
            break;
        case win32_stdcall_callconv:
            o << "win32_stdcall";
            break;
        default:
            o << "unknown calling convention (" << (int)cc << ")";
            break;
    }

    return o;
}

} // namespace pydynd
