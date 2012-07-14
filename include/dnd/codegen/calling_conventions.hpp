//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__CALLING_CONVENTIONS_HPP_
#define _DND__CALLING_CONVENTIONS_HPP_

#include <iostream>

namespace dnd {

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

} // namespace pydnd

#endif // _DND__CALLING_CONVENTIONS_HPP_
