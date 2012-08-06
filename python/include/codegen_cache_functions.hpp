//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__CODEGEN_CACHE_FUNCTIONS_HPP_
#define _DND__CODEGEN_CACHE_FUNCTIONS_HPP_

#include "Python.h"

#include <stdint.h>
#include <sstream>
#include <stdexcept>

#include <dnd/codegen/codegen_cache.hpp>

namespace pydnd {

inline std::string codegen_cache_debug_dump(const dnd::codegen_cache& cgcache)
{
    std::stringstream ss;
    cgcache.debug_dump(ss);
    return ss.str();
}


} // namespace pydnd

#endif // _DND__CODEGEN_CACHE_FUNCTIONS_HPP_
