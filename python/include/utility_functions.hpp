//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__UTILITY_FUNCTIONS_HPP_
#define _DND__UTILITY_FUNCTIONS_HPP_

#include <stdint.h>
#include <sstream>

#include "Python.h"

namespace pydnd {

PyObject* intptr_array_as_tuple(int size, const intptr_t *array);

} // namespace pydnd

#endif // _DND__UTILITY_FUNCTIONS_HPP_