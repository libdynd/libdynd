//
// This header defines some wrapping functions to
// access various ndarray parameters
//

#ifndef _DND__NDARRAY_FUNCTIONS_HPP_
#define _DND__NDARRAY_FUNCTIONS_HPP_

#include <stdint.h>
#include <sstream>

#include <dnd/ndarray.hpp>

#include "Python.h"

namespace pydnd {

inline std::string ndarray_str(const dnd::ndarray& n)
{
    std::stringstream ss;
    ss << n;
    return ss.str();
}

inline std::string ndarray_repr(const dnd::ndarray& n)
{
    std::stringstream ss;
    ss << n;
    return ss.str();
}

inline std::string ndarray_debug_dump(const dnd::ndarray& n)
{
    std::stringstream ss;
    n.debug_dump(ss);
    return ss.str();
}

void ndarray_init(dnd::ndarray& n, PyObject* obj);
void ndarray_init(dnd::ndarray& n, PyObject* obj, const dnd::dtype& d);
dnd::ndarray ndarray_vals(const dnd::ndarray& n);

} // namespace pydnd

#endif // _DND__NDARRAY_FUNCTIONS_HPP_