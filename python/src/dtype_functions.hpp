//
// This header defines some wrapping functions to
// access various dtype parameters
//

#ifndef _DND__DTYPE_FUNCTIONS_HPP_
#define _DND__DTYPE_FUNCTIONS_HPP_

#include <stdint.h>
#include <sstream>

#include <dnd/dtype.hpp>

namespace pydnd {

inline std::string dtype_str(const dnd::dtype& d)
{
    std::stringstream ss;
    ss << d;
    return ss.str();
}

inline std::string dtype_repr(const dnd::dtype& d)
{
    std::stringstream ss;
    if (d.type_id() < dnd::builtin_type_id_count &&
                    d.type_id() != dnd::complex_float32_type_id &&
                    d.type_id() != dnd::complex_float64_type_id) {
        ss << "pydnd." << d;
    } else {
        ss << "pydnd.dtype('" << d << "')";
    }
    return ss.str();
}

} // namespace pydnd

#endif // _DND__DTYPE_FUNCTIONS_HPP_