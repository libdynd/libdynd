//
// This header defines some placement wrappers of dtype and ndarray
// to enable wrapping them without adding extra indirection layers.
//

#ifndef _DND__PLACEMENT_WRAPPERS_HPP_
#define _DND__PLACEMENT_WRAPPERS_HPP_

#include <stdint.h>

#include <dnd/dtype.hpp>

namespace pydnd {

// This is a struct with the same alignment (because of intptr_t)
// and size as dnd::dtype. It's what we wrap in Cython, and use
// placement new and delete to manage its lifetime.
struct dtype_placement_wrapper {
    intptr_t dummy[(sizeof(dnd::dtype) + sizeof(intptr_t) - 1)/sizeof(intptr_t)];
};

inline void dtype_placement_new(dtype_placement_wrapper& v)
{
    // Call placement new
    new (&v) dnd::dtype();
}

inline void dtype_placement_delete(dtype_placement_wrapper& v)
{
    // Call the destructor
    ((dnd::dtype *)(&v))->~dtype();
}

// dtype placement cast
inline dnd::dtype& dpc(dtype_placement_wrapper& v)
{
    return *(dnd::dtype *)&v;
}

inline void dtype_print(dnd::dtype& d)
{
    std::cout << d << "\n";
}

///////////////////////////////////
// Same thing as above, for ndarray

struct ndarray_placement_wrapper {
    intptr_t dummy[(sizeof(dnd::ndarray) + sizeof(intptr_t) - 1)/sizeof(intptr_t)];
};

inline void ndarray_placement_new(ndarray_placement_wrapper& v)
{
    // Call placement new
    new (&v) dnd::ndarray();
}

inline void ndarray_placement_delete(ndarray_placement_wrapper& v)
{
    // Call the destructor
    ((dnd::ndarray *)(&v))->~ndarray();
}

// ndarray placement cast
inline dnd::ndarray& npc(ndarray_placement_wrapper& v)
{
    return *(dnd::ndarray *)&v;
}


} // namespace pydnd

#endif // _DND__PLACEMENT_WRAPPERS_HPP_