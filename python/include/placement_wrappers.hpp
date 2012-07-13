//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some placement wrappers of dtype and ndarray
// to enable wrapping them without adding extra indirection layers.
//

#ifndef _DND__PLACEMENT_WRAPPERS_HPP_
#define _DND__PLACEMENT_WRAPPERS_HPP_

#include <stdint.h>

#include <dnd/dtype.hpp>
#include <dnd/ndarray.hpp>
#include <dnd/codegen/codegen_cache.hpp>

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
inline dnd::dtype& GET(dtype_placement_wrapper& v)
{
    return *(dnd::dtype *)&v;
}

// dtype placement assignment
inline void SET(dtype_placement_wrapper& v, const dnd::dtype& d)
{
    *(dnd::dtype *)&v = d;
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

// placement cast
inline dnd::ndarray& GET(ndarray_placement_wrapper& v)
{
    return *(dnd::ndarray *)&v;
}

// placement assignment
inline void SET(ndarray_placement_wrapper& v, const dnd::ndarray& d)
{
    *(dnd::ndarray *)&v = d;
}

///////////////////////////////////
// Same thing as above, for codegen_cache

struct codegen_cache_placement_wrapper {
    intptr_t dummy[(sizeof(dnd::codegen_cache) + sizeof(intptr_t) - 1)/sizeof(intptr_t)];
};

inline void codegen_cache_placement_new(codegen_cache_placement_wrapper& v)
{
    // Call placement new
    new (&v) dnd::codegen_cache();
}

inline void codegen_cache_placement_delete(codegen_cache_placement_wrapper& v)
{
    // Call the destructor
    ((dnd::codegen_cache *)(&v))->~codegen_cache();
}

// placement cast
inline dnd::codegen_cache& GET(codegen_cache_placement_wrapper& v)
{
    return *(dnd::codegen_cache *)&v;
}

} // namespace pydnd

#endif // _DND__PLACEMENT_WRAPPERS_HPP_
