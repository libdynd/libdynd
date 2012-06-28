//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dnd/ndarray.hpp>
#include <dnd/shape_tools.hpp>
#include <dnd/exceptions.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>
#include <dnd/diagnostics.hpp>

#include "ndarray_expr_node_instances.hpp"

using namespace std;
using namespace dnd;

// Node factory functions

ndarray_node_ref dnd::make_strided_ndarray_node(
            const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, char *originptr,
            const memory_block_ref& memblock)
{
    // TODO: Add a multidimensional DND_ASSERT_ALIGNED check here
    return ndarray_node_ref(new strided_ndarray_node(dt, ndim,
                                        shape, strides, originptr, memblock));
}

