//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <stdexcept>
#include <sstream>

#include <dnd/ndarray.hpp>
#include <dnd/shape_tools.hpp>
#include <dnd/ndarray_expr_node.hpp>

using namespace std;
using namespace dnd;

ndarray dnd::ndarray_expr_node::evaluate() const
{
    return ndarray();
}
