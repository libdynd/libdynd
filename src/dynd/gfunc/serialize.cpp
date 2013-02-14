//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/gfunc/serialize.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>

#include <blosc.h>

using namespace std;
using namespace dynd;

ndobject dynd::gfunc::serialize(const ndobject& val)
{
    // To start, we're only supporting simple one-dimensional arrays of primitives
    const dtype& dt = val.get_dtype();
    if (dt.get_type_id() != strided_dim_type_id) {
        stringstream ss;
        ss << "dynd::gfunc::serialize is currently only a prototype, does not support dtype " << dt;
        throw runtime_error(ss.str());
    }
    const strided_dim_dtype *sad = static_cast<const strided_dim_dtype *>(dt.extended());
    const dtype& et = sad->get_element_dtype();
    if (!et.is_builtin()) {
        stringstream ss;
        ss << "dynd::gfunc::serialize is currently only a prototype, does not support dtype " << dt;
        throw runtime_error(ss.str());
    }

    throw runtime_error("dynd::gfunc::serialize is not implemented yet");
}

ndobject dynd::gfunc::deserialize(const ndobject& /*data*/)
{
    throw runtime_error("dynd::gfunc::deserialize is not implemented yet");
}
