//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/gfunc/serialize.hpp>
#include <dynd/dtypes/strided_dim_type.hpp>

#include <blosc.h>

using namespace std;
using namespace dynd;

nd::array dynd::gfunc::serialize(const nd::array& val)
{
    // To start, we're only supporting simple one-dimensional arrays of primitives
    const ndt::type& dt = val.get_type();
    if (dt.get_type_id() != strided_dim_type_id) {
        stringstream ss;
        ss << "dynd::gfunc::serialize is currently only a prototype, does not support dtype " << dt;
        throw runtime_error(ss.str());
    }
    const strided_dim_type *sad = static_cast<const strided_dim_type *>(dt.extended());
    const ndt::type& et = sad->get_element_type();
    if (!et.is_builtin()) {
        stringstream ss;
        ss << "dynd::gfunc::serialize is currently only a prototype, does not support dtype " << dt;
        throw runtime_error(ss.str());
    }

    throw runtime_error("dynd::gfunc::serialize is not implemented yet");
}

nd::array dynd::gfunc::deserialize(const nd::array& /*data*/)
{
    throw runtime_error("dynd::gfunc::deserialize is not implemented yet");
}
