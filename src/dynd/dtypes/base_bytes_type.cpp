//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/base_bytes_type.hpp>

using namespace std;
using namespace dynd;


base_bytes_type::~base_bytes_type()
{
}

size_t base_bytes_type::get_iterdata_size(size_t DYND_UNUSED(ndim)) const
{
    return 0;
}
