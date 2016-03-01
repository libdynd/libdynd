//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/base_bytes_type.hpp>

using namespace std;
using namespace dynd;

size_t ndt::base_bytes_type::get_iterdata_size(intptr_t DYND_UNUSED(ndim)) const { return 0; }
