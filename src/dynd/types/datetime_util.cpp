//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <time.h>

#include <dynd/types/datetime_util.hpp>
#include <dynd/types/cstruct_type.hpp>

using namespace std;
using namespace dynd;

const ndt::type& datetime_struct::type()
{
    static ndt::type tp = ndt::make_cstruct(
            ndt::make_type<int16_t>(), "year",
            ndt::make_type<int8_t>(), "month",
            ndt::make_type<int8_t>(), "day",
            ndt::make_type<int8_t>(), "hour",
            ndt::make_type<int8_t>(), "minute",
            ndt::make_type<int8_t>(), "second",
            ndt::make_type<int32_t>(), "tick");
    return tp;
}
 
