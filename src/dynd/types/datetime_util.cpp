//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <time.h>

#include <dynd/types/datetime_util.hpp>
#include <dynd/types/datetime_parser.hpp>
#include <dynd/types/cstruct_type.hpp>

using namespace std;
using namespace dynd;

std::string datetime_struct::to_str() const
{
    if (is_valid()) {
        return ymd.to_str() + "T" + hmst.to_str();
    } else {
        return string();
    }
}

void datetime_struct::set_from_str(const std::string &s,
                                   date_parse_order_t ambig, int century_window)
{
    if (!string_to_datetime(s.data(), s.data() + s.size(), *this, ambig,
                            century_window)) {
        stringstream ss;
        ss << "Unable to parse ";
        print_escaped_utf8_string(ss, s);
        ss << " as a datetime";
        throw invalid_argument(ss.str());
    }
}

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
 
