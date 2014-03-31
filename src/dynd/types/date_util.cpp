//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <time.h>

#include <dynd/types/date_util.hpp>
#include <dynd/types/cstruct_type.hpp>

using namespace std;
using namespace dynd;

const int date_ymd::month_lengths[2][12] = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
};

const int date_ymd::month_starts[2][13] = {
    {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365},
    {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366}
};

date_ymd date_ymd::get_current_local_date()
{
    struct tm tm_;
#if defined(_MSC_VER)
    __time64_t rawtime;
    _time64(&rawtime);
    if (_localtime64_s(&tm_, &rawtime) != 0) {
        throw std::runtime_error("Failed to use '_localtime64_s' to convert "
                                "to a local time");
    }
#else
    time_t rawtime;
    time(&rawtime);
    if (localtime_r(&rawtime, &tm_) == NULL) {
        throw std::runtime_error("Failed to use 'localtime_r' to convert "
                                "to a local time");
    }
#endif
    date_ymd ymd;
    ymd.year = tm_.tm_year + 1900;
    ymd.month = tm_.tm_mon + 1;
    ymd.day = tm_.tm_mday;
    return ymd;
}
 
const ndt::type& date_ymd::type()
{
    static ndt::type tp = ndt::make_cstruct(
            ndt::make_type<int16_t>(), "year",
            ndt::make_type<int8_t>(), "month",
            ndt::make_type<int8_t>(), "day");
    return tp;
}

