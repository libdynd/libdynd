#ifndef DATETIME_TYPES_H
#define DATETIME_TYPES_H

#if defined(_MSC_VER) && _MSC_VER < 1600
#include <limits.h>
typedef int int32_t;
typedef __int64 int64_t;
#define INT32_MIN INT_MIN
#define INT64_MIN _I64_MIN
#else
// Request the limits macros
#define __STDC_LIMIT_MACROS
#include <limits.h>
#include <stdint.h>
#endif

namespace datetime {

// Value storage types for date, time, datetime
typedef int32_t date_val_t;
typedef int32_t time_val_t;
typedef int64_t datetime_val_t;

} // namespace datetime

// Not-a-time constants
#define DATETIME_DATE_NAT INT32_MIN
#define DATETIME_TIME_NAT INT32_MIN
#define DATETIME_DATETIME_NAT INT64_MIN

#endif // DATETIME_TYPES_H
