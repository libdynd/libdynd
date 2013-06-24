#ifndef DATETIME_LOCALTIME_H
#define DATETIME_LOCALTIME_H

#include "datetime_main.h"

namespace datetime {

/**
 * Fills in the year, month, and day fields with the
 * current date in local time.
 */
void fill_current_local_date(datetime_fields *out);

/**
 * Fills in the year, month, and day fields with the
 * current date in local time.
 */
void fill_current_local_date(date_ymd *out);

/**
 * Returns the current UTC datetime with a seconds unit.
 */
datetime_val_t get_current_utc_datetime_seconds();

} // namespace datetime

#endif // DATETIME_LOCALTIME_H
