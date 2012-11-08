#ifndef DATETIME_LOCALTIME_H
#define DATETIME_LOCALTIME_H

#include "datetime_main.h"

namespace datetime {

/**
 * Converts a datetimestruct in UTC to a datetimestruct in local time,
 * also returning the timezone offset applied.
 *
 * Throws an exception on failure.
 */
void convert_utc_to_local(datetime_fields *out_dtf_local,
                const datetime_fields *dtf_utc, int *out_timezone_offset);

/**
 * Converts a datetimestruct in local time to a datetimestruct in UTC.
 *
 * Throws an exception on failure.
 */
void convert_local_to_utc(datetime_fields *out_dtf_utc,
                const datetime_fields *dtf_local);

/**
 * Fills in the year, month, and day fields with the
 * current date in local time.
 */
void fill_current_local_date(datetime_fields *out);

/**
 * Returns the current UTC datetime with a seconds unit.
 */
datetime_val_t get_current_utc_datetime_seconds();

} // namespace datetime

#endif // DATETIME_LOCALTIME_H
