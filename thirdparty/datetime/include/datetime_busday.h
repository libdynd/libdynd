#ifndef DATETIME_BUSDAY_H
#define DATETIME_BUSDAY_H

#include "datetime_types.h"

namespace datetime {

enum busday_roll_t {
    // Go forward in time to the following business day.
    busday_roll_forward,
    busday_roll_following = busday_roll_forward,
    // Go backward in time to the preceding business day.
    busday_roll_backward,
    busday_roll_preceding = busday_roll_backward,
    // Go forward in time to the following business day, unless it
    // crosses a month boundary, in which case go backward
    busday_roll_modifiedfollowing,
    // Go backward in time to the preceding business day, unless it
    // crosses a month boundary, in which case go forward.
    busday_roll_modifiedpreceding,
    // Produce a NaT for non-business days.
    busday_roll_nat,
    // Raise an exception for non-business days.
    busday_roll_throw,
    busday_roll_raise = busday_roll_throw
};

// Default 5-day week
extern const bool default_weekmask[7];
enum {busdays_in_default_weekmask = 5};

/** Gets the day of the week for a date with days unit */
int get_day_of_week(date_val_t date);

/** Returns true if the provided date is in the holidays list */
bool is_holiday(date_val_t date, date_val_t *holidays_begin, date_val_t *holidays_end);

/**
 * Finds the earliest holiday which is on or after 'date'. If 'date' does not
 * appear within the holiday range, returns 'holidays_begin' if 'date'
 * is before all holidays, or 'holidays_end' if 'date' is after all
 * holidays.
 *
 * To remove all the holidays before 'date' from a holiday range, do:
 *
 *      holidays_begin = find_holiday_earliest_on_or_after(date,
 *                                          holidays_begin, holidays_end);
 *
 * The holidays list should be normalized, which means any NaT (not-a-time)
 * values, duplicates, and dates already excluded by the weekmask should
 * be removed, and the list should be sorted.
 */
date_val_t *find_earliest_holiday_on_or_after(date_val_t date,
            date_val_t *holidays_begin, date_val_t *holidays_end);
            
/**
 * Finds the earliest holiday which is after 'date'. If 'date' does not
 * appear within the holiday range, returns 'holidays_begin' if 'date'
 * is before all holidays, or 'holidays_end' if 'date' is after all
 * holidays.
 *
 * To remove all the holidays after 'date' from a holiday range, do:
 *
 *      holidays_end = find_holiday_earliest_after(date,
 *                                          holidays_begin, holidays_end);
 *
 * The holidays list should be normalized, which means any NaT (not-a-time)
 * values, duplicates, and dates already excluded by the weekmask should
 * be removed, and the list should be sorted.
 */
date_val_t *find_earliest_holiday_after(date_val_t date,
            date_val_t *holidays_begin, date_val_t *holidays_end);

/**
 * Applies the 'roll' strategy to 'date', placing the result in 'out'
 * and setting 'out_day_of_week' to the day of the week that results.
 *
 * @param dates  An array of dates with 'datetime64[D]' data type.
 * @param out_day_of_week  The day of the week of the returned result.
 * @param weekmask A 7-element boolean mask, 1 for possible business days and 0
 *           for non-business days.
 * @param holidays_begin/holidays_end   A sorted list of dates of 'day' unit,
 *           with any dates falling on a day of the
 *           week without weekmask[i] == 1 already filtered out.
 *
 * Throws an exception on error.
 */
date_val_t apply_business_day_roll(date_val_t date,
                    int *out_day_of_week,
                    busday_roll_t roll,
                    const bool *weekmask = default_weekmask,
                    date_val_t *holidays_begin = 0, date_val_t *holidays_end = 0);

/**
 * Applies a single business day offset.
 *
 * @param dates  An array of dates with 'datetime64[D]' data type.
 * @param weekmask A 7-element boolean mask, 1 for possible business days and 0
 *           for non-business days.
 * @param busdays_in_weekmask  A count of how many 1's there are in weekmask.
 * @param holidays_begin/holidays_end   A sorted list of dates of 'day' unit,
 *           with any dates falling on a day of the
 *           week without weekmask[i] == 1 already filtered out.
 *
 * Throws an exception on error
 */
date_val_t apply_business_day_offset(date_val_t date, int32_t offset,
                    busday_roll_t roll,
                    const bool *weekmask = default_weekmask,
                    int busdays_in_weekmask = busdays_in_default_weekmask,
                    date_val_t *holidays_begin = 0, date_val_t *holidays_end = 0);

/**
 * Applies a single business day count operation, returning the result
 *
 * Throws an exception on error
 */
int32_t apply_business_day_count(date_val_t date_begin, date_val_t date_end,
                    const bool *weekmask = default_weekmask,
                    int busdays_in_weekmask = busdays_in_default_weekmask,
                    date_val_t *holidays_begin = 0, date_val_t *holidays_end = 0);

/**
 * Returns true if the specified date is a business day according to
 * the weekmask and holidays list.
 *
 * @param dates  An array of dates with 'datetime64[D]' data type.
 * @param out      Either NULL, or an array with 'bool' data type
 *           in which to place the resulting dates.
 * @param weekmask A 7-element boolean mask, 1 for possible business days and 0
 *           for non-business days.
 * @param holidays_begin/holidays_end   A sorted list of dates of 'day' unit,
 *           with any dates falling on a day of the
 *           week without weekmask[i] == 1 already filtered out.
 */
inline bool is_business_day(date_val_t date,
                    const bool *weekmask = default_weekmask,
                    date_val_t *holidays_begin = 0, date_val_t *holidays_end = 0)
{
    return weekmask[get_day_of_week(date)] &&
            !is_holiday(date, holidays_begin, holidays_end) &&
            date != DATETIME_DATE_NAT;
}

/**
 * Sorts the the array of dates provided in place and removes
 * NaT, duplicates and any date which is already excluded on account
 * of the weekmask.
 *
 * @param holidays_begin/holidays_end   This list is sorted and
 *                filtered to remove NaT values and days already
 *                excluded by the weekmask. The holidays_end value is
 *                modified in place.
 * @param weekmask A 7-element boolean mask, 1 for possible business days and 0
 *           for non-business days.
 */
void normalize_holidays_list(
            date_val_t *holidays_begin, date_val_t *&holidays_end,
            const bool *weekmask);

} // namespace datetime

#endif // DATETIME_BUSDAY_H
