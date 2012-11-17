/*
 * This file implements business day functionality for NumPy datetime.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#include <stdexcept>
#include <algorithm>

#include "datetime_main.h"
#include "datetime_busday.h"

using namespace datetime;

const bool datetime::default_weekmask[7] = {true, true, true, true, true, false, false};


/** Gets the day of the week for a date with days unit */
int datetime::get_day_of_week(date_val_t date)
{
    int day_of_week;

    /* Get the day of the week for 'date' (1970-01-05 is Monday) */
    day_of_week = (int)((date - 4) % 7);
    if (day_of_week < 0) {
        day_of_week += 7;
    }

    return day_of_week;
}

/*
 * Returns 1 if the date is a holiday (contained in the sorted
 * list of dates), 0 otherwise.
 *
 * The holidays list should be normalized, which means any NaT (not-a-time)
 * values, duplicates, and dates already excluded by the weekmask should
 * be removed, and the list should be sorted.
 */
bool datetime::is_holiday(date_val_t date, date_val_t *holidays_begin, date_val_t *holidays_end)
{
    date_val_t *trial;

    /* Simple binary search */
    while (holidays_begin < holidays_end) {
        trial = holidays_begin + (holidays_end - holidays_begin) / 2;

        if (date < *trial) {
            holidays_end = trial;
        }
        else if (date > *trial) {
            holidays_begin = trial + 1;
        }
        else {
            return 1;
        }
    }

    /* Not found */
    return false;
}

date_val_t *datetime::find_earliest_holiday_on_or_after(date_val_t date,
            date_val_t *holidays_begin, date_val_t *holidays_end)
{
    date_val_t *trial;

    /* Simple binary search */
    while (holidays_begin < holidays_end) {
        trial = holidays_begin + (holidays_end - holidays_begin) / 2;

        if (date < *trial) {
            holidays_end = trial;
        }
        else if (date > *trial) {
            holidays_begin = trial + 1;
        }
        else {
            return trial;
        }
    }

    return holidays_begin;
}

date_val_t *datetime::find_earliest_holiday_after(date_val_t date,
            date_val_t *holidays_begin, date_val_t *holidays_end)
{
    date_val_t *trial;

    /* Simple binary search */
    while (holidays_begin < holidays_end) {
        trial = holidays_begin + (holidays_end - holidays_begin) / 2;

        if (date < *trial) {
            holidays_end = trial;
        }
        else if (date > *trial) {
            holidays_begin = trial + 1;
        }
        else {
            return trial + 1;
        }
    }

    return holidays_begin;
}

date_val_t datetime::apply_business_day_roll(date_val_t date,
                    int *out_day_of_week,
                    busday_roll_t roll,
                    const bool *weekmask,
                    date_val_t *holidays_begin, date_val_t *holidays_end)
{
    int day_of_week;

    /* Deal with NaT input */
    if (date == DATETIME_DATE_NAT) {
        if (roll == busday_roll_throw) {
            throw std::runtime_error("NaT input in apply_business_day_roll with 'raise' policy");
        }
        else {
            return DATETIME_DATE_NAT;
        }
    }

    /* Get the day of the week for 'date' */
    day_of_week = get_day_of_week(date);

    /* Apply the 'roll' if it's not a business day */
    if (weekmask[day_of_week] == 0 ||
                        is_holiday(date, holidays_begin, holidays_end)) {
        date_val_t start_date = date;
        int start_day_of_week = day_of_week;

        switch (roll) {
            case busday_roll_following:
            case busday_roll_modifiedfollowing: {
                do {
                    ++date;
                    if (++day_of_week == 7) {
                        day_of_week = 0;
                    }
                } while (weekmask[day_of_week] == 0 ||
                            is_holiday(date, holidays_begin, holidays_end));

                if (roll == busday_roll_modifiedfollowing) {
                    /* If we crossed a month boundary, do preceding instead */
                    if (days_to_month_number(start_date) !=
                                days_to_month_number(date)) {
                        date = start_date;
                        day_of_week = start_day_of_week;

                        do {
                            --date;
                            if (--day_of_week == -1) {
                                day_of_week = 6;
                            }
                        } while (weekmask[day_of_week] == 0 ||
                            is_holiday(date, holidays_begin, holidays_end));
                    }
                }
                break;
            }
            case busday_roll_preceding:
            case busday_roll_modifiedpreceding: {
                do {
                    --date;
                    if (--day_of_week == -1) {
                        day_of_week = 6;
                    }
                } while (weekmask[day_of_week] == 0 ||
                            is_holiday(date, holidays_begin, holidays_end));

                if (roll == busday_roll_modifiedpreceding) {
                    /* If we crossed a month boundary, do following instead */
                    if (days_to_month_number(start_date) !=
                                days_to_month_number(date)) {
                        date = start_date;
                        day_of_week = start_day_of_week;

                        do {
                            ++date;
                            if (++day_of_week == 7) {
                                day_of_week = 0;
                            }
                        } while (weekmask[day_of_week] == 0 ||
                            is_holiday(date, holidays_begin, holidays_end));
                    }
                }
                break;
            }
            case busday_roll_nat: {
                date = DATETIME_DATE_NAT;
                break;
            }
            case busday_roll_throw: {
                throw std::runtime_error("Holiday input in apply_business_day_roll with 'raise' policy");
            }
        }
    }

    *out_day_of_week = day_of_week;
    return date;
}

date_val_t datetime::apply_business_day_offset(date_val_t date, int32_t offset,
                    busday_roll_t roll,
                    const bool *weekmask, int busdays_in_weekmask,
                    date_val_t *holidays_begin, date_val_t *holidays_end)
{
    int day_of_week = 0;
    date_val_t *holidays_temp;

    /* Roll the date to a business day */
    date = apply_business_day_roll(date, &day_of_week,
                                roll,
                                weekmask,
                                holidays_begin, holidays_end);

    /* If we get a NaT, just return it */
    if (date == DATETIME_DATE_NAT) {
        return date;
    }

    /* Now we're on a valid business day */
    if (offset > 0) {
        /* Remove any earlier holidays */
        holidays_begin = find_earliest_holiday_on_or_after(date,
                                            holidays_begin, holidays_end);

        /* Jump by as many weeks as we can */
        date += (offset / busdays_in_weekmask) * 7;
        offset = offset % busdays_in_weekmask;

        /* Adjust based on the number of holidays we crossed */
        holidays_temp = find_earliest_holiday_after(date,
                                            holidays_begin, holidays_end);
        offset += (int32_t)(holidays_temp - holidays_begin);
        holidays_begin = holidays_temp;

        /* Step until we use up the rest of the offset */
        while (offset > 0) {
            ++date;
            if (++day_of_week == 7) {
                day_of_week = 0;
            }
            if (weekmask[day_of_week] && !is_holiday(date,
                                            holidays_begin, holidays_end)) {
                offset--;
            }
        }
    }
    else if (offset < 0) {
        /* Remove any later holidays */
        holidays_end = find_earliest_holiday_after(date,
                                            holidays_begin, holidays_end);

        /* Jump by as many weeks as we can */
        date += (offset / busdays_in_weekmask) * 7;
        offset = offset % busdays_in_weekmask;

        /* Adjust based on the number of holidays we crossed */
        holidays_temp = find_earliest_holiday_on_or_after(date,
                                            holidays_begin, holidays_end);
        offset -= (int32_t)(holidays_end - holidays_temp);
        holidays_end = holidays_temp;

        /* Step until we use up the rest of the offset */
        while (offset < 0) {
            --date;
            if (--day_of_week == -1) {
                day_of_week = 6;
            }
            if (weekmask[day_of_week] && !is_holiday(date,
                                            holidays_begin, holidays_end)) {
                offset++;
            }
        }
    }

    return date;
}

int32_t datetime::apply_business_day_count(date_val_t date_begin, date_val_t date_end,
                    const bool *weekmask, int busdays_in_weekmask,
                    date_val_t *holidays_begin, date_val_t *holidays_end)
{
    int32_t count, whole_weeks;
    int day_of_week = 0;
    bool swapped = false;

    /* If we get a NaT, raise an error */
    if (date_begin == DATETIME_DATE_NAT || date_end == DATETIME_DATE_NAT) {
        throw std::runtime_error(
                "Cannot compute a business day count with a NaT (not-a-time) date");
    }

    /* Trivial empty date range */
    if (date_begin == date_end) {
        return 0;
    }
    else if (date_begin > date_end) {
        date_val_t tmp = date_begin;
        date_begin = date_end;
        date_end = tmp;
        swapped = true;
    }

    /* Remove any earlier holidays */
    holidays_begin = find_earliest_holiday_on_or_after(date_begin,
                                        holidays_begin, holidays_end);
    /* Remove any later holidays */
    holidays_end = find_earliest_holiday_on_or_after(date_end,
                                        holidays_begin, holidays_end);

    /* Start the count as negative the number of holidays in the range */
    count = -(int32_t)(holidays_end - holidays_begin);

    /* Add the whole weeks between date_begin and date_end */
    whole_weeks = (date_end - date_begin) / 7;
    count += whole_weeks * busdays_in_weekmask;
    date_begin += whole_weeks * 7;

    if (date_begin < date_end) {
        /* Get the day of the week for 'date_begin' */
        day_of_week = get_day_of_week(date_begin);

        /* Count the remaining days one by one */
        while (date_begin < date_end) {
            if (weekmask[day_of_week]) {
                count++;
            }
            ++date_begin;
            if (++day_of_week == 7) {
                day_of_week = 0;
            }
        }
    }

    return swapped ? -count : count;
}

/*
 * Sorts the the array of dates provided in place and removes
 * NaT, duplicates and any date which is already excluded on account
 * of the weekmask.
 *
 * Returns the number of dates left after removing weekmask-excluded
 * dates.
 */
void datetime::normalize_holidays_list(
            date_val_t *holidays_begin, date_val_t *&holidays_end,
            const bool *weekmask)
{
    /* Sort the dates */
    std::sort(holidays_begin, holidays_end);

    date_val_t *dates = holidays_begin;
    size_t count = holidays_end - dates;

    date_val_t lastdate = DATETIME_DATE_NAT;
    size_t trimcount;

    /* Sweep throught the array, eliminating unnecessary values */
    trimcount = 0;
    for (size_t i = 0; i != count; ++i) {
        date_val_t date = dates[i];

        /* Skip any NaT or duplicate */
        if (date != DATETIME_DATE_NAT && date != lastdate) {
            int day_of_week = get_day_of_week(date);

            /*
             * If the holiday falls on a possible business day,
             * then keep it.
             */
            if (weekmask[day_of_week]) {
                dates[trimcount++] = date;
                lastdate = date;
            }
        }
    }

    /* Adjust the end of the holidays array */
    holidays_end = dates + trimcount;
}

