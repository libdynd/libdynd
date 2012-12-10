//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/date_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <datetime_strings.h>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// string to date assignment

namespace { struct string_to_date_assign_kernel {
    struct auxdata_storage {
        dtype src_string_dtype;
        assign_error_mode errmode;
        datetime::datetime_unit_t unit;
        datetime::datetime_conversion_rule_t casting;
    };

    static void single(char *dst, const char *src, unary_kernel_static_data *extra)
    {
        auxdata_storage& ad = get_auxiliary_data<auxdata_storage>(extra->auxdata);
        const extended_string_dtype *esd = static_cast<const extended_string_dtype *>(ad.src_string_dtype.extended());
        *reinterpret_cast<int32_t *>(dst) = datetime::parse_iso_8601_date(
                                esd->get_utf8_string(extra->src_metadata, src, ad.errmode),
                                ad.unit, ad.casting);
    }
};} // anonymous namespace

void dynd::get_string_to_date_assignment_kernel(date_unit_t dst_unit,
                const dtype& src_string_dtype,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel)
{
    if (src_string_dtype.get_kind() != string_kind) {
        stringstream ss;
        ss << "get_string_to_date_assignment_kernel: source dtype " << src_string_dtype << " is not a string dtype";
        throw runtime_error(ss.str());
    }

    out_kernel.kernel.single = &string_to_date_assign_kernel::single;
    out_kernel.kernel.contig = NULL;

    make_auxiliary_data<string_to_date_assign_kernel::auxdata_storage>(out_kernel.auxdata);
    string_to_date_assign_kernel::auxdata_storage& ad = out_kernel.auxdata.get<string_to_date_assign_kernel::auxdata_storage>();
    ad.errmode = errmode;
    ad.src_string_dtype = src_string_dtype;
    switch (dst_unit) {
        case date_unit_day:
            ad.unit = datetime::datetime_unit_day;
            break;
        case date_unit_month:
            ad.unit = datetime::datetime_unit_month;
            break;
        case date_unit_year:
            ad.unit = datetime::datetime_unit_year;
            break;
        default: {
            stringstream ss;
            ss << "unrecognized dynd date unit " << dst_unit;
            throw runtime_error(ss.str());
        }
    }
    switch (errmode) {
        case assign_error_fractional:
        case assign_error_inexact:
            ad.casting = datetime::datetime_conversion_strict;
            break;
        default:
            ad.casting = datetime::datetime_conversion_relaxed;
    }
}

/////////////////////////////////////////
// date to string assignment

namespace {
    struct date_to_string_assign_kernel {
        struct auxdata_storage {
            dtype dst_string_dtype;
            assign_error_mode errmode;
            datetime::datetime_unit_t unit;
        };

        /** Does a single fixed-string copy */
        static void single(char *dst, const char *src, unary_kernel_static_data *extra)
        {
            auxdata_storage& ad = get_auxiliary_data<auxdata_storage>(extra->auxdata);
            const extended_string_dtype *esd = static_cast<const extended_string_dtype *>(ad.dst_string_dtype.extended());
            int32_t date = *reinterpret_cast<const int32_t *>(src);
            esd->set_utf8_string(extra->dst_metadata, dst, ad.errmode, datetime::make_iso_8601_date(date, ad.unit));
        }
    };
} // anonymous namespace

void dynd::get_date_to_string_assignment_kernel(const dtype& dst_string_dtype,
                date_unit_t src_unit,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel)
{
    if (dst_string_dtype.get_kind() != string_kind) {
        stringstream ss;
        ss << "get_date_to_string_assignment_kernel: dest dtype " << dst_string_dtype << " is not a string dtype";
        throw runtime_error(ss.str());
    }

    out_kernel.kernel.single = &date_to_string_assign_kernel::single;
    out_kernel.kernel.contig = NULL;

    make_auxiliary_data<date_to_string_assign_kernel::auxdata_storage>(out_kernel.auxdata);
    date_to_string_assign_kernel::auxdata_storage& ad = out_kernel.auxdata.get<date_to_string_assign_kernel::auxdata_storage>();
    ad.errmode = errmode;
    ad.dst_string_dtype = dst_string_dtype;
    switch (src_unit) {
        case date_unit_day:
            ad.unit = datetime::datetime_unit_day;
            break;
        case date_unit_month:
            ad.unit = datetime::datetime_unit_month;
            break;
        case date_unit_year:
            ad.unit = datetime::datetime_unit_year;
            break;
        default: {
            stringstream ss;
            ss << "unrecognized dynd date unit " << src_unit;
            throw runtime_error(ss.str());
        }
    }
}

/////////////////////////////////////////
// date to string assignment

namespace {
    struct default_date_struct {
        int32_t year;
        int8_t month, day;
    };

    const dtype default_date_struct_dtypes[3] = {
        make_fixedstruct_dtype(make_dtype<int32_t>(), "year", make_dtype<int8_t>(), "month", make_dtype<int8_t>(), "day"),
        make_fixedstruct_dtype(make_dtype<int32_t>(), "year", make_dtype<int8_t>(), "month"),
        make_fixedstruct_dtype(make_dtype<int32_t>(), "year")
    };

    struct date_to_struct_trivial_assign_kernel {
        /** When the destination struct is exactly our desired layout */
        static void single(char *dst, const char *src, unary_kernel_static_data *extra)
        {
            datetime::datetime_unit_t unit = static_cast<datetime::datetime_unit_t>(get_raw_auxiliary_data(extra->auxdata)>>1);
            datetime::datetime_fields fld;
            fld.set_from_date_val(*reinterpret_cast<const int32_t *>(src), unit);
            default_date_struct *dst_struct = reinterpret_cast<default_date_struct *>(dst);
            switch (unit) {
                case datetime::datetime_unit_day:
                    dst_struct->day = fld.day;
                case datetime::datetime_unit_month:
                    dst_struct->month = fld.month;
                case datetime::datetime_unit_year:
                    dst_struct->year = fld.year != DATETIME_DATETIME_NAT ? fld.year : DATETIME_DATE_NAT;
                    break;
            }
        }
    };

    struct date_to_struct_assign_kernel {
        struct auxdata_storage {
            datetime::datetime_unit_t unit;
            kernel_instance<unary_operation_pair_t> kernel;
        };

        /** Does a single copy */
        static void single(char *dst, const char *src, unary_kernel_static_data *extra)
        {
            auxdata_storage& ad = get_auxiliary_data<auxdata_storage>(extra->auxdata);
            datetime::datetime_fields fld;
            fld.set_from_date_val(*reinterpret_cast<const int32_t *>(src), ad.unit);
            // Put the date in our default struct layout
            default_date_struct tmp_date;
            tmp_date.year = fld.year != DATETIME_DATETIME_NAT ? fld.year : DATETIME_DATE_NAT;
            tmp_date.month = fld.month;
            tmp_date.day = fld.day;
            // Copy to the destination
            unary_kernel_static_data kernel_extra(ad.kernel.auxdata, extra->dst_metadata, NULL);
            ad.kernel.kernel.single(dst, reinterpret_cast<const char *>(&tmp_date), &kernel_extra);
        }
    };

} // anonymous namespace

void dynd::get_date_to_struct_assignment_kernel(const dtype& dst_struct_dtype,
                date_unit_t src_unit,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel)
{
    if (dst_struct_dtype.get_kind() != struct_kind) {
        stringstream ss;
        ss << "get_date_to_struct_assignment_kernel: dest dtype " << dst_struct_dtype << " is not a struct dtype";
        throw runtime_error(ss.str());
    }

    datetime::datetime_unit_t src_datetime_unit;
    switch (src_unit) {
        case date_unit_day:
            src_datetime_unit = datetime::datetime_unit_day;
            break;
        case date_unit_month:
            src_datetime_unit = datetime::datetime_unit_month;
            break;
        case date_unit_year:
            src_datetime_unit = datetime::datetime_unit_year;
            break;
        default: {
            stringstream ss;
            ss << "unrecognized dynd date unit " << src_unit;
            throw runtime_error(ss.str());
        }
    }

    if (dst_struct_dtype == default_date_struct_dtypes[src_unit]) {
        out_kernel.kernel.single = &date_to_struct_trivial_assign_kernel::single;
        out_kernel.kernel.contig = NULL;
        make_raw_auxiliary_data(out_kernel.auxdata, static_cast<uintptr_t>(src_datetime_unit)<<1);
        return;
    }

    out_kernel.kernel.single = &date_to_struct_assign_kernel::single;
    out_kernel.kernel.contig = NULL;

    make_auxiliary_data<date_to_struct_assign_kernel::auxdata_storage>(out_kernel.auxdata);
    date_to_struct_assign_kernel::auxdata_storage& ad = out_kernel.auxdata.get<date_to_struct_assign_kernel::auxdata_storage>();
    ad.unit = src_datetime_unit;
    get_dtype_assignment_kernel(dst_struct_dtype, default_date_struct_dtypes[src_unit], errmode, NULL, ad.kernel);
}
