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
        datetime::datetime_conversion_rule_t casting;
    };

    static void single(char *dst, const char *src, unary_kernel_static_data *extra)
    {
        auxdata_storage& ad = get_auxiliary_data<auxdata_storage>(extra->auxdata);
        const base_string_dtype *esd = static_cast<const base_string_dtype *>(ad.src_string_dtype.extended());
        *reinterpret_cast<int32_t *>(dst) = datetime::parse_iso_8601_date(
                                esd->get_utf8_string(extra->src_metadata, src, ad.errmode),
                                datetime::datetime_unit_day, ad.casting);
    }
};} // anonymous namespace

void dynd::get_string_to_date_assignment_kernel(const dtype& src_string_dtype,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel)
{
    if (src_string_dtype.get_kind() != string_kind) {
        stringstream ss;
        ss << "get_string_to_date_assignment_kernel: source dtype " << src_string_dtype << " is not a string dtype";
        throw runtime_error(ss.str());
    }

    out_kernel.kernel.single = &string_to_date_assign_kernel::single;
    out_kernel.kernel.strided = NULL;

    make_auxiliary_data<string_to_date_assign_kernel::auxdata_storage>(out_kernel.extra.auxdata);
    string_to_date_assign_kernel::auxdata_storage& ad = out_kernel.extra.auxdata.get<string_to_date_assign_kernel::auxdata_storage>();
    ad.errmode = errmode;
    ad.src_string_dtype = src_string_dtype;
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
        };

        /** Does a single fixed-string copy */
        static void single(char *dst, const char *src, unary_kernel_static_data *extra)
        {
            auxdata_storage& ad = get_auxiliary_data<auxdata_storage>(extra->auxdata);
            const base_string_dtype *esd = static_cast<const base_string_dtype *>(ad.dst_string_dtype.extended());
            int32_t date = *reinterpret_cast<const int32_t *>(src);
            esd->set_utf8_string(extra->dst_metadata, dst, ad.errmode,
                            datetime::make_iso_8601_date(date, datetime::datetime_unit_day));
        }
    };
} // anonymous namespace

void dynd::get_date_to_string_assignment_kernel(const dtype& dst_string_dtype,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel)
{
    if (dst_string_dtype.get_kind() != string_kind) {
        stringstream ss;
        ss << "get_date_to_string_assignment_kernel: dest dtype " << dst_string_dtype << " is not a string dtype";
        throw runtime_error(ss.str());
    }

    out_kernel.kernel.single = &date_to_string_assign_kernel::single;
    out_kernel.kernel.strided = NULL;

    make_auxiliary_data<date_to_string_assign_kernel::auxdata_storage>(out_kernel.extra.auxdata);
    date_to_string_assign_kernel::auxdata_storage& ad = out_kernel.extra.auxdata.get<date_to_string_assign_kernel::auxdata_storage>();
    ad.errmode = errmode;
    ad.dst_string_dtype = dst_string_dtype;
}

/////////////////////////////////////////
// data for date to/from struct assignment

const dtype dynd::date_dtype_default_struct_dtype =
        make_fixedstruct_dtype(make_dtype<int32_t>(), "year", make_dtype<int8_t>(), "month", make_dtype<int8_t>(), "day");
/////////////////////////////////////////
// date to struct assignment

namespace {
    struct date_to_struct_trivial_assign_kernel {
        /** When the destination struct is exactly our desired layout */
        static void single(char *dst, const char *src, unary_kernel_static_data *DYND_UNUSED(extra))
        {
            datetime::date_ymd fld;
            datetime::days_to_ymd(*reinterpret_cast<const int32_t *>(src), fld);
            date_dtype_default_struct *dst_struct = reinterpret_cast<date_dtype_default_struct *>(dst);
            dst_struct->year = fld.year;
            dst_struct->month = static_cast<int8_t>(fld.month);
            dst_struct->day = static_cast<int8_t>(fld.day);
        }
    };

    struct date_to_struct_assign_kernel {
        struct auxdata_storage {
            kernel_instance<unary_operation_pair_t> kernel;
        };

        /** Does a single copy */
        static void single(char *dst, const char *src, unary_kernel_static_data *extra)
        {
            auxdata_storage& ad = get_auxiliary_data<auxdata_storage>(extra->auxdata);
            datetime::date_ymd fld;
            datetime::days_to_ymd(*reinterpret_cast<const int32_t *>(src), fld);
            // Put the date in our default struct layout
            date_dtype_default_struct tmp_date;
            tmp_date.year = fld.year;
            tmp_date.month = static_cast<int8_t>(fld.month);
            tmp_date.day = static_cast<int8_t>(fld.day);
            // Copy to the destination
            ad.kernel.extra.dst_metadata = extra->dst_metadata;
            ad.kernel.kernel.single(dst, reinterpret_cast<const char *>(&tmp_date), &ad.kernel.extra);
        }
    };

} // anonymous namespace

void dynd::get_date_to_struct_assignment_kernel(const dtype& dst_struct_dtype,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel)
{
    if (dst_struct_dtype.get_kind() != struct_kind) {
        stringstream ss;
        ss << "get_date_to_struct_assignment_kernel: dest dtype " << dst_struct_dtype << " is not a struct dtype";
        throw runtime_error(ss.str());
    }

    if (dst_struct_dtype == date_dtype_default_struct_dtype) {
        out_kernel.kernel.single = &date_to_struct_trivial_assign_kernel::single;
        out_kernel.kernel.strided = NULL;
        out_kernel.extra.auxdata.free();
        return;
    }

    out_kernel.kernel.single = &date_to_struct_assign_kernel::single;
    out_kernel.kernel.strided = NULL;

    make_auxiliary_data<date_to_struct_assign_kernel::auxdata_storage>(out_kernel.extra.auxdata);
    date_to_struct_assign_kernel::auxdata_storage& ad = out_kernel.extra.auxdata.get<date_to_struct_assign_kernel::auxdata_storage>();
    get_dtype_assignment_kernel(dst_struct_dtype, date_dtype_default_struct_dtype, errmode, NULL, ad.kernel);
}

/////////////////////////////////////////
// struct to date assignment


namespace {
    struct struct_to_date_trivial_assign_kernel {
        /** When the source struct is exactly our desired layout */
        static void single(char *dst, const char *src, unary_kernel_static_data *DYND_UNUSED(extra))
        {
            datetime::date_ymd fld;
            const date_dtype_default_struct *src_struct = reinterpret_cast<const date_dtype_default_struct *>(src);
            fld.year = src_struct->year;
            fld.month = src_struct->month;
            fld.day = src_struct->day;
            *reinterpret_cast<int32_t *>(dst) = datetime::ymd_to_days(fld);
        }
    };

    struct struct_to_date_assign_kernel {
        struct auxdata_storage {
            kernel_instance<unary_operation_pair_t> kernel;
        };

        /** Does a single copy */
        static void single(char *dst, const char *src, unary_kernel_static_data *extra)
        {
            auxdata_storage& ad = get_auxiliary_data<auxdata_storage>(extra->auxdata);
            datetime::date_ymd fld;
            // Copy the source struct into our default struct layout
            date_dtype_default_struct tmp_date;
            ad.kernel.extra.src_metadata = extra->src_metadata;
            ad.kernel.kernel.single(reinterpret_cast<char *>(&tmp_date), src, &ad.kernel.extra);
            // Convert to datetime_fields, then to the result date dtype
            fld.day = tmp_date.day;
            fld.month = tmp_date.month;
            fld.year = tmp_date.year;
            *reinterpret_cast<int32_t *>(dst) = datetime::ymd_to_days(fld);
        }
    };

} // anonymous namespace

void dynd::get_struct_to_date_assignment_kernel(const dtype& src_struct_dtype,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel)
{
    if (src_struct_dtype.get_kind() != struct_kind) {
        stringstream ss;
        ss << "get_struct_to_date_assignment_kernel: source dtype " << src_struct_dtype << " is not a struct dtype";
        throw runtime_error(ss.str());
    }

    if (src_struct_dtype == date_dtype_default_struct_dtype) {
        out_kernel.kernel.single = &struct_to_date_trivial_assign_kernel::single;
        out_kernel.kernel.strided = NULL;
        out_kernel.extra.auxdata.free();
        return;
    }

    out_kernel.kernel.single = &struct_to_date_assign_kernel::single;
    out_kernel.kernel.strided = NULL;

    make_auxiliary_data<struct_to_date_assign_kernel::auxdata_storage>(out_kernel.extra.auxdata);
    struct_to_date_assign_kernel::auxdata_storage& ad = out_kernel.extra.auxdata.get<struct_to_date_assign_kernel::auxdata_storage>();
    get_dtype_assignment_kernel(date_dtype_default_struct_dtype, src_struct_dtype, errmode, NULL, ad.kernel);
}
