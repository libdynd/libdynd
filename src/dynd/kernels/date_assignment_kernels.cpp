//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/date_assignment_kernels.hpp>
#include <datetime_strings.h>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// fixedstring to fixedstring assignment

namespace {
    struct string_to_date_assign_kernel {
        struct auxdata_storage {
            date_unit_t dst_unit;
            dtype src_dtype;
            assign_error_mode errmode;
        };

        /** Does a single fixed-string copy */
        static void assign(char *dst, const char *src,
                const auxdata_storage& ad)
        {
            const extended_string_dtype *esd = static_cast<const extended_string_dtype *>(ad.src_dtype.extended());
            // TODO: Kernels should get pointers to metadata too!
            datetime::datetime_unit_t unit;
            datetime::datetime_conversion_rule_t casting;
            switch (ad.dst_unit) {
                case date_unit_day:
                    unit = datetime::datetime_unit_day;
                    break;
                case date_unit_week:
                    unit = datetime::datetime_unit_week;
                    break;
                case date_unit_month:
                    unit = datetime::datetime_unit_month;
                    break;
                case date_unit_year:
                    unit = datetime::datetime_unit_year;
                    break;
            }
            switch (ad.errmode) {
                case assign_error_fractional:
                case assign_error_inexact:
                    casting = datetime::datetime_conversion_strict;
                    break;
                default:
                    casting = datetime::datetime_conversion_relaxed;
            }
            *reinterpret_cast<int32_t *>(dst) = datetime::parse_iso_8601_date(
                                    esd->get_utf8_string(NULL, src, ad.errmode),
                                    unit, casting);

        }

        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const auxdata_storage& ad = get_auxiliary_data<auxdata_storage>(auxdata);
            for (intptr_t i = 0; i < count; ++i) {
                assign(dst, src, ad);

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DYND_UNUSED(dst_stride), const char *src, intptr_t DYND_UNUSED(src_stride),
                            intptr_t, const AuxDataBase *auxdata)
        {
            const auxdata_storage& ad = get_auxiliary_data<auxdata_storage>(auxdata);
            assign(dst, src, ad);
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t DYND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const auxdata_storage& ad = get_auxiliary_data<auxdata_storage>(auxdata);

            // Convert the encoding once, then use memcpy calls for the rest.
            assign(dst, src, ad);
            const char *dst_first = dst;

            for (intptr_t i = 0; i < count; ++i) {
                memcpy(dst, dst_first, 4);

                dst += 4;
            }
        }
    };
} // anonymous namespace

void dynd::get_string_to_date_assignment_kernel(date_unit_t dst_unit,
                const dtype& src_string_dtype,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel)
{
    if (src_string_dtype.get_kind() != string_kind) {
        stringstream ss;
        ss << "get_string_to_date_assignment_kernel: source dtype " << src_string_dtype << " is not a string dtype";
        throw runtime_error(ss.str());
    }

    static specialized_unary_operation_table_t optable = {
        string_to_date_assign_kernel::general_kernel,
        string_to_date_assign_kernel::scalar_kernel,
        string_to_date_assign_kernel::general_kernel,
        string_to_date_assign_kernel::scalar_to_contiguous_kernel};
    out_kernel.specializations = optable;

    make_auxiliary_data<string_to_date_assign_kernel::auxdata_storage>(out_kernel.auxdata);
    string_to_date_assign_kernel::auxdata_storage& ad = out_kernel.auxdata.get<string_to_date_assign_kernel::auxdata_storage>();
    ad.dst_unit = dst_unit;
    ad.errmode = errmode;
    ad.src_dtype = src_string_dtype;
}
