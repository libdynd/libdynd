//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream> // FOR DEBUG
#include <typeinfo> // FOR DEBUG

#include <sstream>
#include <stdexcept>
#include <cstring>
#include <limits>

#include <dnd/dtype_assign.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>
#include <dnd/kernels/assignment_kernels.hpp>
#include <dnd/diagnostics.hpp>

#ifdef __GNUC__
// The -Weffc++ flag warns about derived classes not having a virtual destructor.
// Here, this is explicitly done, because we are only using derived classes
// to inherit a static function, they are never instantiated.
//
// NOTE: The documentation says this is only for g++ 4.6.0 and up.
#pragma GCC diagnostic ignored "-Weffc++"
#endif


using namespace std;
using namespace dnd;

std::ostream& dnd::operator<<(ostream& o, assign_error_mode errmode)
{
    switch (errmode) {
    case assign_error_none:
        o << "none";
        break;
    case assign_error_overflow:
        o << "overflow";
        break;
    case assign_error_fractional:
        o << "fractional";
        break;
    case assign_error_inexact:
        o << "inexact";
        break;
    default:
        o << "invalid error mode(" << (int)errmode << ")";
        break;
    }

    return o;
}

// Returns true if the destination dtype can represent *all* the values
// of the source dtype, false otherwise. This is used, for example,
// to skip any overflow checks when doing value assignments between differing
// types.
bool dnd::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt)
{
    const extended_dtype *dst_ext, *src_ext;

    dst_ext = dst_dt.extended();
    src_ext = src_dt.extended();

    if (dst_ext == NULL && src_ext == NULL) {
        switch (src_dt.kind()) {
            case pattern_kind: // TODO: raise an error?
                return true;
            case bool_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case real_kind:
                    case complex_kind:
                        return true;
                    case string_kind:
                        return dst_dt.itemsize() > 0;
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
                break;
            case int_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                        return false;
                    case int_kind:
                        return dst_dt.itemsize() >= src_dt.itemsize();
                    case uint_kind:
                        return false;
                    case real_kind:
                        return dst_dt.itemsize() > src_dt.itemsize();
                    case complex_kind:
                        return dst_dt.itemsize() > 2 * src_dt.itemsize();
                    case string_kind:
                        // Conservative value for 64-bit, could
                        // check speciifically based on the type_id.
                        return dst_dt.itemsize() >= 21;
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
                break;
            case uint_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                        return false;
                    case int_kind:
                        return dst_dt.itemsize() > src_dt.itemsize();
                    case uint_kind:
                        return dst_dt.itemsize() >= src_dt.itemsize();
                    case real_kind:
                        return dst_dt.itemsize() > src_dt.itemsize();
                    case complex_kind:
                        return dst_dt.itemsize() > 2 * src_dt.itemsize();
                    case string_kind:
                        // Conservative value for 64-bit, could
                        // check speciifically based on the type_id.
                        return dst_dt.itemsize() >= 21;
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
                break;
            case real_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                        return false;
                    case real_kind:
                        return dst_dt.itemsize() >= src_dt.itemsize();
                    case complex_kind:
                        return dst_dt.itemsize() >= 2 * src_dt.itemsize();
                    case string_kind:
                        return dst_dt.itemsize() >= 32;
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
            case complex_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case real_kind:
                        return false;
                    case complex_kind:
                        return dst_dt.itemsize() >= src_dt.itemsize();
                    case string_kind:
                        return dst_dt.itemsize() >= 64;
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
            case string_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case real_kind:
                    case complex_kind:
                        return false;
                    case string_kind:
                        return src_dt.type_id() == dst_dt.type_id() &&
                                dst_dt.itemsize() >= src_dt.itemsize();
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
            case bytes_kind:
                return dst_dt.kind() == bytes_kind  &&
                        dst_dt.itemsize() == src_dt.itemsize();
            default:
                break;
        }

        throw std::runtime_error("unhandled built-in case in is_lossless_assignmently");
    }

    // Use the available extended_dtype to check the casting
    if (src_ext != NULL) {
        return src_ext->is_lossless_assignment(dst_dt, src_dt);
    }
    else {
        return dst_ext->is_lossless_assignment(dst_dt, src_dt);
    }
}


void dnd::dtype_assign(const dtype& dst_dt, char *dst, const dtype& src_dt, const char *src, assign_error_mode errmode)
{
    DND_ASSERT_ALIGNED(dst, 0, dst_dt.alignment(), "dst dtype: " << dst_dt << ", src dtype: " << src_dt);
    DND_ASSERT_ALIGNED(src, 0, src_dt.alignment(), "src dtype: " << src_dt << ", dst dtype: " << dst_dt);
    if (dst_dt.extended() == NULL && src_dt.extended() == NULL) {
        // Try to use the simple single-value assignment for built-in types
        assignment_function_t asn = get_builtin_dtype_assignment_function(dst_dt.type_id(), src_dt.type_id(), errmode);
        if (asn != NULL) {
            asn(dst, src);
            return;
        }

        stringstream ss;
        ss << "assignment from " << src_dt << " to " << dst_dt << " isn't yet supported";
        throw std::runtime_error(ss.str());
    } else {
        // Fall back to the strided assignment functions for the extended dtypes
        unary_specialization_kernel_instance op;
        get_dtype_assignment_kernel(dst_dt, src_dt, errmode, op);
        op.specializations[scalar_unary_specialization](dst, 0, src, 0, 1, op.auxdata);
        return;
    }
}

/*

// A multiple unaligned assignment function which uses one of the single assignment functions as proxy
namespace {
    struct multiple_unaligned_auxiliary_data {
        assign_function_t assign;
        int dst_itemsize, src_itemsize;
    };
}
static void assign_multiple_unaligned(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                                    intptr_t count, const AuxDataBase *auxdata)
{
    const multiple_unaligned_auxiliary_data &mgdata = get_auxiliary_data<multiple_unaligned_auxiliary_data>(auxdata);

    char *dst_cached = reinterpret_cast<char *>(dst);
    const char *src_cached = reinterpret_cast<const char *>(src);

    int dst_itemsize = mgdata.dst_itemsize, src_itemsize = mgdata.src_itemsize;
    // TODO: Probably want to relax the assumption of at most 8 bytes
    int64_t d;
    int64_t s;

    assign_function_t asn = mgdata.assign;

    for (intptr_t i = 0; i < count; ++i) {
        memcpy(&s, src_cached, src_itemsize);
        asn(&d, &s);
        memcpy(dst_cached, &d, dst_itemsize);
        dst_cached += dst_stride;
        src_cached += src_stride;
    }
}
*/


void dnd::dtype_strided_assign(const dtype& dst_dt, char *dst, intptr_t dst_stride,
                            const dtype& src_dt, const char *src, intptr_t src_stride,
                            intptr_t count, assign_error_mode errmode)
{
    kernel_instance<unary_operation_t> op;
    get_dtype_assignment_kernel(dst_dt, dst_stride,
                                src_dt, src_stride,
                                errmode, op);
    op.kernel(dst, dst_stride, src, src_stride, count, op.auxdata);
}

