//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream> // FOR DEBUG

#include <sstream>
#include <stdexcept>
#include <cstring>
#include <limits>

#include <dnd/dtype_assign.hpp>
#include <dnd/dtypes/convert_dtype.hpp>
#include <dnd/kernels/assignment_kernels.hpp>
#include <dnd/diagnostics.hpp>

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
                        return dst_dt.element_size() >= src_dt.element_size();
                    case uint_kind:
                        return false;
                    case real_kind:
                        return dst_dt.element_size() > src_dt.element_size();
                    case complex_kind:
                        return dst_dt.element_size() > 2 * src_dt.element_size();
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
                        return dst_dt.element_size() > src_dt.element_size();
                    case uint_kind:
                        return dst_dt.element_size() >= src_dt.element_size();
                    case real_kind:
                        return dst_dt.element_size() > src_dt.element_size();
                    case complex_kind:
                        return dst_dt.element_size() > 2 * src_dt.element_size();
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
                        return dst_dt.element_size() >= src_dt.element_size();
                    case complex_kind:
                        return dst_dt.element_size() >= 2 * src_dt.element_size();
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
                        return dst_dt.element_size() >= src_dt.element_size();
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
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
            case bytes_kind:
                return dst_dt.kind() == bytes_kind  &&
                        dst_dt.element_size() == src_dt.element_size();
            default:
                break;
        }

        throw std::runtime_error("unhandled built-in case in is_lossless_assignmently");
    }

    // Use the available extended_dtype to check the casting
    if (dst_ext != NULL) {
        // Call with dst_dt (the first parameter) first
        return dst_ext->is_lossless_assignment(dst_dt, src_dt);
    } else {
        // Fall back to src_dt if the dst's extended is NULL
        return src_ext->is_lossless_assignment(dst_dt, src_dt);
    }
}


void dnd::dtype_assign(const dtype& dst_dt, char *dst, const dtype& src_dt, const char *src, assign_error_mode errmode)
{
    DND_ASSERT_ALIGNED(dst, 0, dst_dt.alignment(), "dst dtype: " << dst_dt << ", src dtype: " << src_dt);
    DND_ASSERT_ALIGNED(src, 0, src_dt.alignment(), "src dtype: " << src_dt << ", dst dtype: " << dst_dt);
    if (dst_dt.get_memory_management() != pod_memory_management) {
        throw runtime_error("dtype_assign can only be used with POD destination memory");
    }

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

void dnd::dtype_strided_assign(const dtype& dst_dt, char *dst, intptr_t dst_stride,
                            const dtype& src_dt, const char *src, intptr_t src_stride,
                            intptr_t count, assign_error_mode errmode)
{
    if (dst_dt.get_memory_management() != pod_memory_management) {
        throw runtime_error("dtype_strided_assign can only be used with POD destination memory");
    }

    unary_specialization_kernel_instance op;
    get_dtype_assignment_kernel(dst_dt, src_dt,
                                errmode, op);
    op.specializations[get_unary_specialization(dst_stride, dst_dt.element_size(), src_stride, src_dt.element_size())](
                dst, dst_stride, src, src_stride, count, op.auxdata);
}

