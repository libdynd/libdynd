//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream> // FOR DEBUG

#include <sstream>
#include <stdexcept>
#include <cstring>
#include <limits>

#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/diagnostics.hpp>

using namespace std;
using namespace dynd;

std::ostream& dynd::operator<<(ostream& o, assign_error_mode errmode)
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
        case assign_error_default:
            o << "default";
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
bool dynd::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt)
{
    const extended_dtype *dst_ext, *src_ext;

    dst_ext = dst_dt.extended();
    src_ext = src_dt.extended();

    if (dst_ext == NULL && src_ext == NULL) {
        switch (src_dt.get_kind()) {
            case pattern_kind: // TODO: raise an error?
                return true;
            case bool_kind:
                switch (dst_dt.get_kind()) {
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
                switch (dst_dt.get_kind()) {
                    case bool_kind:
                        return false;
                    case int_kind:
                        return dst_dt.get_element_size() >= src_dt.get_element_size();
                    case uint_kind:
                        return false;
                    case real_kind:
                        return dst_dt.get_element_size() > src_dt.get_element_size();
                    case complex_kind:
                        return dst_dt.get_element_size() > 2 * src_dt.get_element_size();
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
                break;
            case uint_kind:
                switch (dst_dt.get_kind()) {
                    case bool_kind:
                        return false;
                    case int_kind:
                        return dst_dt.get_element_size() > src_dt.get_element_size();
                    case uint_kind:
                        return dst_dt.get_element_size() >= src_dt.get_element_size();
                    case real_kind:
                        return dst_dt.get_element_size() > src_dt.get_element_size();
                    case complex_kind:
                        return dst_dt.get_element_size() > 2 * src_dt.get_element_size();
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
                break;
            case real_kind:
                switch (dst_dt.get_kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                        return false;
                    case real_kind:
                        return dst_dt.get_element_size() >= src_dt.get_element_size();
                    case complex_kind:
                        return dst_dt.get_element_size() >= 2 * src_dt.get_element_size();
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
            case complex_kind:
                switch (dst_dt.get_kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case real_kind:
                        return false;
                    case complex_kind:
                        return dst_dt.get_element_size() >= src_dt.get_element_size();
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
            case string_kind:
                switch (dst_dt.get_kind()) {
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
                return dst_dt.get_kind() == bytes_kind  &&
                        dst_dt.get_element_size() == src_dt.get_element_size();
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


void dynd::dtype_assign(const dtype& dst_dt, const char *dst_metadata, char *dst_data,
                const dtype& src_dt, const char *src_metadata, const char *src_data,
                assign_error_mode errmode, const eval::eval_context *ectx)
{
    DYND_ASSERT_ALIGNED(dst, 0, dst_dt.get_alignment(), "dst dtype: " << dst_dt << ", src dtype: " << src_dt);
    DYND_ASSERT_ALIGNED(src, 0, src_dt.get_alignment(), "src dtype: " << src_dt << ", dst dtype: " << dst_dt);

    if (errmode == assign_error_default) {
        if (ectx != NULL) {
            errmode = ectx->default_assign_error_mode;
        } else if (dst_dt == src_dt) {
            errmode = assign_error_none;
        } else {
            stringstream ss;
            ss << "assignment from " << src_dt << " to " << dst_dt << " with default error mode requires an eval_context";
            throw runtime_error(ss.str());
        }
    }

    unary_kernel_static_data extra;
    extra.dst_metadata = dst_metadata;
    extra.src_metadata = src_metadata;

    if (dst_dt.extended() == NULL && src_dt.extended() == NULL) {
        // Try to use the simple single-value assignment for built-in types
        unary_operation_pair_t asn = get_builtin_dtype_assignment_function(dst_dt.get_type_id(), src_dt.get_type_id(), errmode);
        if (asn.single != NULL) {
            extra.auxdata = NULL;
            asn.single(dst_data, src_data, &extra);
            return;
        }

        stringstream ss;
        ss << "assignment from " << src_dt << " to " << dst_dt << " with error mode " << errmode << " isn't yet supported";
        throw std::runtime_error(ss.str());
    } else {
        // Fall back to the strided assignment functions for the extended dtypes
        kernel_instance<unary_operation_pair_t> op;
        get_dtype_assignment_kernel(dst_dt, src_dt, errmode, ectx, op);
        extra.auxdata = op.auxdata;
        op.kernel.single(dst_data, src_data, &extra);
        return;
    }
}
