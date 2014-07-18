//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream> // FOR DEBUG

#include <sstream>
#include <stdexcept>
#include <cstring>
#include <limits>

#include <dynd/typed_data_assign.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

std::ostream& dynd::operator<<(ostream& o, assign_error_mode errmode)
{
    switch (errmode) {
        case assign_error_nocheck:
            o << "nocheck";
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

// Returns true if the destination type can represent *all* the values
// of the source type, false otherwise. This is used, for example,
// to skip any overflow checks when doing value assignments between differing
// types.
bool dynd::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp)
{
    if (dst_tp.is_builtin() && src_tp.is_builtin()) {
        switch (src_tp.get_kind()) {
            case symbolic_kind: // TODO: raise an error?
                return true;
            case bool_kind:
                switch (dst_tp.get_kind()) {
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
                switch (dst_tp.get_kind()) {
                    case bool_kind:
                        return false;
                    case int_kind:
                        return dst_tp.get_data_size() >= src_tp.get_data_size();
                    case uint_kind:
                        return false;
                    case real_kind:
                        return dst_tp.get_data_size() > src_tp.get_data_size();
                    case complex_kind:
                        return dst_tp.get_data_size() > 2 * src_tp.get_data_size();
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
                break;
            case uint_kind:
                switch (dst_tp.get_kind()) {
                    case bool_kind:
                        return false;
                    case int_kind:
                        return dst_tp.get_data_size() > src_tp.get_data_size();
                    case uint_kind:
                        return dst_tp.get_data_size() >= src_tp.get_data_size();
                    case real_kind:
                        return dst_tp.get_data_size() > src_tp.get_data_size();
                    case complex_kind:
                        return dst_tp.get_data_size() > 2 * src_tp.get_data_size();
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
                break;
            case real_kind:
                switch (dst_tp.get_kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                        return false;
                    case real_kind:
                        return dst_tp.get_data_size() >= src_tp.get_data_size();
                    case complex_kind:
                        return dst_tp.get_data_size() >= 2 * src_tp.get_data_size();
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
            case complex_kind:
                switch (dst_tp.get_kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case real_kind:
                        return false;
                    case complex_kind:
                        return dst_tp.get_data_size() >= src_tp.get_data_size();
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
            case string_kind:
                switch (dst_tp.get_kind()) {
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
                return dst_tp.get_kind() == bytes_kind  &&
                        dst_tp.get_data_size() == src_tp.get_data_size();
            default:
                break;
        }

        throw std::runtime_error("unhandled built-in case in is_lossless_assignmently");
    }

    // Use the available base_type to check the casting
    if (!dst_tp.is_builtin()) {
        // Call with dst_dt (the first parameter) first
        return dst_tp.extended()->is_lossless_assignment(dst_tp, src_tp);
    } else {
        // Fall back to src_dt if the dst's extended is NULL
        return src_tp.extended()->is_lossless_assignment(dst_tp, src_tp);
    }
}

void dynd::typed_data_copy(const ndt::type& tp,
                const char *dst_arrmeta, char *dst_data,
                const char *src_arrmeta, const char *src_data)
{
    size_t data_size = tp.get_data_size();
    if (tp.is_pod()) {
        memcpy(dst_data, src_data, data_size);
    } else {
        unary_ckernel_builder k;
        make_assignment_kernel(&k, 0, tp, dst_arrmeta, tp, src_arrmeta,
                               kernel_request_single,
                               &eval::default_eval_context);
        k(dst_data, src_data);
    }
}

void dynd::typed_data_assign(const ndt::type &dst_tp, const char *dst_arrmeta,
                             char *dst_data, const ndt::type &src_tp,
                             const char *src_arrmeta, const char *src_data,
                             const eval::eval_context *ectx)
{
  DYND_ASSERT_ALIGNED(dst, 0, dst_tp.get_data_alignment(),
                      "dst type: " << dst_tp << ", src type: " << src_tp);
  DYND_ASSERT_ALIGNED(src, 0, src_dt.get_data_alignment(),
                      "src type: " << src_tp << ", dst type: " << dst_tp);

  unary_ckernel_builder k;
  make_assignment_kernel(&k, 0, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
                         kernel_request_single, ectx);
  k(dst_data, src_data);
}

void dynd::typed_data_assign(const ndt::type &dst_tp, const char *dst_arrmeta,
                             char *dst_data, const nd::array &src_arr,
                             const eval::eval_context *ectx)
{
  typed_data_assign(dst_tp, dst_arrmeta, dst_data, src_arr.get_type(),
                    src_arr.get_arrmeta(), src_arr.get_readonly_originptr(),
                    ectx);
}
