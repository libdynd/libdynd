//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream> // FOR DEBUG

#include <sstream>
#include <stdexcept>
#include <cstring>
#include <limits>

#include <dynd/func/callable.hpp>
#include <dynd/func/assignment.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

std::ostream &dynd::operator<<(ostream &o, assign_error_mode errmode)
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
bool dynd::is_lossless_assignment(const ndt::type &dst_tp, const ndt::type &src_tp)
{
  if (dst_tp.is_builtin() && src_tp.is_builtin()) {
    switch (src_tp.get_kind()) {
    case kind_kind: // TODO: raise an error?
      return true;
    case pattern_kind: // TODO: raise an error?
      return true;
    case bool_kind:
      switch (dst_tp.get_kind()) {
      case bool_kind:
      case sint_kind:
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
    case sint_kind:
      switch (dst_tp.get_kind()) {
      case bool_kind:
        return false;
      case sint_kind:
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
      case sint_kind:
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
      case sint_kind:
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
      case sint_kind:
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
      case sint_kind:
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
      return dst_tp.get_kind() == bytes_kind && dst_tp.get_data_size() == src_tp.get_data_size();
    default:
      break;
    }

    throw std::runtime_error("unhandled built-in case in is_lossless_assignmently");
  }

  // Use the available base_type to check the casting
  if (!dst_tp.is_builtin()) {
    // Call with dst_dt (the first parameter) first
    return dst_tp.extended()->is_lossless_assignment(dst_tp, src_tp);
  }
  else {
    // Fall back to src_dt if the dst's extended is NULL
    return src_tp.extended()->is_lossless_assignment(dst_tp, src_tp);
  }
}

void dynd::typed_data_assign(const ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data, const ndt::type &src_tp,
                             const char *src_arrmeta, const char *src_data, assign_error_mode error_mode)
{
  DYND_ASSERT_ALIGNED(dst, 0, dst_tp.get_data_alignment(), "dst type: " << dst_tp << ", src type: " << src_tp);
  DYND_ASSERT_ALIGNED(src, 0, src_dt.get_data_alignment(), "src type: " << src_tp << ", dst type: " << dst_tp);

  nd::array kwd = nd::empty(ndt::option_type::make(ndt::type::make<int>()));
  *reinterpret_cast<int *>(kwd.data()) = static_cast<int>(error_mode);
  std::map<std::string, ndt::type> tp_vars;
  nd::assign::get()->call(dst_tp, dst_arrmeta, dst_data, 1, &src_tp, &src_arrmeta, const_cast<char *const *>(&src_data),
                          1, &kwd, tp_vars);
}
