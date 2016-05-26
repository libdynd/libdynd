//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/types/fixed_bytes_type.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>

using namespace std;
using namespace dynd;

void ndt::fixed_bytes_type::get_bytes_range(const char **out_begin, const char **out_end,
                                            const char *DYND_UNUSED(arrmeta), const char *data) const
{
  *out_begin = data;
  *out_end = data + get_data_size();
}

void ndt::fixed_bytes_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
  hexadecimal_print_summarized(o, data, get_data_size(), 80);
}

void ndt::fixed_bytes_type::print_type(std::ostream &o) const
{
  o << "fixed_bytes[" << get_data_size();
  size_t alignment = get_data_alignment();
  if (alignment != 1) {
    o << ", align=" << get_data_alignment();
  }
  o << "]";
}

bool ndt::fixed_bytes_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    }
    else if (src_tp.get_id() == fixed_bytes_id) {
      const fixed_bytes_type *src_fs = static_cast<const fixed_bytes_type *>(src_tp.extended());
      return get_data_size() == src_fs->get_data_size();
    }
    else {
      return false;
    }
  }
  else {
    return false;
  }
}

bool ndt::fixed_bytes_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_id() != fixed_bytes_id) {
    return false;
  }
  else {
    const fixed_bytes_type *dt = static_cast<const fixed_bytes_type *>(&rhs);
    return get_data_size() == dt->get_data_size() && get_data_alignment() == dt->get_data_alignment();
  }
}
