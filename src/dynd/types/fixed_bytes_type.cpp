//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/types/fixed_bytes_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>

using namespace std;
using namespace dynd;

ndt::fixed_bytes_type::fixed_bytes_type(intptr_t data_size, intptr_t data_alignment)
    : base_bytes_type(fixed_bytes_type_id, bytes_kind, data_size, data_alignment, type_flag_none, 0)
{
  if (data_alignment > data_size) {
    std::stringstream ss;
    ss << "Cannot make a bytes[" << data_size << ", align=";
    ss << data_alignment << "] type, its alignment is greater than its size";
    throw std::runtime_error(ss.str());
  }
  if (data_alignment != 1 && data_alignment != 2 && data_alignment != 4 && data_alignment != 8 &&
      data_alignment != 16) {
    std::stringstream ss;
    ss << "Cannot make a bytes[" << data_size << ", align=";
    ss << data_alignment << "] type, its alignment is not a small power of two";
    throw std::runtime_error(ss.str());
  }
  if ((data_size & (data_alignment - 1)) != 0) {
    std::stringstream ss;
    ss << "Cannot make a fixed_bytes[" << data_size << ", align=";
    ss << data_alignment << "] type, its alignment does not divide into its element size";
    throw std::runtime_error(ss.str());
  }
}

ndt::fixed_bytes_type::~fixed_bytes_type() {}

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
    else if (src_tp.get_type_id() == fixed_bytes_type_id) {
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
  else if (rhs.get_type_id() != fixed_bytes_type_id) {
    return false;
  }
  else {
    const fixed_bytes_type *dt = static_cast<const fixed_bytes_type *>(&rhs);
    return get_data_size() == dt->get_data_size() && get_data_alignment() == dt->get_data_alignment();
  }
}
