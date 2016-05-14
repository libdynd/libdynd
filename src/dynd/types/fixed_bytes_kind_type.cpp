//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/exceptions.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/fixed_bytes_kind_type.hpp>

using namespace std;
using namespace dynd;

ndt::fixed_bytes_kind_type::fixed_bytes_kind_type() : base_bytes_type(fixed_bytes_id, 0, 0, type_flag_symbolic, 0) {}

size_t ndt::fixed_bytes_kind_type::get_default_data_size() const {
  stringstream ss;
  ss << "Cannot get default data size of type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::fixed_bytes_kind_type::print_data(std::ostream &DYND_UNUSED(o), const char *DYND_UNUSED(arrmeta),
                                            const char *DYND_UNUSED(data)) const {
  throw type_error("Cannot store data of symbolic fixed_bytes_kind type");
}

void ndt::fixed_bytes_kind_type::print_type(std::ostream &o) const { o << "FixedBytes"; }

bool ndt::fixed_bytes_kind_type::is_expression() const { return false; }

bool ndt::fixed_bytes_kind_type::is_unique_data_owner(const char *DYND_UNUSED(arrmeta)) const { return false; }

ndt::type ndt::fixed_bytes_kind_type::get_canonical_type() const { return type(this, true); }

bool ndt::fixed_bytes_kind_type::is_lossless_assignment(const type &DYND_UNUSED(dst_tp),
                                                        const type &DYND_UNUSED(src_tp)) const {
  return false;
}

void ndt::fixed_bytes_kind_type::get_bytes_range(const char **DYND_UNUSED(out_begin), const char **DYND_UNUSED(out_end),
                                                 const char *DYND_UNUSED(arrmeta),
                                                 const char *DYND_UNUSED(data)) const {
  stringstream ss;
  ss << "Cannot get bytes range for symbolic type " << ndt::type(this, true);
  throw runtime_error(ss.str());
}

bool ndt::fixed_bytes_kind_type::operator==(const base_type &rhs) const {
  if (this == &rhs) {
    return true;
  } else {
    return rhs.is_symbolic() && rhs.get_id() == fixed_bytes_id;
  }
}

void ndt::fixed_bytes_kind_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta),
                                                           bool DYND_UNUSED(blockref_alloc)) const {
  stringstream ss;
  ss << "Cannot default construct arrmeta for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::fixed_bytes_kind_type::arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta),
                                                        const char *DYND_UNUSED(src_arrmeta),
                                                        const nd::memory_block &DYND_UNUSED(embedded_reference)) const {
  stringstream ss;
  ss << "Cannot copy construct arrmeta for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::fixed_bytes_kind_type::arrmeta_reset_buffers(char *DYND_UNUSED(arrmeta)) const {}

void ndt::fixed_bytes_kind_type::arrmeta_finalize_buffers(char *DYND_UNUSED(arrmeta)) const {}

void ndt::fixed_bytes_kind_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {}

void ndt::fixed_bytes_kind_type::arrmeta_debug_print(const char *DYND_UNUSED(arrmeta), std::ostream &DYND_UNUSED(o),
                                                     const std::string &DYND_UNUSED(indent)) const {
  stringstream ss;
  ss << "Cannot have arrmeta for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::fixed_bytes_kind_type::data_destruct(const char *DYND_UNUSED(arrmeta), char *DYND_UNUSED(data)) const {
  stringstream ss;
  ss << "Cannot have data for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::fixed_bytes_kind_type::data_destruct_strided(const char *DYND_UNUSED(arrmeta), char *DYND_UNUSED(data),
                                                       intptr_t DYND_UNUSED(stride), size_t DYND_UNUSED(count)) const {
  stringstream ss;
  ss << "Cannot have data for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

bool ndt::fixed_bytes_kind_type::match(const type &candidate_tp,
                                       std::map<std::string, type> &DYND_UNUSED(tp_vars)) const {
  return candidate_tp.get_id() == fixed_bytes_id;
}
