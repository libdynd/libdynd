//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/exceptions.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/int_kind_sym_type.hpp>

using namespace std;
using namespace dynd;

ndt::int_kind_sym_type::int_kind_sym_type() : base_type(int_sym_id, 0, 1, type_flag_symbolic, 0, 0, 0) {}

size_t ndt::int_kind_sym_type::get_default_data_size() const {
  stringstream ss;
  ss << "Cannot get default data size of type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::int_kind_sym_type::print_data(std::ostream &DYND_UNUSED(o), const char *DYND_UNUSED(arrmeta),
                                        const char *DYND_UNUSED(data)) const {
  throw type_error("Cannot store data of symbolic kind_sym type");
}

void ndt::int_kind_sym_type::print_type(std::ostream &o) const { o << "Int"; }

bool ndt::int_kind_sym_type::is_expression() const { return false; }

bool ndt::int_kind_sym_type::is_unique_data_owner(const char *DYND_UNUSED(arrmeta)) const { return false; }

void ndt::int_kind_sym_type::transform_child_types(type_transform_fn_t DYND_UNUSED(transform_fn),
                                                   intptr_t DYND_UNUSED(arrmeta_offset), void *DYND_UNUSED(extra),
                                                   type &out_transformed_tp,
                                                   bool &DYND_UNUSED(out_was_transformed)) const {
  out_transformed_tp = type(this, true);
}

ndt::type ndt::int_kind_sym_type::get_canonical_type() const { return type(this, true); }

bool ndt::int_kind_sym_type::is_lossless_assignment(const type &DYND_UNUSED(dst_tp),
                                                    const type &DYND_UNUSED(src_tp)) const {
  return false;
}

bool ndt::int_kind_sym_type::operator==(const base_type &rhs) const {
  if (this == &rhs) {
    return true;
  } else {
    return rhs.get_id() == int_sym_id;
  }
}

void ndt::int_kind_sym_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta),
                                                       bool DYND_UNUSED(blockref_alloc)) const {
  stringstream ss;
  ss << "Cannot default construct arrmeta for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::int_kind_sym_type::arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta),
                                                    const char *DYND_UNUSED(src_arrmeta),
                                                    const nd::memory_block &DYND_UNUSED(embedded_reference)) const {
  stringstream ss;
  ss << "Cannot copy construct arrmeta for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

size_t ndt::int_kind_sym_type::arrmeta_copy_construct_onedim(char *DYND_UNUSED(dst_arrmeta),
                                                             const char *DYND_UNUSED(src_arrmeta),
                                                             memory_block_data *DYND_UNUSED(embedded_reference)) const {
  stringstream ss;
  ss << "Cannot copy construct arrmeta for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::int_kind_sym_type::arrmeta_reset_buffers(char *DYND_UNUSED(arrmeta)) const {}

void ndt::int_kind_sym_type::arrmeta_finalize_buffers(char *DYND_UNUSED(arrmeta)) const {}

void ndt::int_kind_sym_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {}

void ndt::int_kind_sym_type::arrmeta_debug_print(const char *DYND_UNUSED(arrmeta), std::ostream &DYND_UNUSED(o),
                                                 const std::string &DYND_UNUSED(indent)) const {
  stringstream ss;
  ss << "Cannot have arrmeta for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::int_kind_sym_type::data_destruct(const char *DYND_UNUSED(arrmeta), char *DYND_UNUSED(data)) const {
  stringstream ss;
  ss << "Cannot have data for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::int_kind_sym_type::data_destruct_strided(const char *DYND_UNUSED(arrmeta), char *DYND_UNUSED(data),
                                                   intptr_t DYND_UNUSED(stride), size_t DYND_UNUSED(count)) const {
  stringstream ss;
  ss << "Cannot have data for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

bool ndt::int_kind_sym_type::match(const type &candidate_tp, std::map<std::string, type> &DYND_UNUSED(tp_vars)) const {
  // Matches against the 'kind' of the candidate type
  type_id_t base_id = candidate_tp.get_base_id();
  return base_id == uint_kind_id || base_id == int_kind_id;
}
