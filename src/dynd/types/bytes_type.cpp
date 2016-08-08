//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/exceptions.hpp>
#include <dynd/parse_util.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/datashape_parser.hpp>
#include <dynd/types/fixed_bytes_type.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

void ndt::bytes_type::get_bytes_range(const char **out_begin, const char **out_end, const char *DYND_UNUSED(arrmeta),
                                      const char *data) const {
  *out_begin = reinterpret_cast<const bytes *>(data)->begin();
  *out_end = reinterpret_cast<const bytes *>(data)->end();
}

void ndt::bytes_type::set_bytes_data(const char *DYND_UNUSED(arrmeta), char *data, const char *bytes_begin,
                                     const char *bytes_end) const {
  bytes *d = reinterpret_cast<bytes *>(data);
  if (d->begin() != NULL) {
    throw runtime_error("assigning to a bytes data element requires that it be "
                        "initialized to NULL");
  }

  // Allocate the output array data, then copy it
  d->assign(bytes_begin, bytes_end - bytes_begin);
}

void ndt::bytes_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta), const char *data) const {
  if (reinterpret_cast<const bytes *>(data)->empty()) {
    o << "NULL";
  } else {
    // Print as hexadecimal
    hexadecimal_print_summarized(o, reinterpret_cast<const bytes *>(data)->data(),
                                 reinterpret_cast<const bytes *>(data)->size(), 80);
  }
}

void ndt::bytes_type::print_type(std::ostream &o) const {
  o << "bytes";
  if (m_alignment != 1) {
    o << "[align=" << m_alignment << "]";
  }
}

bool ndt::bytes_type::is_unique_data_owner(const char *DYND_UNUSED(arrmeta)) const { return true; }

ndt::type ndt::bytes_type::get_canonical_type() const { return type(this, true); }

void ndt::bytes_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *DYND_UNUSED(arrmeta),
                                const char *data) const {
  if (data == NULL) {
    out_shape[i] = -1;
  } else {
    const bytes *d = reinterpret_cast<const bytes *>(data);
    out_shape[i] = d->end() - d->begin();
  }
  if (i + 1 < ndim) {
    stringstream ss;
    ss << "requested too many dimensions from type " << type(this, true);
    throw runtime_error(ss.str());
  }
}

bool ndt::bytes_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const {
  if (dst_tp.extended() == this) {
    if (src_tp.get_base_id() == bytes_kind_id) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

bool ndt::bytes_type::operator==(const base_type &rhs) const {
  if (this == &rhs) {
    return true;
  } else if (rhs.get_id() != bytes_id) {
    return false;
  } else {
    const bytes_type *dt = static_cast<const bytes_type *>(&rhs);
    return m_alignment == dt->m_alignment;
  }
}

void ndt::bytes_type::data_destruct(const char *DYND_UNUSED(arrmeta), char *data) const {
  reinterpret_cast<bytes *>(data)->~bytes();
}

void ndt::bytes_type::data_destruct_strided(const char *DYND_UNUSED(arrmeta), char *data, intptr_t stride,
                                            size_t count) const {
  for (size_t i = 0; i != count; ++i) {
    reinterpret_cast<bytes *>(data)->~bytes();
    data += stride;
  }
}

std::map<std::string, std::pair<ndt::type, const char *>> ndt::bytes_type::get_dynamic_type_properties() const {
  std::map<std::string, std::pair<ndt::type, const char *>> properties;
  properties["target_alignment"] = {ndt::type("size"), reinterpret_cast<const char *>(&m_alignment)};

  return properties;
}

// bytes_type : bytes[align=<alignment>]
ndt::type ndt::bytes_type::parse_type_args(type_id_t DYND_UNUSED(id), const char *&rbegin, const char *end,
                                           std::map<std::string, ndt::type> &DYND_UNUSED(symtable)) {
  const char *begin = rbegin;
  if (datashape::parse_token(begin, end, '[')) {
    if (datashape::parse_token(begin, end, "align")) {
      // bytes type with an alignment
      if (!datashape::parse_token(begin, end, '=')) {
        throw datashape::internal_parse_error(begin, "expected an =");
      }
      std::string align_val = datashape::parse_number(begin, end);
      if (align_val.empty()) {
        throw datashape::internal_parse_error(begin, "expected an integer");
      }
      if (!datashape::parse_token(begin, end, ']')) {
        throw datashape::internal_parse_error(begin, "expected closing ']'");
      }
      rbegin = begin;
      return ndt::make_type<ndt::bytes_type>(atoi(align_val.c_str()));
    }
    throw datashape::internal_parse_error(begin, "expected 'align'");
  } else {
    return ndt::make_type<ndt::bytes_type>(1);
  }
}