//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/type_type.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

ndt::type_type::type_type()
    : base_type(dynd::type_id, sizeof(ndt::type), sizeof(ndt::type), type_flag_zeroinit | type_flag_destructor, 0, 0, 0)
{
}

ndt::type_type::type_type(const type &pattern_tp)
    : base_type(dynd::type_id, sizeof(ndt::type), sizeof(ndt::type), type_flag_zeroinit | type_flag_destructor, 0, 0,
                0),
      m_pattern_tp(pattern_tp)
{
  if (!m_pattern_tp.is_symbolic()) {
    throw type_error("type_type must have a symbolic type for a pattern");
  }
}

void ndt::type_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
  o << *reinterpret_cast<const ndt::type *>(data);
}

void ndt::type_type::print_type(std::ostream &o) const
{
  o << "type";
  if (!m_pattern_tp.is_null()) {
    o << " | " << m_pattern_tp;
  }
}

bool ndt::type_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_id() != type_id) {
    return false;
  }
  else {
    return m_pattern_tp == static_cast<const type_type *>(&rhs)->m_pattern_tp;
  }
}

void ndt::type_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta), bool DYND_UNUSED(blockref_alloc)) const {}

void ndt::type_type::arrmeta_copy_construct(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    const intrusive_ptr<memory_block_data> &DYND_UNUSED(embedded_reference)) const
{
}

void ndt::type_type::arrmeta_reset_buffers(char *DYND_UNUSED(arrmeta)) const {}

void ndt::type_type::arrmeta_finalize_buffers(char *DYND_UNUSED(arrmeta)) const {}

void ndt::type_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {}

void ndt::type_type::data_destruct(const char *DYND_UNUSED(arrmeta), char *data) const
{
  reinterpret_cast<type *>(data)->~type();
}

void ndt::type_type::data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const
{
  for (size_t i = 0; i != count; ++i, data += stride) {
    data_destruct(arrmeta, data);
  }
}
