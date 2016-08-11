//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/type_type.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

ndt::type_type::type_type(type_id_t id)
    : base_type(id, sizeof(ndt::type), sizeof(ndt::type), type_flag_zeroinit | type_flag_destructor, 0, 0, 0) {}

void ndt::type_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta), const char *data) const {
  o << *reinterpret_cast<const ndt::type *>(data);
}

void ndt::type_type::print_type(std::ostream &o) const { o << "type"; }

bool ndt::type_type::operator==(const base_type &rhs) const { return this == &rhs || rhs.get_id() == type_id; }

void ndt::type_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta), bool DYND_UNUSED(blockref_alloc)) const {}

void ndt::type_type::arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
                                            const nd::memory_block &DYND_UNUSED(embedded_reference)) const {}

void ndt::type_type::arrmeta_reset_buffers(char *DYND_UNUSED(arrmeta)) const {}

void ndt::type_type::arrmeta_finalize_buffers(char *DYND_UNUSED(arrmeta)) const {}

void ndt::type_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {}

void ndt::type_type::data_destruct(const char *DYND_UNUSED(arrmeta), char *data) const {
  reinterpret_cast<type *>(data)->~type();
}

void ndt::type_type::data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const {
  for (size_t i = 0; i != count; ++i, data += stride) {
    data_destruct(arrmeta, data);
  }
}
