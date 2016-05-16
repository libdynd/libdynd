//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/buffer.hpp>

using namespace std;
using namespace dynd;

void nd::buffer::debug_print(std::ostream &o, const std::string &indent) const {
  o << indent << "------ buffer\n";
  if (m_ptr) {
    o << " address: " << (void *)m_ptr << "\n";
    o << " refcount: " << m_ptr->get_use_count() << "\n";
    o << " type:\n";
    o << "  pointer: " << (void *)get_type().extended() << "\n";
    o << "  type: " << get_type() << "\n";
    if (!get_type().is_builtin()) {
      o << "  type refcount: " << get_type().extended()->get_use_count() << "\n";
    }
    o << " arrmeta:\n";
    o << "  flags: " << get_flags() << " (";
    if (get_flags() & read_access_flag)
      o << "read_access ";
    if (get_flags() & write_access_flag)
      o << "write_access ";
    if (get_flags() & immutable_access_flag)
      o << "immutable ";
    o << ")\n";
    if (!get_type().is_builtin()) {
      o << "  type-specific arrmeta:\n";
      get_type()->arrmeta_debug_print(get()->metadata(), o, indent + "   ");
    }
    o << " data:\n";
    o << "   pointer: " << static_cast<void *>(m_ptr->m_data) << "\n";
    o << "   reference: " << static_cast<void *>(m_ptr->m_owner.get());
    if (!get_owner()) {
      o << " (embedded in array memory)\n";
    } else {
      o << "\n";
    }
    if (get_owner()) {
      get_owner()->debug_print(o, "    ");
    }
  } else {
    o << indent << "NULL\n";
  }
  o << indent << "------" << endl;
}
