//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/base_memory_type.hpp>

using namespace std;
using namespace dynd;

base_memory_type::~base_memory_type()
{
}

ndt::type base_memory_type::get_canonical_type() const
{
    return m_target_tp;
}

bool base_memory_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_kind() != memory_kind) {
        return false;
    } else {
        const base_memory_type *dt = static_cast<const base_memory_type*>(&rhs);
        return m_target_tp == dt->m_target_tp;
    }
}
