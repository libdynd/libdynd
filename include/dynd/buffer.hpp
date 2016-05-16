//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/memblock/array_memory_block.hpp>

namespace dynd {
namespace nd {

  class DYNDT_API buffer : public intrusive_ptr<const array_preamble> {
  public:
    using intrusive_ptr<const array_preamble>::intrusive_ptr;

    buffer() = default;

    /** The type */
    const ndt::type &get_type() const { return m_ptr->m_tp; }

    const memory_block &get_owner() const { return m_ptr->m_owner; }

    /** The flags, including access permissions. */
    uint64_t get_flags() const { return m_ptr->m_flags; }

    char *data() const {
      if (m_ptr->m_flags & write_access_flag) {
        return m_ptr->m_data;
      }

      throw std::runtime_error("tried to write to a dynd array that is not writable");
    }

    const char *cdata() const { return m_ptr->m_data; }

    memory_block get_data_memblock() const {
      if (m_ptr->m_owner) {
        return m_ptr->m_owner;
      }

      return *this;
    }

    bool is_immutable() const { return (m_ptr->m_flags & immutable_access_flag) != 0; }
  };

  inline memory_block::memory_block(const buffer &other)
      : intrusive_ptr<base_memory_block>(const_cast<array_preamble *>(other.get()), true) {}

} // namespace dynd::nd
} // namespace dynd
