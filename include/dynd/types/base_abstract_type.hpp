//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API base_abstract_type : public base_type {
  public:
    base_abstract_type(type_id_t id, uint32_t flags, size_t ndim, size_t fixed_ndim)
        : base_type(id, 0, 1, flags | type_flag_symbolic, 0, ndim, fixed_ndim) {}

    size_t get_default_data_size() const {
      std::stringstream ss;
      ss << "Cannot get default data size of type " << type(this, true);
      throw std::runtime_error(ss.str());
    }

    void print_data(std::ostream &DYND_UNUSED(o), const char *DYND_UNUSED(arrmeta),
                    const char *DYND_UNUSED(data)) const {
      throw type_error("Cannot store data of symbolic any kind type");
    }

    void arrmeta_default_construct(char *DYND_UNUSED(arrmeta), bool DYND_UNUSED(blockref_alloc)) const {
      std::stringstream ss;
      ss << "Cannot default construct arrmeta for symbolic type " << type(this, true);
      throw std::runtime_error(ss.str());
    }

    void arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
                                const nd::memory_block &DYND_UNUSED(embedded_reference)) const {
      std::stringstream ss;
      ss << "Cannot copy construct arrmeta for symbolic type " << type(this, true);
      throw std::runtime_error(ss.str());
    }

    void arrmeta_reset_buffers(char *DYND_UNUSED(arrmeta)) const {}

    void arrmeta_finalize_buffers(char *DYND_UNUSED(arrmeta)) const {}

    void arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {}

    void arrmeta_debug_print(const char *DYND_UNUSED(arrmeta), std::ostream &DYND_UNUSED(o),
                             const std::string &DYND_UNUSED(indent)) const {
      std::stringstream ss;
      ss << "Cannot have arrmeta for symbolic type " << type(this, true);
      throw std::runtime_error(ss.str());
    }

    void data_destruct(const char *DYND_UNUSED(arrmeta), char *DYND_UNUSED(data)) const {
      std::stringstream ss;
      ss << "Cannot have data for symbolic type " << type(this, true);
      throw std::runtime_error(ss.str());
    }

    void data_destruct_strided(const char *DYND_UNUSED(arrmeta), char *DYND_UNUSED(data), intptr_t DYND_UNUSED(stride),
                               size_t DYND_UNUSED(count)) const {
      std::stringstream ss;
      ss << "Cannot have data for symbolic type " << type(this, true);
      throw std::runtime_error(ss.str());
    }
  };

} // namespace dynd::ndt
} // namespace dynd
