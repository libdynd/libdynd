//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/static_type_instances.hpp>

namespace dynd {

/**
 * A dynd type whose nd::array instances themselves contain
 * dynd types.
 */
class sym_type_type : public base_type {
  ndt::type m_sym_tp;

public:
  sym_type_type(const ndt::type &sym_tp);

  virtual ~sym_type_type();

  void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

  void print_type(std::ostream &o) const;

  bool operator==(const base_type &rhs) const;

/*



  void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
  void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                              memory_block_data *embedded_reference) const;
  void arrmeta_reset_buffers(char *arrmeta) const;
  void arrmeta_finalize_buffers(char *arrmeta) const;
  void arrmeta_destruct(char *arrmeta) const;
  void arrmeta_debug_print(const char *DYND_UNUSED(arrmeta),
                           std::ostream &DYND_UNUSED(o),
                           const std::string &DYND_UNUSED(indent)) const
  {
  }
*/
};

namespace ndt {
  /** Returns type "type" */
  inline const ndt::type make_sym_type_type(const ndt::type &sym_tp)
  {
    return ndt::type(new sym_type_type(sym_tp), false);
  }
} // namespace ndt

} // namespace dynd
