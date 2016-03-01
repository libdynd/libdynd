//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/type.hpp>

using namespace std;
using namespace dynd;

ndt::base_expr_type::base_expr_type(type_id_t type_id, size_t data_size, size_t alignment, flags_type flags,
                                    size_t arrmeta_size, size_t ndim)
    : base_type(type_id, data_size, alignment, flags, arrmeta_size, ndim, 0)
{
}

bool ndt::base_expr_type::is_expression() const { return true; }

ndt::type ndt::base_expr_type::get_canonical_type() const { return get_value_type(); }

void ndt::base_expr_type::arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const
{
  const type &dt = get_operand_type();
  if (!dt.is_builtin()) {
    dt.extended()->arrmeta_default_construct(arrmeta, blockref_alloc);
  }
}

void ndt::base_expr_type::arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                                 const intrusive_ptr<memory_block_data> &embedded_reference) const
{
  const type &dt = get_operand_type();
  if (!dt.is_builtin()) {
    dt.extended()->arrmeta_copy_construct(dst_arrmeta, src_arrmeta, embedded_reference);
  }
}

void ndt::base_expr_type::arrmeta_destruct(char *arrmeta) const
{
  const type &dt = get_operand_type();
  if (!dt.is_builtin()) {
    dt.extended()->arrmeta_destruct(arrmeta);
  }
}

void ndt::base_expr_type::arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const
{
  const type &dt = get_operand_type();
  if (!dt.is_builtin()) {
    dt.extended()->arrmeta_debug_print(arrmeta, o, indent);
  }
}

size_t ndt::base_expr_type::get_iterdata_size(intptr_t DYND_UNUSED(ndim)) const { return 0; }
