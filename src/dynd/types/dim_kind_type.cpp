//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/buffer.hpp>
#include <dynd/types/dim_kind_type.hpp>

using namespace std;
using namespace dynd;

ndt::dim_kind_type::dim_kind_type(type_id_t id, const type &element_tp)
    : base_dim_type(id, element_tp, 0, element_tp.get_data_alignment(), sizeof(size_stride_t), type_flag_symbolic,
                    true) {
  // Propagate the inherited flags from the element
  this->flags |= (element_tp.get_flags() & (type_flags_operand_inherited | type_flags_value_inherited));
}

bool ndt::dim_kind_type::match(const type &candidate_tp, std::map<std::string, type> &DYND_UNUSED(tp_vars)) const {
  return candidate_tp.get_ndim() > 0 && m_element_tp.match(candidate_tp.get_type_at_dimension(NULL, 1));
}

void ndt::dim_kind_type::print_data(std::ostream &DYND_UNUSED(o), const char *DYND_UNUSED(arrmeta),
                                    const char *DYND_UNUSED(data)) const {
  throw std::runtime_error("cannot print data of dim_kind_type");
}

void ndt::dim_kind_type::print_type(std::ostream &o) const { o << "Dim * " << m_element_tp; }

size_t
ndt::dim_kind_type::arrmeta_copy_construct_onedim(char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
                                                  const nd::memory_block &DYND_UNUSED(embedded_reference)) const {
  stringstream ss;
  ss << "Cannot copy construct arrmeta for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

ndt::type ndt::dim_kind_type::get_type_at_dimension(char **DYND_UNUSED(inout_arrmeta), intptr_t i,
                                                    intptr_t total_ndim) const {
  if (i == 0) {
    return type(this, true);
  } else {
    return m_element_tp.get_type_at_dimension(NULL, i - 1, total_ndim + 1);
  }
}

intptr_t ndt::dim_kind_type::get_dim_size(const char *DYND_UNUSED(arrmeta), const char *DYND_UNUSED(data)) const {
  return -1;
}

ndt::type ndt::dim_kind_type::with_element_type(const type &element_tp) const {
  return make_type<dim_kind_type>(element_tp);
}

bool ndt::dim_kind_type::operator==(const base_type &rhs) const {
  return this == &rhs ||
         (rhs.get_id() == dim_kind_id && m_element_tp == reinterpret_cast<const dim_kind_type &>(rhs).m_element_tp);
}

ndt::type ndt::dim_kind_type::construct_type(type_id_t DYND_UNUSED(id), const nd::buffer &args,
                                             const ndt::type &element_type) {
  if (!args.is_null()) {
    throw invalid_argument("Dim dimension kind does not accept arguments");
  }
  return ndt::make_type<ndt::dim_kind_type>(element_type);
}