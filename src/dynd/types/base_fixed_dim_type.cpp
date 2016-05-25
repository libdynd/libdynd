//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/exceptions.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/base_fixed_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/pow_dimsym_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>
#include <dynd/types/typevar_type.hpp>

using namespace std;
using namespace dynd;

ndt::base_fixed_dim_type::base_fixed_dim_type(const type &element_tp)
    : base_dim_type(fixed_dim_id, element_tp, 0, element_tp.get_data_alignment(), sizeof(size_stride_t),
                    type_flag_symbolic, true) {
  // Propagate the inherited flags from the element
  this->flags |= (element_tp.get_flags() & (type_flags_operand_inherited | type_flags_value_inherited));
}

size_t ndt::base_fixed_dim_type::get_default_data_size() const {
  stringstream ss;
  ss << "Cannot get default data size of type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::base_fixed_dim_type::print_data(std::ostream &DYND_UNUSED(o), const char *DYND_UNUSED(arrmeta),
                                          const char *DYND_UNUSED(data)) const {
  throw type_error("Cannot store data of symbolic fixed_dim type");
}

void ndt::base_fixed_dim_type::print_type(std::ostream &o) const { o << "Fixed * " << m_element_tp; }

bool ndt::base_fixed_dim_type::is_expression() const { return m_element_tp.is_expression(); }

bool ndt::base_fixed_dim_type::is_unique_data_owner(const char *DYND_UNUSED(arrmeta)) const { return false; }

void ndt::base_fixed_dim_type::transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset,
                                                     void *extra, type &out_transformed_tp,
                                                     bool &out_was_transformed) const {
  type tmp_tp;
  bool was_transformed = false;
  transform_fn(m_element_tp, arrmeta_offset, extra, tmp_tp, was_transformed);
  if (was_transformed) {
    out_transformed_tp = type(new base_fixed_dim_type(tmp_tp), false);
    out_was_transformed = true;
  } else {
    out_transformed_tp = type(this, true);
  }
}

ndt::type ndt::base_fixed_dim_type::get_canonical_type() const {
  return type(new base_fixed_dim_type(m_element_tp.get_canonical_type()), false);
}

ndt::type ndt::base_fixed_dim_type::at_single(intptr_t DYND_UNUSED(i0), const char **DYND_UNUSED(inout_arrmeta),
                                              const char **DYND_UNUSED(inout_data)) const {
  return m_element_tp;
}

ndt::type ndt::base_fixed_dim_type::get_type_at_dimension(char **DYND_UNUSED(inout_arrmeta), intptr_t i,
                                                          intptr_t total_ndim) const {
  if (i == 0) {
    return type(this, true);
  } else {
    return m_element_tp.get_type_at_dimension(NULL, i - 1, total_ndim + 1);
  }
}

intptr_t ndt::base_fixed_dim_type::get_dim_size(const char *DYND_UNUSED(arrmeta), const char *DYND_UNUSED(data)) const {
  return -1;
}

void ndt::base_fixed_dim_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                                         const char *DYND_UNUSED(arrmeta), const char *DYND_UNUSED(data)) const {
  out_shape[i] = -1;

  // Process the later shape values
  if (i + 1 < ndim) {
    if (!m_element_tp.is_builtin()) {
      m_element_tp.extended()->get_shape(ndim, i + 1, out_shape, NULL, NULL);
    } else {
      stringstream ss;
      ss << "requested too many dimensions from type " << type(this, true);
      throw runtime_error(ss.str());
    }
  }
}

bool ndt::base_fixed_dim_type::is_lossless_assignment(const type &DYND_UNUSED(dst_tp),
                                                      const type &DYND_UNUSED(src_tp)) const {
  return false;
}

bool ndt::base_fixed_dim_type::operator==(const base_type &rhs) const {
  if (this == &rhs) {
    return true;
  }

  if (rhs.get_id() != fixed_dim_id) {
    return false;
  }

  if (static_cast<const base_fixed_dim_type *>(&rhs)->is_sized()) {
    return false;
  }

  const base_fixed_dim_type *dt = static_cast<const base_fixed_dim_type *>(&rhs);
  return m_element_tp == dt->m_element_tp;
}

void ndt::base_fixed_dim_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta),
                                                         bool DYND_UNUSED(blockref_alloc)) const {
  stringstream ss;
  ss << "Cannot default construct arrmeta for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::base_fixed_dim_type::arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta),
                                                      const char *DYND_UNUSED(src_arrmeta),
                                                      const nd::memory_block &DYND_UNUSED(embedded_reference)) const {
  stringstream ss;
  ss << "Cannot copy construct arrmeta for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

size_t
ndt::base_fixed_dim_type::arrmeta_copy_construct_onedim(char *DYND_UNUSED(dst_arrmeta),
                                                        const char *DYND_UNUSED(src_arrmeta),
                                                        const nd::memory_block &DYND_UNUSED(embedded_reference)) const {
  stringstream ss;
  ss << "Cannot copy construct arrmeta for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::base_fixed_dim_type::arrmeta_reset_buffers(char *DYND_UNUSED(arrmeta)) const {}

void ndt::base_fixed_dim_type::arrmeta_finalize_buffers(char *DYND_UNUSED(arrmeta)) const {}

void ndt::base_fixed_dim_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {}

void ndt::base_fixed_dim_type::arrmeta_debug_print(const char *DYND_UNUSED(arrmeta), std::ostream &DYND_UNUSED(o),
                                                   const std::string &DYND_UNUSED(indent)) const {
  stringstream ss;
  ss << "Cannot have arrmeta for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::base_fixed_dim_type::data_destruct(const char *DYND_UNUSED(arrmeta), char *DYND_UNUSED(data)) const {
  stringstream ss;
  ss << "Cannot have data for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::base_fixed_dim_type::data_destruct_strided(const char *DYND_UNUSED(arrmeta), char *DYND_UNUSED(data),
                                                     intptr_t DYND_UNUSED(stride), size_t DYND_UNUSED(count)) const {
  stringstream ss;
  ss << "Cannot have data for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

bool ndt::base_fixed_dim_type::match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const {
  switch (candidate_tp.get_id()) {
  case fixed_dim_id:
    if (candidate_tp.is_symbolic()) {
      return m_element_tp.match(candidate_tp.extended<base_fixed_dim_type>()->get_element_type(), tp_vars);
    } else {
      return m_element_tp.match(candidate_tp.extended<fixed_dim_type>()->get_element_type(), tp_vars);
    }
  default:
    return false;
  }
}

std::map<std::string, std::pair<ndt::type, const char *>>
ndt::base_fixed_dim_type::get_dynamic_type_properties() const {
  std::map<std::string, std::pair<ndt::type, const char *>> properties;
  properties["element_type"] = {ndt::type("type"), reinterpret_cast<const char *>(&m_element_tp)};

  return properties;
}

ndt::type ndt::base_fixed_dim_type::with_element_type(const type &element_tp) const {
  return make_type<fixed_dim_kind_type>(element_tp);
}
