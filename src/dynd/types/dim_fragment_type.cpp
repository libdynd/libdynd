//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/dim_fragment_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

ndt::dim_fragment_type::dim_fragment_type(intptr_t ndim, const intptr_t *tagged_dims)
    : base_dim_type(dim_fragment_id, make_type<void>(), 0, 1, 0, type_flag_symbolic, false),
      m_tagged_dims(ndim, tagged_dims) {
  this->m_ndim = static_cast<uint8_t>(ndim);
}

static inline ndt::type get_tagged_dims_from_type(intptr_t ndim, const ndt::type &tp, intptr_t *out_tagged_dims) {
  ndt::type dtp = tp.without_memory_type();
  for (int i = 0; i < ndim; ++i) {
    switch (dtp.get_id()) {
    case fixed_dim_id:
      if (dtp.is_symbolic()) {
        out_tagged_dims[i] = -2;
      } else {
        out_tagged_dims[i] = dtp.extended<ndt::fixed_dim_type>()->get_fixed_dim_size();
      }
      break;
    case var_dim_id:
      out_tagged_dims[i] = -1;
      break;
    default: {
      stringstream ss;
      ss << "dim_fragment_type failed to get shape from type " << tp;
      throw type_error(ss.str());
    }
    }
    dtp = dtp.extended<ndt::base_dim_type>()->get_element_type();
  }
  return dtp;
}

static inline bool broadcast_tagged_dims_from_type(intptr_t ndim, ndt::type tp, const intptr_t *tagged_dims,
                                                   intptr_t *out_tagged_dims) {
  tp = tp.without_memory_type();
  for (intptr_t i = 0; i < ndim; ++i) {
    intptr_t tagged_dim = tagged_dims[i], dim_size;
    switch (tp.get_id()) {
    case fixed_dim_id:
      if (tp.is_symbolic()) {
        if (tagged_dim < 0) {
          out_tagged_dims[i] = -2;
        }
      } else {
        dim_size = tp.extended<ndt::fixed_dim_type>()->get_fixed_dim_size();
        if (tagged_dim < 0 || tagged_dim == 1) {
          out_tagged_dims[i] = dim_size;
        } else if (tagged_dim != dim_size && dim_size != 1) {
          return false;
        }
      }
      break;
    case var_dim_id:
      // All broadcasting is done dynamically for var
      break;
    default: {
      stringstream ss;
      ss << "dim_fragment_type failed to get shape from type " << tp;
      throw type_error(ss.str());
    }
    }
    tp = tp.extended<ndt::base_dim_type>()->get_element_type();
  }
  return true;
}

ndt::dim_fragment_type::dim_fragment_type(intptr_t ndim, const type &tp)
    : base_dim_type(dim_fragment_id, make_type<void>(), 0, 1, 0, type_flag_symbolic, false), m_tagged_dims(ndim) {
  if (ndim > tp.get_ndim()) {
    stringstream ss;
    ss << "Tried to make a dimension fragment from type " << tp << " with " << ndim
       << " dimensions, but the type only has " << tp.get_ndim() << " dimensions";
    throw type_error(ss.str());
  }
  get_tagged_dims_from_type(ndim, tp, m_tagged_dims.get());
  this->m_ndim = static_cast<uint8_t>(ndim);
}

ndt::type ndt::dim_fragment_type::broadcast_with_type(intptr_t ndim, const type &tp) const {
  if (ndim == 0) {
    return type(this, true);
  }
  intptr_t this_ndim = get_ndim();
  // In each case, we fill in the leading dimensions from the
  // higher-dimension input, then broadcast the rest together
  if (ndim > this_ndim) {
    dimvector shape(ndim);
    type dtp = get_tagged_dims_from_type(ndim - this_ndim, tp, shape.get());
    if (!broadcast_tagged_dims_from_type(this_ndim, dtp, get_tagged_dims(), shape.get() + (ndim - this_ndim))) {
      return type();
    }
    return make_dim_fragment(ndim, shape.get());
  } else if (ndim < this_ndim) {
    dimvector shape(this_ndim);
    memcpy(shape.get(), get_tagged_dims(), (this_ndim - ndim) * sizeof(intptr_t));
    if (!broadcast_tagged_dims_from_type(ndim, tp, get_tagged_dims() + (this_ndim - ndim),
                                         shape.get() + (this_ndim - ndim))) {
      return type();
    }
    return make_dim_fragment(this_ndim, shape.get());
  } else {
    dimvector shape(ndim);
    memcpy(shape.get(), get_tagged_dims(), this_ndim * sizeof(intptr_t));
    if (!broadcast_tagged_dims_from_type(ndim, tp, get_tagged_dims(), shape.get())) {
      return type();
    }
    return make_dim_fragment(ndim, shape.get());
  }
}

ndt::type ndt::dim_fragment_type::apply_to_dtype(const type &dtp) const {
  intptr_t ndim = get_ndim();
  if (ndim > 0) {
    type tp = dtp;
    for (intptr_t i = ndim - 1; i >= 0; --i) {
      switch (m_tagged_dims[i]) {
      case dim_fragment_var:
        tp = make_type<var_dim_type>(tp);
        break;
      case dim_fragment_fixed_sym:
        tp = make_fixed_dim_kind(tp);
        break;
      default:
        tp = make_fixed_dim(m_tagged_dims[i], tp);
        break;
      }
    }
    return tp;
  } else {
    return dtp;
  }
}

void ndt::dim_fragment_type::print_data(std::ostream &DYND_UNUSED(o), const char *DYND_UNUSED(arrmeta),
                                        const char *DYND_UNUSED(data)) const {
  throw type_error("Cannot store data of dim_fragment type");
}

void ndt::dim_fragment_type::print_type(std::ostream &o) const {
  o << "dim_fragment[";
  intptr_t ndim = get_ndim();
  for (intptr_t i = 0; i < ndim; ++i) {
    if (m_tagged_dims[i] == dim_fragment_var) {
      o << "var * ";
    } else if (m_tagged_dims[i] == dim_fragment_fixed_sym) {
      o << "Fixed * ";
    } else {
      o << "fixed[" << m_tagged_dims[i] << "]";
    }
  }
  o << "void]";
}

ndt::type ndt::dim_fragment_type::apply_linear_index(intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
                                                     size_t DYND_UNUSED(current_i), const type &DYND_UNUSED(root_tp),
                                                     bool DYND_UNUSED(leading_dimension)) const {
  throw type_error("Cannot store data of dim_fragment type");
}

intptr_t ndt::dim_fragment_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices), const char *DYND_UNUSED(arrmeta),
    const type &DYND_UNUSED(result_tp), char *DYND_UNUSED(out_arrmeta),
    const intrusive_ptr<memory_block_data> &DYND_UNUSED(embedded_reference), size_t DYND_UNUSED(current_i),
    const type &DYND_UNUSED(root_tp), bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
    intrusive_ptr<memory_block_data> &DYND_UNUSED(inout_dataref)) const {
  throw type_error("Cannot store data of dim_fragment type");
}

intptr_t ndt::dim_fragment_type::get_dim_size(const char *DYND_UNUSED(arrmeta), const char *DYND_UNUSED(data)) const {
  if (get_ndim() > 0) {
    return m_tagged_dims[0] >= 0 ? m_tagged_dims[0] : -1;
  } else {
    return -1;
  }
}

bool ndt::dim_fragment_type::is_lossless_assignment(const type &DYND_UNUSED(dst_tp),
                                                    const type &DYND_UNUSED(src_tp)) const {
  return false;
}

bool ndt::dim_fragment_type::operator==(const base_type &rhs) const {
  if (this == &rhs) {
    return true;
  } else if (rhs.get_id() != dim_fragment_id) {
    return false;
  } else {
    const dim_fragment_type *dft = static_cast<const dim_fragment_type *>(&rhs);
    return get_ndim() == rhs.get_ndim() &&
           memcmp(m_tagged_dims.get(), dft->m_tagged_dims.get(), get_ndim() * sizeof(intptr_t)) == 0;
  }
}

void ndt::dim_fragment_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta),
                                                       bool DYND_UNUSED(blockref_alloc)) const {
  throw type_error("Cannot store data of dim_fragment type");
}

void ndt::dim_fragment_type::arrmeta_copy_construct(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    const intrusive_ptr<memory_block_data> &DYND_UNUSED(embedded_reference)) const {
  throw type_error("Cannot store data of dim_fragment type");
}

size_t ndt::dim_fragment_type::arrmeta_copy_construct_onedim(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    const intrusive_ptr<memory_block_data> &DYND_UNUSED(embedded_reference)) const {
  throw type_error("Cannot store data of dim_fragment type");
}

void ndt::dim_fragment_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {
  throw type_error("Cannot store data of dim_fragment type");
}

ndt::type ndt::dim_fragment_type::with_element_type(const type &DYND_UNUSED(element_tp)) const {
  throw runtime_error("with_element_type is not implemented for dim_fragment_type");
}
