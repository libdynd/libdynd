//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/base_tuple_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/shortvector.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/ensure_immutable_contig.hpp>
#include <dynd/buffer_storage.hpp>

using namespace std;
using namespace dynd;

ndt::base_tuple_type::base_tuple_type(type_id_t type_id, const nd::array &field_types, flags_type flags,
                                      bool layout_in_arrmeta, bool variadic)
    : base_type(type_id, variadic ? kind_kind : tuple_kind, 0, 1,
                flags | type_flag_indexable | (variadic ? type_flag_symbolic : 0), 0, 0, 0),
      m_field_count(field_types.get_dim_size()), m_field_types(field_types),
      m_arrmeta_offsets(nd::empty(m_field_count, type::make<uintptr_t>())), m_variadic(variadic)
{
  if (!nd::ensure_immutable_contig<type>(m_field_types)) {
    stringstream ss;
    ss << "dynd tuple type requires an array of types, got an array with "
          "type " << m_field_types.get_type();
    throw invalid_argument(ss.str());
  }

  // Calculate the needed element alignment and arrmeta offsets
  size_t arrmeta_offset = 0;
  if (layout_in_arrmeta) {
    arrmeta_offset = get_field_count() * sizeof(size_t);
  }
  uintptr_t *arrmeta_offsets = reinterpret_cast<uintptr_t *>(m_arrmeta_offsets.get_readwrite_originptr());
  m_members.data_alignment = 1;
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    const type &ft = get_field_type(i);
    size_t field_alignment = ft.get_data_alignment();
    // Accumulate the biggest field alignment as the type alignment
    if (field_alignment > m_members.data_alignment) {
      m_members.data_alignment = (uint8_t)field_alignment;
    }
    // Inherit any operand flags from the fields
    m_members.flags |= (ft.get_flags() & type_flags_operand_inherited);
    // Calculate the arrmeta offsets
    arrmeta_offsets[i] = arrmeta_offset;
    arrmeta_offset += ft.get_arrmeta_size();
  }
  m_members.arrmeta_size = arrmeta_offset;

  m_arrmeta_offsets.flag_as_immutable();
}

ndt::base_tuple_type::~base_tuple_type()
{
}

void ndt::base_tuple_type::print_data(std::ostream &o, const char *arrmeta, const char *data) const
{
  const uintptr_t *arrmeta_offsets = reinterpret_cast<const uintptr_t *>(m_arrmeta_offsets.get_readonly_originptr());
  const size_t *data_offsets = get_data_offsets(arrmeta);
  o << "[";
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    get_field_type(i).print_data(o, arrmeta + arrmeta_offsets[i], data + data_offsets[i]);
    if (i != i_end - 1) {
      o << ", ";
    }
  }
  o << "]";
}

bool ndt::base_tuple_type::is_expression() const
{
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    if (get_field_type(i).is_expression()) {
      return true;
    }
  }
  return false;
}

bool ndt::base_tuple_type::is_unique_data_owner(const char *arrmeta) const
{
  const uintptr_t *arrmeta_offsets = get_arrmeta_offsets_raw();
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    const type &ft = get_field_type(i);
    if (!ft.is_builtin() && !ft.extended()->is_unique_data_owner(arrmeta + arrmeta_offsets[i])) {
      return false;
    }
  }
  return true;
}

size_t ndt::base_tuple_type::get_default_data_size() const
{
  intptr_t field_count = get_field_count();
  // Default layout is to match the field order - could reorder the elements for
  // more efficient packing
  size_t s = 0;
  for (intptr_t i = 0; i != field_count; ++i) {
    const type &ft = get_field_type(i);
    s = inc_to_alignment(s, ft.get_data_alignment());
    if (!ft.is_builtin()) {
      s += ft.extended()->get_default_data_size();
    } else {
      s += ft.get_data_size();
    }
  }
  s = inc_to_alignment(s, m_members.data_alignment);
  return s;
}

void ndt::base_tuple_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta,
                                     const char *DYND_UNUSED(data)) const
{
  out_shape[i] = m_variadic ? -1 : get_field_count();
  if (i < ndim - 1) {
    const uintptr_t *arrmeta_offsets = get_arrmeta_offsets_raw();
    dimvector tmpshape(ndim);
    // Accumulate the shape from all the field shapes
    for (intptr_t fi = 0, fi_end = get_field_count(); fi != fi_end; ++fi) {
      const type &ft = get_field_type(fi);
      if (!ft.is_builtin()) {
        ft.extended()->get_shape(ndim, i + 1, tmpshape.get(), arrmeta ? (arrmeta + arrmeta_offsets[fi]) : NULL, NULL);
      } else {
        stringstream ss;
        ss << "requested too many dimensions from type " << ft;
        throw runtime_error(ss.str());
      }
      if (fi == 0) {
        // Copy the shape from the first field
        memcpy(out_shape + i + 1, tmpshape.get() + i + 1, (ndim - i - 1) * sizeof(intptr_t));
      } else {
        // Merge the shape from the rest
        for (intptr_t k = i + 1; k < ndim; ++k) {
          // If we see different sizes, make the output -1
          if (out_shape[k] != -1 && out_shape[k] != tmpshape[k]) {
            out_shape[k] = -1;
          }
        }
      }
    }
  }
}

void ndt::base_tuple_type::get_vars(std::unordered_set<std::string> &vars) const
{
  for (intptr_t i = 0; i < m_field_count; ++i) {
    get_field_type(i).get_vars(vars);
  }
}

ndt::type ndt::base_tuple_type::apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i,
                                                   const type &root_tp, bool leading_dimension) const
{
  if (nindices == 0) {
    return type(this, true);
  } else {
    bool remove_dimension;
    intptr_t start_index, index_stride, dimension_size;
    apply_single_linear_index(*indices, get_field_count(), current_i, &root_tp, remove_dimension, start_index,
                              index_stride, dimension_size);
    if (remove_dimension) {
      return get_field_type(start_index)
          .apply_linear_index(nindices - 1, indices + 1, current_i + 1, root_tp, leading_dimension);
    } else if (nindices == 1 && start_index == 0 && index_stride == 1 && dimension_size == get_field_count()) {
      // This is a do-nothing index, keep the same type
      return type(this, true);
    } else {
      // Take the subset of the fields in-place
      nd::array tmp_field_types(nd::empty(dimension_size, make_type()));
      type *tmp_field_types_raw = reinterpret_cast<type *>(tmp_field_types.get_readwrite_originptr());

      for (intptr_t i = 0; i < dimension_size; ++i) {
        intptr_t idx = start_index + i * index_stride;
        tmp_field_types_raw[i] =
            get_field_type(idx).apply_linear_index(nindices - 1, indices + 1, current_i + 1, root_tp, false);
      }

      tmp_field_types.flag_as_immutable();
      return tuple_type::make(tmp_field_types);
    }
  }
}

intptr_t ndt::base_tuple_type::apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta,
                                                  const type &result_tp, char *out_arrmeta,
                                                  const intrusive_ptr<memory_block_data> &embedded_reference,
                                                  size_t current_i, const type &root_tp, bool leading_dimension,
                                                  char **inout_data,
                                                  intrusive_ptr<memory_block_data> &inout_dataref) const
{
  if (nindices == 0) {
    // If there are no more indices, copy the arrmeta verbatim
    arrmeta_copy_construct(out_arrmeta, arrmeta, embedded_reference);
    return 0;
  } else {
    const uintptr_t *offsets = get_data_offsets(arrmeta);
    const uintptr_t *arrmeta_offsets = get_arrmeta_offsets_raw();
    bool remove_dimension;
    intptr_t start_index, index_stride, dimension_size;
    apply_single_linear_index(*indices, get_field_count(), current_i, &root_tp, remove_dimension, start_index,
                              index_stride, dimension_size);
    if (remove_dimension) {
      intrusive_ptr<memory_block_data> tmp;
      const type &ft = get_field_type(start_index);
      intptr_t offset = offsets[start_index];
      if (!ft.is_builtin()) {
        if (leading_dimension) {
          // In the case of a leading dimension, first bake the offset into
          // the data pointer, so that it's pointing at the right element
          // for the collapsing of leading dimensions to work correctly.
          *inout_data += offset;
          offset = ft.extended()->apply_linear_index(nindices - 1, indices + 1, arrmeta + arrmeta_offsets[start_index],
                                                     result_tp, out_arrmeta, embedded_reference, current_i + 1, root_tp,
                                                     true, inout_data, inout_dataref);
        } else {
          offset += ft.extended()->apply_linear_index(nindices - 1, indices + 1, arrmeta + arrmeta_offsets[start_index],
                                                      result_tp, out_arrmeta, embedded_reference, current_i + 1,
                                                      root_tp, false, NULL, tmp);
        }
      }
      return offset;
    } else {
      intrusive_ptr<memory_block_data> tmp;
      intptr_t *out_offsets = reinterpret_cast<intptr_t *>(out_arrmeta);
      const tuple_type *result_e_dt = result_tp.extended<tuple_type>();
      for (intptr_t i = 0; i < dimension_size; ++i) {
        intptr_t idx = start_index + i * index_stride;
        out_offsets[i] = offsets[idx];
        const type &ft = result_e_dt->get_field_type(i);
        if (!ft.is_builtin()) {
          out_offsets[i] +=
              ft.extended()->apply_linear_index(nindices - 1, indices + 1, arrmeta + arrmeta_offsets[idx], ft,
                                                out_arrmeta + result_e_dt->get_arrmeta_offset(i), embedded_reference,
                                                current_i + 1, root_tp, false, NULL, tmp);
        }
      }
      return 0;
    }
  }
}

void ndt::base_tuple_type::arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const
{
  const uintptr_t *arrmeta_offsets = get_arrmeta_offsets_raw();
  uintptr_t *data_offsets = get_arrmeta_data_offsets(arrmeta);
  const type *field_tps = get_field_types_raw();
  // If the arrmeta has data offsets, fill them in
  if (data_offsets != NULL) {
    fill_default_data_offsets(get_field_count(), field_tps, data_offsets);
  }

  // Default construct the arrmeta for all the fields
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    const type &tp = field_tps[i];
    if (!tp.is_builtin()) {
      try
      {
        tp.extended()->arrmeta_default_construct(arrmeta + arrmeta_offsets[i], blockref_alloc);
      }
      catch (...)
      {
        // Since we're explicitly controlling the memory, need to manually do
        // the cleanup too
        for (intptr_t j = 0; j < i; ++j) {
          const type &ft = get_field_type(j);
          if (!ft.is_builtin()) {
            ft.extended()->arrmeta_destruct(arrmeta + arrmeta_offsets[i]);
          }
        }
        throw;
      }
    }
  }
}

void ndt::base_tuple_type::arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                                  const intrusive_ptr<memory_block_data> &embedded_reference) const
{
  uintptr_t *dst_data_offsets = get_arrmeta_data_offsets(dst_arrmeta);
  if (dst_data_offsets != 0) {
    // Copy all the field offsets
    memcpy(dst_data_offsets, get_data_offsets(src_arrmeta), get_field_count() * sizeof(uintptr_t));
  }
  // Copy construct all the field's arrmeta
  const uintptr_t *arrmeta_offsets = get_arrmeta_offsets_raw();
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    const type &field_dt = get_field_type(i);
    if (!field_dt.is_builtin()) {
      field_dt.extended()->arrmeta_copy_construct(dst_arrmeta + arrmeta_offsets[i], src_arrmeta + arrmeta_offsets[i],
                                                  embedded_reference);
    }
  }
}

void ndt::base_tuple_type::arrmeta_reset_buffers(char *arrmeta) const
{
  const uintptr_t *arrmeta_offsets = get_arrmeta_offsets_raw();
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    const type &field_dt = get_field_type(i);
    if (field_dt.get_arrmeta_size() > 0) {
      field_dt.extended()->arrmeta_reset_buffers(arrmeta + arrmeta_offsets[i]);
    }
  }
}

void ndt::base_tuple_type::arrmeta_finalize_buffers(char *arrmeta) const
{
  const uintptr_t *arrmeta_offsets = get_arrmeta_offsets_raw();
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    const type &field_dt = get_field_type(i);
    if (!field_dt.is_builtin()) {
      field_dt.extended()->arrmeta_finalize_buffers(arrmeta + arrmeta_offsets[i]);
    }
  }
}

void ndt::base_tuple_type::arrmeta_destruct(char *arrmeta) const
{
  const uintptr_t *arrmeta_offsets = get_arrmeta_offsets_raw();
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    const type &field_dt = get_field_type(i);
    if (!field_dt.is_builtin()) {
      field_dt.extended()->arrmeta_destruct(arrmeta + arrmeta_offsets[i]);
    }
  }
}

void ndt::base_tuple_type::data_destruct(const char *arrmeta, char *data) const
{
  const uintptr_t *arrmeta_offsets = get_arrmeta_offsets_raw();
  const size_t *data_offsets = get_data_offsets(arrmeta);
  intptr_t field_count = get_field_count();
  for (intptr_t i = 0; i != field_count; ++i) {
    const type &ft = get_field_type(i);
    if (ft.get_flags() & type_flag_destructor) {
      ft.extended()->data_destruct(arrmeta + arrmeta_offsets[i], data + data_offsets[i]);
    }
  }
}

void ndt::base_tuple_type::data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const
{
  const uintptr_t *arrmeta_offsets = get_arrmeta_offsets_raw();
  const size_t *data_offsets = get_data_offsets(arrmeta);
  intptr_t field_count = get_field_count();
  // Destruct all the fields a chunk at a time, in an
  // attempt to have some kind of locality
  while (count > 0) {
    size_t chunk_size = min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
    for (intptr_t i = 0; i != field_count; ++i) {
      const type &ft = get_field_type(i);
      if (ft.get_flags() & type_flag_destructor) {
        ft.extended()->data_destruct_strided(arrmeta + arrmeta_offsets[i], data + data_offsets[i], stride, chunk_size);
      }
    }
    data += stride * chunk_size;
    count -= chunk_size;
  }
}

void ndt::base_tuple_type::foreach_leading(const char *arrmeta, char *data, foreach_fn_t callback,
                                           void *callback_data) const
{
  if (get_field_count() != 0) {
    const size_t *data_offsets = get_data_offsets(arrmeta);
    const size_t *arrmeta_offsets = get_arrmeta_offsets_raw();
    for (intptr_t i = 0, i_end = get_field_count(); i < i_end; ++i) {
      callback(get_field_type(i), arrmeta + arrmeta_offsets[i], data + data_offsets[i], callback_data);
    }
  }
}

bool ndt::base_tuple_type::match(const char *arrmeta, const type &candidate_tp, const char *candidate_arrmeta,
                                 std::map<std::string, type> &tp_vars) const
{
  intptr_t candidate_field_count = candidate_tp.extended<base_tuple_type>()->get_field_count();
  bool candidate_variadic = candidate_tp.extended<base_tuple_type>()->is_variadic();

  if ((m_field_count == candidate_field_count && !candidate_variadic) ||
      ((candidate_field_count >= m_field_count) && m_variadic)) {
    auto arrmeta_offsets = get_arrmeta_offsets_raw();
    // Match against the types
    const type *fields = get_field_types_raw();
    const type *candidate_fields = candidate_tp.extended<base_tuple_type>()->get_field_types_raw();
    for (intptr_t i = 0; i != m_field_count; ++i) {
      if (!fields[i].match(DYND_INC_IF_NOT_NULL(arrmeta, arrmeta_offsets[i]), candidate_fields[i], candidate_arrmeta,
                           tp_vars)) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}
