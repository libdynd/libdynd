//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/buffer.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/memblock/objectarray_memory_block.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/memblock/zeroinit_memory_block.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

ndt::var_dim_type::var_dim_type(const type &element_tp)
    : base_dim_type(var_dim_id, element_tp, sizeof(data_type), alignof(data_type), sizeof(metadata_type),
                    type_flag_zeroinit | type_flag_blockref, false) {
  // NOTE: The element type may have type_flag_destructor set. In this case,
  //       the var_dim type does NOT need to also set it, because the lifetime
  //       of the elements it allocates is owned by the
  //       objectarray_memory_block,
  //       not by the var_dim elements.
  // Propagate just the value-inherited flags from the element
  this->flags |= (element_tp.get_flags() & type_flags_value_inherited);
}

void ndt::var_dim_type::print_data(std::ostream &o, const char *arrmeta, const char *data) const {
  const metadata_type *md = reinterpret_cast<const metadata_type *>(arrmeta);
  const data_type *d = reinterpret_cast<const data_type *>(data);
  const char *element_data = d->begin + md->offset;
  strided_array_summarized(o, get_element_type(), arrmeta + sizeof(metadata_type), element_data, d->size, md->stride);
}

void ndt::var_dim_type::print_type(std::ostream &o) const { o << "var * " << m_element_tp; }

bool ndt::var_dim_type::is_expression() const { return m_element_tp.is_expression(); }

bool ndt::var_dim_type::is_unique_data_owner(const char *arrmeta) const {
  const metadata_type *md = reinterpret_cast<const metadata_type *>(arrmeta);
  if (md->blockref && md->blockref->get_use_count() != 1) {
    return false;
  }
  if (m_element_tp.is_builtin()) {
    return true;
  } else {
    return m_element_tp.extended()->is_unique_data_owner(arrmeta + sizeof(metadata_type));
  }
}

void ndt::var_dim_type::transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                                              type &out_transformed_tp, bool &out_was_transformed) const {
  type tmp_tp;
  bool was_transformed = false;
  transform_fn(m_element_tp, arrmeta_offset + sizeof(metadata_type), extra, tmp_tp, was_transformed);
  if (was_transformed) {
    out_transformed_tp = type(new var_dim_type(tmp_tp), false);
    out_was_transformed = true;
  } else {
    out_transformed_tp = type(this, true);
  }
}

ndt::type ndt::var_dim_type::get_canonical_type() const {
  return type(new var_dim_type(m_element_tp.get_canonical_type()), false);
}

ndt::type ndt::var_dim_type::apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i,
                                                const type &root_tp, bool leading_dimension) const {
  if (nindices == 0) {
    return type(this, true);
  } else if (nindices == 1) {
    if (indices->step() == 0) {
      if (leading_dimension) {
        if (m_element_tp.is_builtin()) {
          return m_element_tp;
        } else {
          return m_element_tp.apply_linear_index(0, NULL, current_i, root_tp, true);
        }
      } else {
        // TODO: This is incorrect, but is here as a stopgap to be replaced by a
        // sliced<> type
        return make_type<pointer_type>(m_element_tp);
      }
    } else {
      if (indices->is_nop()) {
        // If the indexing operation does nothing, then leave things unchanged
        return type(this, true);
      } else {
        // TODO: sliced_var_dim_type
        throw runtime_error("TODO: implement "
                            "var_dim_type::apply_linear_index for general "
                            "slices");
      }
    }
  } else {
    if (indices->step() == 0) {
      if (leading_dimension) {
        return m_element_tp.apply_linear_index(nindices - 1, indices + 1, current_i + 1, root_tp, true);
      } else {
        // TODO: This is incorrect, but is here as a stopgap to be replaced by a
        // sliced<> type
        return make_type<pointer_type>(
            m_element_tp.apply_linear_index(nindices - 1, indices + 1, current_i + 1, root_tp, false));
      }
    } else {
      if (indices->is_nop()) {
        // If the indexing operation does nothing, then leave things unchanged
        type edt = m_element_tp.apply_linear_index(nindices - 1, indices + 1, current_i + 1, root_tp, false);
        return type(new var_dim_type(edt), false);
      } else {
        // TODO: sliced_var_dim_type
        throw runtime_error("TODO: implement "
                            "var_dim_type::apply_linear_index for general "
                            "slices");
        // return ndt::type(new
        // var_dim_type(m_element_tp.apply_linear_index(nindices-1, indices+1,
        // current_i+1, root_tp)), false);
      }
    }
  }
}

intptr_t ndt::var_dim_type::apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta,
                                               const type &result_tp, char *out_arrmeta,
                                               const nd::memory_block &embedded_reference, size_t current_i,
                                               const type &root_tp, bool leading_dimension, char **inout_data,
                                               nd::memory_block &inout_dataref) const {
  if (nindices == 0) {
    // If there are no more indices, copy the arrmeta verbatim
    arrmeta_copy_construct(out_arrmeta, arrmeta, embedded_reference);
    return 0;
  } else {
    const metadata_type *md = reinterpret_cast<const metadata_type *>(arrmeta);
    if (leading_dimension) {
      const data_type *d = reinterpret_cast<const data_type *>(*inout_data);
      bool remove_dimension;
      intptr_t start_index, index_stride, dimension_size;
      apply_single_linear_index(*indices, d->size, current_i, &root_tp, remove_dimension, start_index, index_stride,
                                dimension_size);
      if (remove_dimension) {
        // First dereference to point at the actual element
        *inout_data = d->begin + md->offset + start_index * md->stride;
        inout_dataref = md->blockref ? md->blockref : embedded_reference;
        // Then apply a 0-sized index to the element type
        if (!m_element_tp.is_builtin()) {
          return m_element_tp.extended()->apply_linear_index(nindices - 1, indices + 1, arrmeta + sizeof(metadata_type),
                                                             result_tp, out_arrmeta, embedded_reference, current_i,
                                                             root_tp, true, inout_data, inout_dataref);
        } else {
          return 0;
        }
      } else if (indices->is_nop()) {
        nd::memory_block tmp;
        // If the indexing operation does nothing, then leave things unchanged
        metadata_type *out_md = reinterpret_cast<metadata_type *>(out_arrmeta);
        out_md->blockref = md->blockref ? md->blockref : embedded_reference;
        out_md->stride = md->stride;
        out_md->offset = md->offset;
        if (!m_element_tp.is_builtin()) {
          const var_dim_type *vad = result_tp.extended<var_dim_type>();
          out_md->offset += m_element_tp.extended()->apply_linear_index(
              nindices - 1, indices + 1, arrmeta + sizeof(metadata_type), vad->get_element_type(),
              out_arrmeta + sizeof(metadata_type), embedded_reference, current_i, root_tp, false, NULL, tmp);
        }
        return 0;
      } else {
        // TODO: sliced_var_dim_type
        throw runtime_error("TODO: implement var_dim_type::apply_linear_index "
                            "for general slices");
        // return ndt::type(this, true);
      }
    } else {
      if (indices->step() == 0) {
        nd::memory_block tmp;
        // TODO: This is incorrect, but is here as a stopgap to be replaced by a
        // sliced<> type
        pointer_type_arrmeta *out_md = reinterpret_cast<pointer_type_arrmeta *>(out_arrmeta);
        out_md->blockref = md->blockref ? md->blockref : embedded_reference;
        out_md->offset = indices->start() * md->stride;
        if (!m_element_tp.is_builtin()) {
          const pointer_type *result_etp = result_tp.extended<pointer_type>();
          out_md->offset += m_element_tp.extended()->apply_linear_index(
              nindices - 1, indices + 1, arrmeta + sizeof(metadata_type), result_etp->get_target_type(),
              out_arrmeta + sizeof(pointer_type_arrmeta), embedded_reference, current_i + 1, root_tp, false, NULL, tmp);
        }
        return 0;
      } else if (indices->is_nop()) {
        nd::memory_block tmp;
        // If the indexing operation does nothing, then leave things unchanged
        metadata_type *out_md = reinterpret_cast<metadata_type *>(out_arrmeta);
        out_md->blockref = md->blockref ? md->blockref : embedded_reference;
        out_md->stride = md->stride;
        out_md->offset = md->offset;
        if (!m_element_tp.is_builtin()) {
          const var_dim_type *vad = result_tp.extended<var_dim_type>();
          out_md->offset += m_element_tp.extended()->apply_linear_index(
              nindices - 1, indices + 1, arrmeta + sizeof(metadata_type), vad->get_element_type(),
              out_arrmeta + sizeof(metadata_type), embedded_reference, current_i, root_tp, false, NULL, tmp);
        }
        return 0;
      } else {
        // TODO: sliced_var_dim_type
        throw runtime_error("TODO: implement var_dim_type::apply_linear_index "
                            "for general slices");
        // return ndt::type(this, true);
      }
    }
  }
}

ndt::type ndt::var_dim_type::at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const {
  if (inout_arrmeta) {
    const metadata_type *md = reinterpret_cast<const metadata_type *>(*inout_arrmeta);
    // Modify the arrmeta
    *inout_arrmeta += sizeof(metadata_type);
    // If requested, modify the data pointer
    if (inout_data) {
      const data_type *d = reinterpret_cast<const data_type *>(*inout_data);
      // Bounds-checking of the index
      i0 = apply_single_index(i0, d->size, NULL);
      *inout_data = d->begin + md->offset + i0 * md->stride;
    }
  }
  return m_element_tp;
}

ndt::type ndt::var_dim_type::get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim) const {
  if (i == 0) {
    return type(this, true);
  } else {
    if (inout_arrmeta) {
      *inout_arrmeta += sizeof(metadata_type);
    }
    return m_element_tp.get_type_at_dimension(inout_arrmeta, i - 1, total_ndim + 1);
  }
}

intptr_t ndt::var_dim_type::get_dim_size(const char *DYND_UNUSED(arrmeta), const char *data) const {
  if (data != NULL) {
    return reinterpret_cast<const data_type *>(data)->size;
  } else {
    return -1;
  }
}

void ndt::var_dim_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta,
                                  const char *data) const {
  if (arrmeta == NULL || data == NULL) {
    out_shape[i] = -1;
    data = NULL;
  } else {
    const metadata_type *md = reinterpret_cast<const metadata_type *>(arrmeta);
    const data_type *d = reinterpret_cast<const data_type *>(data);
    out_shape[i] = d->size;
    if (d->size == 1 && d->begin != NULL) {
      data = d->begin + md->offset;
    } else {
      data = NULL;
    }
  }

  // Process the later shape values
  if (i + 1 < ndim) {
    if (!m_element_tp.is_builtin()) {
      m_element_tp.extended()->get_shape(ndim, i + 1, out_shape, arrmeta ? (arrmeta + sizeof(metadata_type)) : NULL,
                                         data);
    } else {
      stringstream ss;
      ss << "requested too many dimensions from type " << type(this, true);
      throw runtime_error(ss.str());
    }
  }
}

void ndt::var_dim_type::get_strides(size_t i, intptr_t *out_strides, const char *arrmeta) const {
  const metadata_type *md = reinterpret_cast<const metadata_type *>(arrmeta);

  out_strides[i] = md->stride;

  // Process the later shape values
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->get_strides(i + 1, out_strides, arrmeta + sizeof(metadata_type));
  }
}

axis_order_classification_t ndt::var_dim_type::classify_axis_order(const char *arrmeta) const {
  // Treat the var_dim type as C-order
  if (m_element_tp.get_ndim() > 1) {
    axis_order_classification_t aoc = m_element_tp.extended()->classify_axis_order(arrmeta + sizeof(metadata_type));
    return (aoc == axis_order_none || aoc == axis_order_c) ? axis_order_c : axis_order_neither;
  } else {
    return axis_order_c;
  }
}

bool ndt::var_dim_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const {
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    } else if (src_tp.get_id() == var_dim_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

bool ndt::var_dim_type::operator==(const base_type &rhs) const {
  if (this == &rhs) {
    return true;
  } else if (rhs.get_id() != var_dim_id) {
    return false;
  } else {
    const var_dim_type *dt = static_cast<const var_dim_type *>(&rhs);
    return m_element_tp == dt->m_element_tp;
  }
}

void ndt::var_dim_type::arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const {
  size_t element_size =
      m_element_tp.is_builtin() ? m_element_tp.get_data_size() : m_element_tp.extended()->get_default_data_size();

  metadata_type *md = reinterpret_cast<metadata_type *>(arrmeta);
  md->stride = element_size;
  md->offset = 0;
  // Allocate a memory block
  if (blockref_alloc) {
    uint32_t flags = m_element_tp.get_flags();
    if (flags & type_flag_destructor) {
      md->blockref = nd::make_memory_block<nd::objectarray_memory_block>(m_element_tp, sizeof(metadata_type), arrmeta,
                                                                         element_size, 64);
    } else if (flags & type_flag_zeroinit) {
      md->blockref = nd::make_zeroinit_memory_block(m_element_tp);
    } else {
      md->blockref = nd::make_memory_block<nd::pod_memory_block>(m_element_tp);
    }
  }
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_default_construct(arrmeta + sizeof(metadata_type), blockref_alloc);
  }
}

void ndt::var_dim_type::arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                               const nd::memory_block &embedded_reference) const {
  const metadata_type *src_md = reinterpret_cast<const metadata_type *>(src_arrmeta);
  metadata_type *dst_md = reinterpret_cast<metadata_type *>(dst_arrmeta);
  dst_md->stride = src_md->stride;
  dst_md->offset = src_md->offset;
  dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_copy_construct(dst_arrmeta + sizeof(metadata_type),
                                                    src_arrmeta + sizeof(metadata_type), embedded_reference);
  }
}

size_t ndt::var_dim_type::arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                                                        const nd::memory_block &embedded_reference) const {
  const metadata_type *src_md = reinterpret_cast<const metadata_type *>(src_arrmeta);
  metadata_type *dst_md = reinterpret_cast<metadata_type *>(dst_arrmeta);
  dst_md->stride = src_md->stride;
  dst_md->offset = src_md->offset;
  dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
  return sizeof(metadata_type);
}

void ndt::var_dim_type::arrmeta_reset_buffers(char *arrmeta) const {
  const metadata_type *md = reinterpret_cast<const metadata_type *>(arrmeta);

  if (m_element_tp.get_arrmeta_size() > 0) {
    m_element_tp.extended()->arrmeta_reset_buffers(arrmeta + sizeof(metadata_type));
  }

  if (md->blockref) {
    md->blockref->reset();
  }
}

void ndt::var_dim_type::arrmeta_finalize_buffers(char *arrmeta) const {
  // Finalize any child arrmeta
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_finalize_buffers(arrmeta + sizeof(metadata_type));
  }

  // Finalize the blockref buffer we own
  metadata_type *md = reinterpret_cast<metadata_type *>(arrmeta);
  if (md->blockref) {
    // Finalize the memory block
    if (m_element_tp.get_flags() & type_flag_destructor) {
      md->blockref->finalize();
    } else {
      md->blockref->finalize();
    }
  }
}

void ndt::var_dim_type::arrmeta_destruct(char *arrmeta) const {
  reinterpret_cast<metadata_type *>(arrmeta)->~metadata_type();
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_destruct(arrmeta + sizeof(metadata_type));
  }
}

void ndt::var_dim_type::arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const {
  const metadata_type *md = reinterpret_cast<const metadata_type *>(arrmeta);
  o << indent << "var_dim arrmeta\n";
  o << indent << " stride: " << md->stride << "\n";
  o << indent << " offset: " << md->offset << "\n";
  md->blockref->debug_print(o, indent + " ");
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_debug_print(arrmeta + sizeof(metadata_type), o, indent + "  ");
  }
}

void ndt::var_dim_type::data_destruct(const char *arrmeta, char *data) const {
  m_element_tp.extended()->data_destruct_strided(
      arrmeta + sizeof(metadata_type), reinterpret_cast<data_type *>(data)->begin,
      reinterpret_cast<const metadata_type *>(arrmeta)->stride, reinterpret_cast<data_type *>(data)->size);
}

void ndt::var_dim_type::data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const {
  const metadata_type *md = reinterpret_cast<const metadata_type *>(arrmeta);
  arrmeta += sizeof(metadata_type);
  intptr_t child_stride = md->stride;
  size_t child_size = reinterpret_cast<data_type *>(data)->size;

  for (size_t i = 0; i != count; ++i, data += stride) {
    m_element_tp.extended()->data_destruct_strided(arrmeta, data, child_stride, child_size);
  }
}

size_t ndt::var_dim_type::get_iterdata_size(intptr_t DYND_UNUSED(ndim)) const {
  throw runtime_error("TODO: implement var_dim_type::get_iterdata_size");
}

size_t ndt::var_dim_type::iterdata_construct(iterdata_common *DYND_UNUSED(iterdata),
                                             const char **DYND_UNUSED(inout_arrmeta), intptr_t DYND_UNUSED(ndim),
                                             const intptr_t *DYND_UNUSED(shape),
                                             type &DYND_UNUSED(out_uniform_tp)) const {
  throw runtime_error("TODO: implement var_dim_type::iterdata_construct");
}

size_t ndt::var_dim_type::iterdata_destruct(iterdata_common *DYND_UNUSED(iterdata), intptr_t DYND_UNUSED(ndim)) const {
  throw runtime_error("TODO: implement var_dim_type::iterdata_destruct");
}

void ndt::var_dim_type::foreach_leading(const char *arrmeta, char *data, foreach_fn_t callback,
                                        void *callback_data) const {
  const metadata_type *md = reinterpret_cast<const metadata_type *>(arrmeta);
  const char *child_arrmeta = arrmeta + sizeof(metadata_type);
  const data_type *d = reinterpret_cast<const data_type *>(data);
  data = d->begin + md->offset;
  intptr_t stride = md->stride;
  for (intptr_t i = 0, i_end = d->size; i < i_end; ++i, data += stride) {
    callback(m_element_tp, child_arrmeta, data, callback_data);
  }
}

std::map<std::string, std::pair<ndt::type, const char *>> ndt::var_dim_type::get_dynamic_type_properties() const {
  std::map<std::string, std::pair<ndt::type, const char *>> properties;
  properties["element_type"] = {ndt::type("type"), reinterpret_cast<const char *>(&m_element_tp)};

  return properties;
}

ndt::type ndt::var_dim_type::with_element_type(const type &element_tp) const {
  return make_type<var_dim_type>(element_tp);
}
