//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/memblock/zeroinit_memory_block.hpp>
#include <dynd/memblock/objectarray_memory_block.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/var_dim_assignment_kernels.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/func/apply.hpp>

using namespace std;
using namespace dynd;

ndt::var_dim_type::var_dim_type(const type &element_tp)
    : base_dim_type(var_dim_type_id, element_tp, sizeof(var_dim_type_data), sizeof(const char *),
                    sizeof(var_dim_type_arrmeta), type_flag_zeroinit | type_flag_blockref, false)
{
  // NOTE: The element type may have type_flag_destructor set. In this case,
  //       the var_dim type does NOT need to also set it, because the lifetime
  //       of the elements it allocates is owned by the
  //       objectarray_memory_block,
  //       not by the var_dim elements.
  // Propagate just the value-inherited flags from the element
  m_members.flags |= (element_tp.get_flags() & type_flags_value_inherited);

  // Copy nd::array properties and functions from the first non-array dimension
  get_scalar_properties_and_functions(m_array_properties, m_array_functions);
}

ndt::var_dim_type::~var_dim_type()
{
}

void ndt::var_dim_type::print_data(std::ostream &o, const char *arrmeta, const char *data) const
{
  const var_dim_type_arrmeta *md = reinterpret_cast<const var_dim_type_arrmeta *>(arrmeta);
  const var_dim_type_data *d = reinterpret_cast<const var_dim_type_data *>(data);
  const char *element_data = d->begin + md->offset;
  strided_array_summarized(o, get_element_type(), arrmeta + sizeof(var_dim_type_arrmeta), element_data, d->size,
                           md->stride);
}

void ndt::var_dim_type::print_type(std::ostream &o) const
{
  o << "var * " << m_element_tp;
}

bool ndt::var_dim_type::is_expression() const
{
  return m_element_tp.is_expression();
}

bool ndt::var_dim_type::is_unique_data_owner(const char *arrmeta) const
{
  const var_dim_type_arrmeta *md = reinterpret_cast<const var_dim_type_arrmeta *>(arrmeta);
  if (md->blockref != NULL &&
      (md->blockref->m_use_count != 1 ||
       (md->blockref->m_type != pod_memory_block_type && md->blockref->m_type != zeroinit_memory_block_type &&
        md->blockref->m_type != objectarray_memory_block_type))) {
    return false;
  }
  if (m_element_tp.is_builtin()) {
    return true;
  } else {
    return m_element_tp.extended()->is_unique_data_owner(arrmeta + sizeof(var_dim_type_arrmeta));
  }
}

void ndt::var_dim_type::transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                                              type &out_transformed_tp, bool &out_was_transformed) const
{
  type tmp_tp;
  bool was_transformed = false;
  transform_fn(m_element_tp, arrmeta_offset + sizeof(var_dim_type_arrmeta), extra, tmp_tp, was_transformed);
  if (was_transformed) {
    out_transformed_tp = type(new var_dim_type(tmp_tp), false);
    out_was_transformed = true;
  } else {
    out_transformed_tp = type(this, true);
  }
}

ndt::type ndt::var_dim_type::get_canonical_type() const
{
  return type(new var_dim_type(m_element_tp.get_canonical_type()), false);
}

ndt::type ndt::var_dim_type::apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i,
                                                const type &root_tp, bool leading_dimension) const
{
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
        return pointer_type::make(m_element_tp);
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
        return pointer_type::make(
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
                                               memory_block_data *embedded_reference, size_t current_i,
                                               const type &root_tp, bool leading_dimension, char **inout_data,
                                               memory_block_data **inout_dataref) const
{
  if (nindices == 0) {
    // If there are no more indices, copy the arrmeta verbatim
    arrmeta_copy_construct(out_arrmeta, arrmeta, embedded_reference);
    return 0;
  } else {
    const var_dim_type_arrmeta *md = reinterpret_cast<const var_dim_type_arrmeta *>(arrmeta);
    if (leading_dimension) {
      const var_dim_type_data *d = reinterpret_cast<const var_dim_type_data *>(*inout_data);
      bool remove_dimension;
      intptr_t start_index, index_stride, dimension_size;
      apply_single_linear_index(*indices, d->size, current_i, &root_tp, remove_dimension, start_index, index_stride,
                                dimension_size);
      if (remove_dimension) {
        // First dereference to point at the actual element
        *inout_data = d->begin + md->offset + start_index * md->stride;
        if (*inout_dataref) {
          memory_block_decref(*inout_dataref);
        }
        *inout_dataref = md->blockref ? md->blockref : embedded_reference;
        memory_block_incref(*inout_dataref);
        // Then apply a 0-sized index to the element type
        if (!m_element_tp.is_builtin()) {
          return m_element_tp.extended()->apply_linear_index(
              nindices - 1, indices + 1, arrmeta + sizeof(var_dim_type_arrmeta), result_tp, out_arrmeta,
              embedded_reference, current_i, root_tp, true, inout_data, inout_dataref);
        } else {
          return 0;
        }
      } else if (indices->is_nop()) {
        // If the indexing operation does nothing, then leave things unchanged
        var_dim_type_arrmeta *out_md = reinterpret_cast<var_dim_type_arrmeta *>(out_arrmeta);
        out_md->blockref = md->blockref ? md->blockref : embedded_reference;
        memory_block_incref(out_md->blockref);
        out_md->stride = md->stride;
        out_md->offset = md->offset;
        if (!m_element_tp.is_builtin()) {
          const var_dim_type *vad = result_tp.extended<var_dim_type>();
          out_md->offset += m_element_tp.extended()->apply_linear_index(
              nindices - 1, indices + 1, arrmeta + sizeof(var_dim_type_arrmeta), vad->get_element_type(),
              out_arrmeta + sizeof(var_dim_type_arrmeta), embedded_reference, current_i, root_tp, false, NULL, NULL);
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
        // TODO: This is incorrect, but is here as a stopgap to be replaced by a
        // sliced<> type
        pointer_type_arrmeta *out_md = reinterpret_cast<pointer_type_arrmeta *>(out_arrmeta);
        out_md->blockref = md->blockref ? md->blockref : embedded_reference;
        memory_block_incref(out_md->blockref);
        out_md->offset = indices->start() * md->stride;
        if (!m_element_tp.is_builtin()) {
          const pointer_type *result_etp = result_tp.extended<pointer_type>();
          out_md->offset += m_element_tp.extended()->apply_linear_index(
              nindices - 1, indices + 1, arrmeta + sizeof(var_dim_type_arrmeta), result_etp->get_target_type(),
              out_arrmeta + sizeof(pointer_type_arrmeta), embedded_reference, current_i + 1, root_tp, false, NULL,
              NULL);
        }
        return 0;
      } else if (indices->is_nop()) {
        // If the indexing operation does nothing, then leave things unchanged
        var_dim_type_arrmeta *out_md = reinterpret_cast<var_dim_type_arrmeta *>(out_arrmeta);
        out_md->blockref = md->blockref ? md->blockref : embedded_reference;
        memory_block_incref(out_md->blockref);
        out_md->stride = md->stride;
        out_md->offset = md->offset;
        if (!m_element_tp.is_builtin()) {
          const var_dim_type *vad = result_tp.extended<var_dim_type>();
          out_md->offset += m_element_tp.extended()->apply_linear_index(
              nindices - 1, indices + 1, arrmeta + sizeof(var_dim_type_arrmeta), vad->get_element_type(),
              out_arrmeta + sizeof(var_dim_type_arrmeta), embedded_reference, current_i, root_tp, false, NULL, NULL);
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

ndt::type ndt::var_dim_type::at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const
{
  if (inout_arrmeta) {
    const var_dim_type_arrmeta *md = reinterpret_cast<const var_dim_type_arrmeta *>(*inout_arrmeta);
    // Modify the arrmeta
    *inout_arrmeta += sizeof(var_dim_type_arrmeta);
    // If requested, modify the data pointer
    if (inout_data) {
      const var_dim_type_data *d = reinterpret_cast<const var_dim_type_data *>(*inout_data);
      // Bounds-checking of the index
      i0 = apply_single_index(i0, d->size, NULL);
      *inout_data = d->begin + md->offset + i0 * md->stride;
    }
  }
  return m_element_tp;
}

ndt::type ndt::var_dim_type::get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim) const
{
  if (i == 0) {
    return type(this, true);
  } else {
    if (inout_arrmeta) {
      *inout_arrmeta += sizeof(var_dim_type_arrmeta);
    }
    return m_element_tp.get_type_at_dimension(inout_arrmeta, i - 1, total_ndim + 1);
  }
}

intptr_t ndt::var_dim_type::get_dim_size(const char *DYND_UNUSED(arrmeta), const char *data) const
{
  if (data != NULL) {
    return reinterpret_cast<const var_dim_type_data *>(data)->size;
  } else {
    return -1;
  }
}

void ndt::var_dim_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta,
                                  const char *data) const
{
  if (arrmeta == NULL || data == NULL) {
    out_shape[i] = -1;
    data = NULL;
  } else {
    const var_dim_type_arrmeta *md = reinterpret_cast<const var_dim_type_arrmeta *>(arrmeta);
    const var_dim_type_data *d = reinterpret_cast<const var_dim_type_data *>(data);
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
      m_element_tp.extended()->get_shape(ndim, i + 1, out_shape,
                                         arrmeta ? (arrmeta + sizeof(var_dim_type_arrmeta)) : NULL, data);
    } else {
      stringstream ss;
      ss << "requested too many dimensions from type " << type(this, true);
      throw runtime_error(ss.str());
    }
  }
}

void ndt::var_dim_type::get_strides(size_t i, intptr_t *out_strides, const char *arrmeta) const
{
  const var_dim_type_arrmeta *md = reinterpret_cast<const var_dim_type_arrmeta *>(arrmeta);

  out_strides[i] = md->stride;

  // Process the later shape values
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->get_strides(i + 1, out_strides, arrmeta + sizeof(var_dim_type_arrmeta));
  }
}

axis_order_classification_t ndt::var_dim_type::classify_axis_order(const char *arrmeta) const
{
  // Treat the var_dim type as C-order
  if (m_element_tp.get_ndim() > 1) {
    axis_order_classification_t aoc =
        m_element_tp.extended()->classify_axis_order(arrmeta + sizeof(var_dim_type_arrmeta));
    return (aoc == axis_order_none || aoc == axis_order_c) ? axis_order_c : axis_order_neither;
  } else {
    return axis_order_c;
  }
}

bool ndt::var_dim_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    } else if (src_tp.get_type_id() == var_dim_type_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

bool ndt::var_dim_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != var_dim_type_id) {
    return false;
  } else {
    const var_dim_type *dt = static_cast<const var_dim_type *>(&rhs);
    return m_element_tp == dt->m_element_tp;
  }
}

void ndt::var_dim_type::arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const
{
  size_t element_size =
      m_element_tp.is_builtin() ? m_element_tp.get_data_size() : m_element_tp.extended()->get_default_data_size();

  var_dim_type_arrmeta *md = reinterpret_cast<var_dim_type_arrmeta *>(arrmeta);
  md->stride = element_size;
  md->offset = 0;
  // Allocate a memory block
  if (blockref_alloc) {
    base_type::flags_type flags = m_element_tp.get_flags();
    if (flags & type_flag_destructor) {
      md->blockref = make_objectarray_memory_block(m_element_tp, arrmeta, element_size).release();
    } else if (flags & type_flag_zeroinit) {
      md->blockref = make_zeroinit_memory_block().release();
    } else {
      md->blockref = make_pod_memory_block().release();
    }
  }
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_default_construct(arrmeta + sizeof(var_dim_type_arrmeta), blockref_alloc);
  }
}

void ndt::var_dim_type::arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                               memory_block_data *embedded_reference) const
{
  const var_dim_type_arrmeta *src_md = reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta);
  var_dim_type_arrmeta *dst_md = reinterpret_cast<var_dim_type_arrmeta *>(dst_arrmeta);
  dst_md->stride = src_md->stride;
  dst_md->offset = src_md->offset;
  dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
  if (dst_md->blockref) {
    memory_block_incref(dst_md->blockref);
  }
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_copy_construct(dst_arrmeta + sizeof(var_dim_type_arrmeta),
                                                    src_arrmeta + sizeof(var_dim_type_arrmeta), embedded_reference);
  }
}

size_t ndt::var_dim_type::arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                                                        memory_block_data *embedded_reference) const
{
  const var_dim_type_arrmeta *src_md = reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta);
  var_dim_type_arrmeta *dst_md = reinterpret_cast<var_dim_type_arrmeta *>(dst_arrmeta);
  dst_md->stride = src_md->stride;
  dst_md->offset = src_md->offset;
  dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
  if (dst_md->blockref) {
    memory_block_incref(dst_md->blockref);
  }
  return sizeof(var_dim_type_arrmeta);
}

void ndt::var_dim_type::arrmeta_reset_buffers(char *arrmeta) const
{
  const var_dim_type_arrmeta *md = reinterpret_cast<const var_dim_type_arrmeta *>(arrmeta);

  if (m_element_tp.get_arrmeta_size() > 0) {
    m_element_tp.extended()->arrmeta_reset_buffers(arrmeta + sizeof(var_dim_type_arrmeta));
  }

  if (md->blockref != NULL) {
    uint32_t br_type = md->blockref->m_type;
    if (br_type == zeroinit_memory_block_type || br_type == pod_memory_block_type) {
      memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
      allocator->reset(md->blockref);
      return;
    } else if (br_type == objectarray_memory_block_type) {
      memory_block_objectarray_allocator_api *allocator = get_memory_block_objectarray_allocator_api(md->blockref);
      allocator->reset(md->blockref);
      return;
    }
  }

  stringstream ss;
  ss << "can only reset the buffers of a var_dim type ";
  ss << "if it was default-constructed. Its blockref is ";
  if (md->blockref == NULL) {
    ss << "NULL";
  } else {
    ss << "of the wrong type " << (memory_block_type_t)md->blockref->m_type;
  }
  throw runtime_error(ss.str());
}

void ndt::var_dim_type::arrmeta_finalize_buffers(char *arrmeta) const
{
  // Finalize any child arrmeta
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_finalize_buffers(arrmeta + sizeof(var_dim_type_arrmeta));
  }

  // Finalize the blockref buffer we own
  var_dim_type_arrmeta *md = reinterpret_cast<var_dim_type_arrmeta *>(arrmeta);
  if (md->blockref != NULL) {
    // Finalize the memory block
    if (m_element_tp.get_flags() & type_flag_destructor) {
      memory_block_objectarray_allocator_api *allocator = get_memory_block_objectarray_allocator_api(md->blockref);
      if (allocator != NULL) {
        allocator->finalize(md->blockref);
      }
    } else {
      memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
      if (allocator != NULL) {
        allocator->finalize(md->blockref);
      }
    }
  }
}

void ndt::var_dim_type::arrmeta_destruct(char *arrmeta) const
{
  var_dim_type_arrmeta *md = reinterpret_cast<var_dim_type_arrmeta *>(arrmeta);
  if (md->blockref) {
    memory_block_decref(md->blockref);
  }
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_destruct(arrmeta + sizeof(var_dim_type_arrmeta));
  }
}

void ndt::var_dim_type::arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const
{
  const var_dim_type_arrmeta *md = reinterpret_cast<const var_dim_type_arrmeta *>(arrmeta);
  o << indent << "var_dim arrmeta\n";
  o << indent << " stride: " << md->stride << "\n";
  o << indent << " offset: " << md->offset << "\n";
  memory_block_debug_print(md->blockref, o, indent + " ");
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_debug_print(arrmeta + sizeof(var_dim_type_arrmeta), o, indent + "  ");
  }
}

size_t ndt::var_dim_type::get_iterdata_size(intptr_t DYND_UNUSED(ndim)) const
{
  throw runtime_error("TODO: implement var_dim_type::get_iterdata_size");
}

size_t ndt::var_dim_type::iterdata_construct(iterdata_common *DYND_UNUSED(iterdata),
                                             const char **DYND_UNUSED(inout_arrmeta), intptr_t DYND_UNUSED(ndim),
                                             const intptr_t *DYND_UNUSED(shape),
                                             type &DYND_UNUSED(out_uniform_tp)) const
{
  throw runtime_error("TODO: implement var_dim_type::iterdata_construct");
}

size_t ndt::var_dim_type::iterdata_destruct(iterdata_common *DYND_UNUSED(iterdata), intptr_t DYND_UNUSED(ndim)) const
{
  throw runtime_error("TODO: implement var_dim_type::iterdata_destruct");
}

intptr_t ndt::var_dim_type::make_assignment_kernel(void *ckb, intptr_t ckb_offset, const type &dst_tp,
                                                   const char *dst_arrmeta, const type &src_tp, const char *src_arrmeta,
                                                   kernel_request_t kernreq, const eval::eval_context *ectx) const
{
  if (this == dst_tp.extended()) {
    intptr_t src_size, src_stride;
    type src_el_tp;
    const char *src_el_arrmeta;

    if (src_tp.get_ndim() < dst_tp.get_ndim()) {
      // If the src has fewer dimensions, broadcast it across this one
      return make_broadcast_to_var_dim_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
                                                         kernreq, ectx);
    } else if (src_tp.get_type_id() == var_dim_type_id) {
      // var_dim to var_dim
      return make_var_dim_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta, kernreq, ectx);
    } else if (src_tp.get_as_strided(src_arrmeta, &src_size, &src_stride, &src_el_tp, &src_el_arrmeta)) {
      // strided_dim to var_dim
      return make_strided_to_var_dim_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_size, src_stride,
                                                       src_el_tp, src_el_arrmeta, kernreq, ectx);
    } else if (!src_tp.is_builtin()) {
      // Give the src type a chance to make a kernel
      return src_tp.extended()->make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
                                                       kernreq, ectx);
    } else {
      stringstream ss;
      ss << "Cannot assign from " << src_tp << " to " << dst_tp;
      throw dynd::type_error(ss.str());
    }
  } else if (dst_tp.get_kind() == string_kind) {
    return make_any_to_string_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta, kernreq,
                                                ectx);
  } else if (dst_tp.get_ndim() < src_tp.get_ndim()) {
    throw broadcast_error(dst_tp, dst_arrmeta, src_tp, src_arrmeta);
  } else {
    if (dst_tp.get_type_id() == fixed_dim_type_id) {
      // var_dim to fixed_dim
      return make_var_to_fixed_dim_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta, kernreq,
                                                     ectx);
    } else {
      stringstream ss;
      ss << "Cannot assign from " << src_tp << " to " << dst_tp;
      throw dynd::type_error(ss.str());
    }
  }
}

void ndt::var_dim_type::foreach_leading(const char *arrmeta, char *data, foreach_fn_t callback,
                                        void *callback_data) const
{
  const var_dim_type_arrmeta *md = reinterpret_cast<const var_dim_type_arrmeta *>(arrmeta);
  const char *child_arrmeta = arrmeta + sizeof(var_dim_type_arrmeta);
  const var_dim_type_data *d = reinterpret_cast<const var_dim_type_data *>(data);
  data = d->begin + md->offset;
  intptr_t stride = md->stride;
  for (intptr_t i = 0, i_end = d->size; i < i_end; ++i, data += stride) {
    callback(m_element_tp, child_arrmeta, data, callback_data);
  }
}

void ndt::var_dim_type::get_dynamic_type_properties(const std::pair<std::string, nd::callable> **out_properties,
                                                    size_t *out_count) const
{
  struct get_element_type {
    type tp;

    get_element_type(type tp) : tp(tp)
    {
    }

    type operator()() const
    {
      return tp.extended<base_dim_type>()->get_element_type();
    }
  };

  static pair<std::string, nd::callable> var_dim_type_properties[] = {
      pair<std::string, nd::callable>("element_type", nd::functional::apply<get_element_type, type>("self"))};

  *out_properties = var_dim_type_properties;
  *out_count = sizeof(var_dim_type_properties) / sizeof(var_dim_type_properties[0]);
}

void ndt::var_dim_type::get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties,
                                                     size_t *out_count) const
{
  *out_properties = m_array_properties.empty() ? NULL : &m_array_properties[0];
  *out_count = (int)m_array_properties.size();
}

void ndt::var_dim_type::get_dynamic_array_functions(const std::pair<std::string, gfunc::callable> **out_functions,
                                                    size_t *out_count) const
{
  *out_functions = m_array_functions.empty() ? NULL : &m_array_functions[0];
  *out_count = (int)m_array_functions.size();
}

ndt::type ndt::var_dim_type::with_element_type(const type &element_tp) const
{
  return make(element_tp);
}

void ndt::var_dim_element_initialize(const type &tp, const char *arrmeta, char *data, intptr_t count)
{
  if (tp.get_type_id() != var_dim_type_id) {
    stringstream ss;
    ss << "internal error: expected a var_dim type, not " << tp;
    throw dynd::type_error(ss.str());
  }
  const var_dim_type *vdt = tp.extended<var_dim_type>();
  const var_dim_type_arrmeta *md = reinterpret_cast<const var_dim_type_arrmeta *>(arrmeta);
  var_dim_type_data *d = reinterpret_cast<var_dim_type_data *>(data);
  if (d->begin != NULL) {
    throw runtime_error("internal error: var_dim element data must be NULL to initialize");
  }
  if (md->offset != 0) {
    throw runtime_error("internal error: var_dim arrmeta offset must be "
                        "zero to initialize");
  }
  // Allocate the element
  memory_block_data *memblock = md->blockref;
  if (memblock == NULL) {
    throw runtime_error("internal error: var_dim arrmeta has no memblock");
  } else if (memblock->m_type == objectarray_memory_block_type) {
    memory_block_objectarray_allocator_api *allocator = get_memory_block_objectarray_allocator_api(memblock);

    // Allocate the output array data
    d->begin = allocator->allocate(memblock, count);
    d->size = count;
  } else if (memblock->m_type == pod_memory_block_type || memblock->m_type == zeroinit_memory_block_type) {
    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(memblock);

    // Allocate the output array data
    char *dst_end = NULL;
    allocator->allocate(memblock, count * md->stride, vdt->get_target_alignment(), &d->begin, &dst_end);
    d->size = count;
  } else {
    stringstream ss;
    ss << "var_dim_element_initialize internal error: ";
    ss << "var_dim arrmeta has memblock type " << (memory_block_type_t)memblock->m_type;
    ss << " that is not writable";
    throw runtime_error(ss.str());
  }
}

void ndt::var_dim_element_resize(const type &tp, const char *arrmeta, char *data, intptr_t count)
{
  if (tp.get_type_id() != var_dim_type_id) {
    stringstream ss;
    ss << "internal error: expected a var_dim type, not " << tp;
    throw dynd::type_error(ss.str());
  }
  const var_dim_type_arrmeta *md = reinterpret_cast<const var_dim_type_arrmeta *>(arrmeta);
  var_dim_type_data *d = reinterpret_cast<var_dim_type_data *>(data);
  if (d->begin == NULL) {
    // Allow resize to do the initialization as well
    var_dim_element_initialize(tp, arrmeta, data, count);
    return;
  }
  // Resize the element
  memory_block_data *memblock = md->blockref;
  if (memblock == NULL) {
    throw runtime_error("internal error: var_dim arrmeta has no memblock");
  } else if (memblock->m_type == objectarray_memory_block_type) {
    memory_block_objectarray_allocator_api *allocator = get_memory_block_objectarray_allocator_api(memblock);

    // Resize the output array data
    d->begin = allocator->resize(memblock, d->begin, count);
    d->size = count;
  } else if (memblock->m_type == pod_memory_block_type || memblock->m_type == zeroinit_memory_block_type) {
    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(memblock);

    // Resize the output array data
    char *dst_end = d->begin + d->size * md->stride;
    allocator->resize(memblock, count * md->stride, &d->begin, &dst_end);
    d->size = count;
  } else {
    stringstream ss;
    ss << "var_dim_element_resize internal error: ";
    ss << "var_dim arrmeta has memblock type " << (memory_block_type_t)memblock->m_type;
    ss << " that is not writable";
    throw runtime_error(ss.str());
  }
}
