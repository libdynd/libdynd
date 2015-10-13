//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/callable.hpp>
#include <dynd/func/apply.hpp>
#include <dynd/types/c_contiguous_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/kernels/elwise.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/types/typevar_type.hpp>

using namespace std;
using namespace dynd;

ndt::fixed_dim_type::fixed_dim_type(intptr_t dim_size, const type &element_tp)
    : base_dim_type(fixed_dim_type_id, element_tp, 0, element_tp.get_data_alignment(), sizeof(fixed_dim_type_arrmeta),
                    type_flag_none, true),
      m_dim_size(dim_size)
{
  // Propagate the inherited flags from the element
  m_members.flags |= (element_tp.get_flags() & (type_flags_operand_inherited | type_flags_value_inherited));
  // Copy nd::array properties and functions from the first non-array dimension
  get_scalar_properties_and_functions(m_array_properties, m_array_functions);
}

ndt::fixed_dim_type::~fixed_dim_type()
{
}

size_t ndt::fixed_dim_type::get_default_data_size() const
{
  if (!m_element_tp.is_builtin()) {
    return m_dim_size * m_element_tp.extended()->get_default_data_size();
  } else {
    return m_dim_size * m_element_tp.get_data_size();
  }
}

void ndt::fixed_dim_type::print_data(std::ostream &o, const char *arrmeta, const char *data) const
{
  const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta);
  strided_array_summarized(o, get_element_type(), arrmeta + sizeof(fixed_dim_type_arrmeta), data, m_dim_size,
                           md->stride);
}

void ndt::fixed_dim_type::print_type(std::ostream &o) const
{
  o << m_dim_size << " * " << m_element_tp;
}

bool ndt::fixed_dim_type::is_c_contiguous(const char *arrmeta) const
{
  if (arrmeta == NULL) {
    return false;
  }

  const size_stride_t *ss = reinterpret_cast<const size_stride_t *>(arrmeta);
  if (static_cast<intptr_t>(m_element_tp.get_default_data_size()) == ss->stride) {
    return m_element_tp.is_c_contiguous(arrmeta + sizeof(size_stride_t));
  }

  return false;
}

bool ndt::fixed_dim_type::is_expression() const
{
  return m_element_tp.is_expression();
}

bool ndt::fixed_dim_type::is_unique_data_owner(const char *arrmeta) const
{
  if (m_element_tp.is_builtin()) {
    return true;
  } else {
    return m_element_tp.extended()->is_unique_data_owner(arrmeta + sizeof(fixed_dim_type_arrmeta));
  }
}

void ndt::fixed_dim_type::transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                                                type &out_transformed_tp, bool &out_was_transformed) const
{
  type tmp_tp;
  bool was_transformed = false;
  transform_fn(m_element_tp, arrmeta_offset + sizeof(fixed_dim_type_arrmeta), extra, tmp_tp, was_transformed);
  if (was_transformed) {
    out_transformed_tp = type(new fixed_dim_type(m_dim_size, tmp_tp), false);
    out_was_transformed = true;
  } else {
    out_transformed_tp = type(this, true);
  }
}

ndt::type ndt::fixed_dim_type::get_canonical_type() const
{
  return type(new fixed_dim_type(m_dim_size, m_element_tp.get_canonical_type()), false);
}

ndt::type ndt::fixed_dim_type::apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i,
                                                  const type &root_tp, bool leading_dimension) const
{
  if (nindices == 0) {
    return type(this, true);
  } else {
    bool remove_dimension;
    intptr_t start_index, index_stride, dimension_size;
    apply_single_linear_index(*indices, m_dim_size, current_i, &root_tp, remove_dimension, start_index, index_stride,
                              dimension_size);
    if (remove_dimension) {
      return m_element_tp.apply_linear_index(nindices - 1, indices + 1, current_i + 1, root_tp, leading_dimension);
    } else {
      return make_fixed_dim(dimension_size,
                            m_element_tp.apply_linear_index(nindices - 1, indices + 1, current_i + 1, root_tp, false));
    }
  }
}

intptr_t ndt::fixed_dim_type::apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta,
                                                 const type &result_tp, char *out_arrmeta,
                                                 memory_block_data *embedded_reference, size_t current_i,
                                                 const type &root_tp, bool leading_dimension, char **inout_data,
                                                 memory_block_data **inout_dataref) const
{
  const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta);
  fixed_dim_type_arrmeta *out_md = reinterpret_cast<fixed_dim_type_arrmeta *>(out_arrmeta);
  if (nindices == 0) {
    // If there are no more indices, copy the rest verbatim
    arrmeta_copy_construct(out_arrmeta, arrmeta, embedded_reference);
    return 0;
  } else {
    bool remove_dimension;
    intptr_t start_index, index_stride, dimension_size;
    apply_single_linear_index(*indices, m_dim_size, current_i, &root_tp, remove_dimension, start_index, index_stride,
                              dimension_size);
    if (remove_dimension) {
      // Apply the strided offset and continue applying the index
      intptr_t offset = md->stride * start_index;
      if (!m_element_tp.is_builtin()) {
        if (leading_dimension) {
          // In the case of a leading dimension, first bake the offset into
          // the data pointer, so that it's pointing at the right element
          // for the collapsing of leading dimensions to work correctly.
          *inout_data += offset;
          offset = m_element_tp.extended()->apply_linear_index(
              nindices - 1, indices + 1, arrmeta + sizeof(fixed_dim_type_arrmeta), result_tp, out_arrmeta,
              embedded_reference, current_i + 1, root_tp, true, inout_data, inout_dataref);
        } else {
          offset += m_element_tp.extended()->apply_linear_index(
              nindices - 1, indices + 1, arrmeta + sizeof(fixed_dim_type_arrmeta), result_tp, out_arrmeta,
              embedded_reference, current_i + 1, root_tp, false, NULL, NULL);
        }
      }
      return offset;
    } else {
      // Produce the new offset data, stride, and size for the resulting array
      intptr_t offset = md->stride * start_index;
      out_md->stride = md->stride * index_stride;
      out_md->dim_size = dimension_size;
      if (!m_element_tp.is_builtin()) {
        const fixed_dim_type *result_etp = result_tp.extended<fixed_dim_type>();
        offset += m_element_tp.extended()->apply_linear_index(
            nindices - 1, indices + 1, arrmeta + sizeof(fixed_dim_type_arrmeta), result_etp->get_element_type(),
            out_arrmeta + sizeof(fixed_dim_type_arrmeta), embedded_reference, current_i + 1, root_tp, false, NULL,
            NULL);
      }
      return offset;
    }
  }
}

ndt::type ndt::fixed_dim_type::at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const
{
  // Bounds-checking of the index
  i0 = apply_single_index(i0, m_dim_size, NULL);
  if (inout_arrmeta) {
    const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(*inout_arrmeta);
    // Modify the arrmeta
    *inout_arrmeta += sizeof(fixed_dim_type_arrmeta);
    // If requested, modify the data
    if (inout_data) {
      *inout_data += i0 * md->stride;
    }
  }
  return m_element_tp;
}

ndt::type ndt::fixed_dim_type::get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim) const
{
  if (i == 0) {
    return type(this, true);
  } else {
    if (inout_arrmeta) {
      *inout_arrmeta += sizeof(fixed_dim_type_arrmeta);
    }
    return m_element_tp.get_type_at_dimension(inout_arrmeta, i - 1, total_ndim + 1);
  }
}

intptr_t ndt::fixed_dim_type::get_dim_size(const char *DYND_UNUSED(arrmeta), const char *DYND_UNUSED(data)) const
{
  return m_dim_size;
}

void ndt::fixed_dim_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta,
                                    const char *data) const
{
  out_shape[i] = m_dim_size;
  if (m_dim_size != 1) {
    data = NULL;
  }

  // Process the later shape values
  if (i + 1 < ndim) {
    if (!m_element_tp.is_builtin()) {
      m_element_tp.extended()->get_shape(ndim, i + 1, out_shape,
                                         arrmeta ? (arrmeta + sizeof(fixed_dim_type_arrmeta)) : NULL, data);
    } else {
      stringstream ss;
      ss << "requested too many dimensions from type " << type(this, true);
      throw runtime_error(ss.str());
    }
  }
}

void ndt::fixed_dim_type::get_strides(size_t i, intptr_t *out_strides, const char *arrmeta) const
{
  const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta);

  out_strides[i] = md->stride;

  // Process the later shape values
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->get_strides(i + 1, out_strides, arrmeta + sizeof(fixed_dim_type_arrmeta));
  }
}

axis_order_classification_t ndt::fixed_dim_type::classify_axis_order(const char *arrmeta) const
{
  const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta);
  if (m_element_tp.get_ndim() > 0) {
    if (md->stride != 0) {
      // Call the helper function to do the classification
      return classify_strided_axis_order(md->stride >= 0 ? md->stride : -md->stride, m_element_tp,
                                         arrmeta + sizeof(fixed_dim_type_arrmeta));
    } else {
      // Use the classification of the element type
      return m_element_tp.extended()->classify_axis_order(arrmeta + sizeof(fixed_dim_type_arrmeta));
    }
  } else {
    return axis_order_none;
  }
}

bool ndt::fixed_dim_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    } else if (src_tp.get_type_id() == fixed_dim_type_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

bool ndt::fixed_dim_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != fixed_dim_type_id) {
    return false;
  } else if (rhs.get_kind() == kind_kind) {
    return false;
  } else {
    const fixed_dim_type *dt = static_cast<const fixed_dim_type *>(&rhs);
    return m_element_tp == dt->m_element_tp && m_dim_size == dt->m_dim_size;
  }
}

void ndt::fixed_dim_type::arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const
{
  size_t element_size =
      m_element_tp.is_builtin() ? m_element_tp.get_data_size() : m_element_tp.extended()->get_default_data_size();

  fixed_dim_type_arrmeta *md = reinterpret_cast<fixed_dim_type_arrmeta *>(arrmeta);
  md->dim_size = get_fixed_dim_size();
  md->stride = m_dim_size > 1 ? element_size : 0;
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_default_construct(arrmeta + sizeof(fixed_dim_type_arrmeta), blockref_alloc);
  }
}

void ndt::fixed_dim_type::arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                                 memory_block_data *embedded_reference) const
{
  const fixed_dim_type_arrmeta *src_md = reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta);
  fixed_dim_type_arrmeta *dst_md = reinterpret_cast<fixed_dim_type_arrmeta *>(dst_arrmeta);
  *dst_md = *src_md;
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_copy_construct(dst_arrmeta + sizeof(fixed_dim_type_arrmeta),
                                                    src_arrmeta + sizeof(fixed_dim_type_arrmeta), embedded_reference);
  }
}

size_t ndt::fixed_dim_type::arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                                                          memory_block_data *DYND_UNUSED(embedded_reference)) const
{
  const fixed_dim_type_arrmeta *src_md = reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta);
  fixed_dim_type_arrmeta *dst_md = reinterpret_cast<fixed_dim_type_arrmeta *>(dst_arrmeta);
  *dst_md = *src_md;
  return sizeof(fixed_dim_type_arrmeta);
}

void ndt::fixed_dim_type::arrmeta_reset_buffers(char *arrmeta) const
{
  if (m_element_tp.get_arrmeta_size() > 0) {
    m_element_tp.extended()->arrmeta_reset_buffers(arrmeta + sizeof(fixed_dim_type_arrmeta));
  }
}

void ndt::fixed_dim_type::arrmeta_finalize_buffers(char *arrmeta) const
{
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_finalize_buffers(arrmeta + sizeof(fixed_dim_type_arrmeta));
  }
}

void ndt::fixed_dim_type::arrmeta_destruct(char *arrmeta) const
{
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_destruct(arrmeta + sizeof(fixed_dim_type_arrmeta));
  }
}

void ndt::fixed_dim_type::arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const
{
  const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta);
  o << indent << "fixed_dim arrmeta\n";
  o << indent << " size: " << md->dim_size;
  if (md->dim_size != get_fixed_dim_size()) {
    o << " INTERNAL INCONSISTENCY, type size: " << get_fixed_dim_size();
  }
  o << "\n";
  o << indent << " stride: " << md->stride << "\n";
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_debug_print(arrmeta + sizeof(fixed_dim_type_arrmeta), o, indent + " ");
  }
}

size_t ndt::fixed_dim_type::get_iterdata_size(intptr_t ndim) const
{
  if (ndim == 0) {
    return 0;
  } else if (ndim == 1) {
    return sizeof(fixed_dim_type_iterdata);
  } else {
    return m_element_tp.get_iterdata_size(ndim - 1) + sizeof(fixed_dim_type_iterdata);
  }
}

// Does one iterator increment for this type
static char *iterdata_incr(iterdata_common *iterdata, intptr_t level)
{
  fixed_dim_type_iterdata *id = reinterpret_cast<fixed_dim_type_iterdata *>(iterdata);
  if (level == 0) {
    id->data += id->stride;
    return id->data;
  } else {
    id->data = (id + 1)->common.incr(&(id + 1)->common, level - 1);
    return id->data;
  }
}

static char *iterdata_reset(iterdata_common *iterdata, char *data, intptr_t ndim)
{
  fixed_dim_type_iterdata *id = reinterpret_cast<fixed_dim_type_iterdata *>(iterdata);
  if (ndim == 1) {
    id->data = data;
    return data;
  } else {
    id->data = (id + 1)->common.reset(&(id + 1)->common, data, ndim - 1);
    return id->data;
  }
}

size_t ndt::fixed_dim_type::iterdata_construct(iterdata_common *iterdata, const char **inout_arrmeta, intptr_t ndim,
                                               const intptr_t *shape, type &out_uniform_tp) const
{
  const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(*inout_arrmeta);
  *inout_arrmeta += sizeof(fixed_dim_type_arrmeta);
  size_t inner_size = 0;
  if (ndim > 1) {
    // Place any inner iterdata earlier than the outer iterdata
    inner_size =
        m_element_tp.extended()->iterdata_construct(iterdata, inout_arrmeta, ndim - 1, shape + 1, out_uniform_tp);
    iterdata = reinterpret_cast<iterdata_common *>(reinterpret_cast<char *>(iterdata) + inner_size);
  } else {
    out_uniform_tp = m_element_tp;
  }

  fixed_dim_type_iterdata *id = reinterpret_cast<fixed_dim_type_iterdata *>(iterdata);

  id->common.incr = &iterdata_incr;
  id->common.reset = &iterdata_reset;
  id->data = NULL;
  id->stride = md->stride;

  return inner_size + sizeof(fixed_dim_type_iterdata);
}

size_t ndt::fixed_dim_type::iterdata_destruct(iterdata_common *iterdata, intptr_t ndim) const
{
  size_t inner_size = 0;
  if (ndim > 1) {
    inner_size = m_element_tp.extended()->iterdata_destruct(iterdata, ndim - 1);
  }
  // No dynamic data to free
  return inner_size + sizeof(fixed_dim_type_iterdata);
}

void ndt::fixed_dim_type::data_destruct(const char *arrmeta, char *data) const
{
  const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta);
  m_element_tp.extended()->data_destruct_strided(arrmeta + sizeof(fixed_dim_type_arrmeta), data, md->stride,
                                                 m_dim_size);
}

void ndt::fixed_dim_type::data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const
{
  const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta);
  arrmeta += sizeof(fixed_dim_type_arrmeta);
  intptr_t child_stride = md->stride;
  size_t child_size = m_dim_size;

  for (size_t i = 0; i != count; ++i, data += stride) {
    m_element_tp.extended()->data_destruct_strided(arrmeta, data, child_stride, child_size);
  }
}

intptr_t ndt::fixed_dim_type::make_assignment_kernel(void *ckb, intptr_t ckb_offset, const type &dst_tp,
                                                     const char *dst_arrmeta, const type &src_tp,
                                                     const char *src_arrmeta, kernel_request_t kernreq,
                                                     const eval::eval_context *ectx) const
{
  if (this == dst_tp.extended()) {
    const fixed_dim_type_arrmeta *dst_md = reinterpret_cast<const fixed_dim_type_arrmeta *>(dst_arrmeta);
    intptr_t src_size, src_stride;
    type src_el_tp;
    const char *src_el_arrmeta;

    if (src_tp.get_ndim() < dst_tp.get_ndim()) {
      src_stride = 0;
      nd::functional::elwise_ck<fixed_dim_type_id, fixed_dim_type_id, 1>::make(
          ckb, kernreq, ckb_offset, get_fixed_dim_size(), dst_md->stride, &src_stride);

      return ::make_assignment_kernel(ckb, ckb_offset, m_element_tp, dst_arrmeta + sizeof(fixed_dim_type_arrmeta),
                                      src_tp, src_arrmeta, kernel_request_strided, ectx);
    } else if (src_tp.get_as_strided(src_arrmeta, &src_size, &src_stride, &src_el_tp, &src_el_arrmeta)) {
      nd::functional::elwise_ck<fixed_dim_type_id, fixed_dim_type_id, 1>::make(
          ckb, kernreq, ckb_offset, get_fixed_dim_size(), dst_md->stride, &src_stride);

      // Check for a broadcasting error
      if (src_size != 1 && get_fixed_dim_size() != src_size) {
        throw broadcast_error(dst_tp, dst_arrmeta, src_tp, src_arrmeta);
      }

      return ::make_assignment_kernel(ckb, ckb_offset, m_element_tp, dst_arrmeta + sizeof(fixed_dim_type_arrmeta),
                                      src_el_tp, src_el_arrmeta, kernel_request_strided, ectx);
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
    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw dynd::type_error(ss.str());
  }
}

void ndt::fixed_dim_type::foreach_leading(const char *arrmeta, char *data, foreach_fn_t callback,
                                          void *callback_data) const
{
  const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta);
  const char *child_arrmeta = arrmeta + sizeof(fixed_dim_type_arrmeta);
  intptr_t stride = md->stride;
  for (intptr_t i = 0, i_end = m_dim_size; i < i_end; ++i, data += stride) {
    callback(m_element_tp, child_arrmeta, data, callback_data);
  }
}

ndt::type ndt::make_fixed_dim(size_t dim_size, const type &element_tp)
{
  return type(new fixed_dim_type(dim_size, element_tp), false);
}

ndt::type ndt::make_fixed_dim(intptr_t ndim, const intptr_t *shape, const type &uniform_tp)
{
  type result = uniform_tp;
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    result = make_fixed_dim(shape[i], result);
  }
  return result;
}

void ndt::fixed_dim_type::reorder_default_constructed_strides(char *dst_arrmeta, const type &src_tp,
                                                              const char *src_arrmeta) const
{
  if (m_element_tp.get_type_id() != fixed_dim_type_id) {
    // Nothing to do if there's just one reorderable dimension
    return;
  }

  if (get_ndim() > src_tp.get_ndim()) {
    // If the destination has more dimensions than the source,
    // do the reordering starting from where they match, to
    // follow the broadcasting rules.
    if (m_element_tp.get_type_id() == fixed_dim_type_id) {
      const fixed_dim_type *sdd = m_element_tp.extended<fixed_dim_type>();
      sdd->reorder_default_constructed_strides(dst_arrmeta + sizeof(fixed_dim_type_arrmeta), src_tp, src_arrmeta);
    }
    return;
  }

  // Find the total number of dimensions we might be reordering, then process
  // them all at once. This code handles a whole chain of fixed_dim_type
  // instances at once.
  size_t ndim = 1;
  type last_dt = m_element_tp;
  do {
    ++ndim;
    last_dt = last_dt.extended<fixed_dim_type>()->get_element_type();
  } while (last_dt.get_type_id() == fixed_dim_type_id);

  dimvector strides(ndim);
  type last_src_tp = src_tp;
  intptr_t previous_stride = 0;
  size_t ndim_partial = 0;
  // Get representative strides from all the strided source dimensions
  bool c_order = true;
  for (size_t i = 0; i < ndim; ++i) {
    intptr_t stride;
    switch (last_src_tp.get_type_id()) {
    case fixed_dim_type_id: {
      const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta);
      stride = md->stride;
      last_src_tp = last_src_tp.extended<base_dim_type>()->get_element_type();
      src_arrmeta += sizeof(fixed_dim_type_arrmeta);
      break;
    }
    default:
      stride = numeric_limits<intptr_t>::max();
      break;
    }
    ndim_partial = i + 1;
    // To check for C-order, we skip over any 0-strides, and
    // check if a stride ever gets  bigger instead of always
    // getting smaller.
    if (stride != 0) {
      if (stride == numeric_limits<intptr_t>::max()) {
        break;
      }
      if (previous_stride != 0 && previous_stride < stride) {
        c_order = false;
      }
      previous_stride = stride;
    }
    strides[i] = stride;
  }

  // If it wasn't all C-order, reorder the axes
  if (!c_order) {
    shortvector<int> axis_perm(ndim_partial);
    strides_to_axis_perm(ndim_partial, strides.get(), axis_perm.get());
    fixed_dim_type_arrmeta *md = reinterpret_cast<fixed_dim_type_arrmeta *>(dst_arrmeta);
    intptr_t stride = md[ndim_partial - 1].stride;
    if (stride == 0) {
      // Because of the rule that size one dimensions have
      // zero stride, may have to look further
      intptr_t i = ndim_partial - 2;
      do {
        stride = md[i].stride;
      } while (stride == 0 && i >= 0);
    }
    for (size_t i = 0; i < ndim_partial; ++i) {
      int i_perm = axis_perm[i];
      fixed_dim_type_arrmeta &i_md = md[i_perm];
      intptr_t dim_size = i_md.dim_size;
      i_md.stride = dim_size > 1 ? stride : 0;
      stride *= dim_size;
    }
  }

  // If that didn't cover all the dimensions, then get the
  // axis order classification to handle the rest
  if (ndim_partial < ndim && !last_src_tp.is_builtin()) {
    axis_order_classification_t aoc = last_src_tp.extended()->classify_axis_order(src_arrmeta);
    // TODO: Allow user control by adding a "default axis order" to the
    // evaluation context
    if (aoc == axis_order_f) {
      // If it's F-order, reverse the ordering of the strides
      fixed_dim_type_arrmeta *md = reinterpret_cast<fixed_dim_type_arrmeta *>(dst_arrmeta);
      intptr_t stride = md[ndim - 1].stride;
      if (stride == 0) {
        // Because of the rule that size one dimensions have
        // zero stride, may have to look further
        intptr_t i = ndim - 2;
        do {
          stride = md[i].stride;
        } while (stride == 0 && i >= (intptr_t)ndim_partial);
      }
      for (size_t i = ndim_partial; i != ndim; ++i) {
        intptr_t dim_size = md[i].dim_size;
        md[i].stride = dim_size > 1 ? stride : 0;
        stride *= dim_size;
      }
    }
  }
}

bool ndt::fixed_dim_type::match(const char *arrmeta, const type &candidate_tp, const char *candidate_arrmeta,
                                std::map<std::string, type> &tp_vars) const
{
  switch (candidate_tp.get_type_id()) {
  case fixed_dim_type_id:
    // TODO XXX If the arrmeta is not NULL, the strides should be checked too
    return get_fixed_dim_size() == candidate_tp.extended<fixed_dim_type>()->get_fixed_dim_size() &&
           m_element_tp.match((arrmeta == NULL) ? arrmeta : (arrmeta + sizeof(fixed_dim_type_arrmeta)),
                              candidate_tp.extended<fixed_dim_type>()->m_element_tp,
                              (candidate_arrmeta == NULL) ? candidate_arrmeta
                                                          : (candidate_arrmeta + sizeof(fixed_dim_type_arrmeta)),
                              tp_vars);
  case c_contiguous_type_id:
    return is_c_contiguous(arrmeta) &&
           match(arrmeta, candidate_tp.extended<c_contiguous_type>()->get_child_type(), candidate_arrmeta, tp_vars);
  default:
    return false;
  }
}

void ndt::fixed_dim_type::get_dynamic_type_properties(const std::pair<std::string, nd::callable> **out_properties,
                                                      size_t *out_count) const
{
  static pair<std::string, nd::callable> fixed_dim_type_properties[] = {
      pair<std::string, nd::callable>(
          "fixed_dim_size",
          nd::functional::apply([](type self) { return self.extended<fixed_dim_type>()->get_fixed_dim_size(); },
                                "self")),
      pair<std::string, nd::callable>(
          "element_type",
          nd::functional::apply([](type self) { return type(self.extended<fixed_dim_type>()->get_element_type()); },
                                "self"))};

  *out_properties = fixed_dim_type_properties;
  *out_count = sizeof(fixed_dim_type_properties) / sizeof(fixed_dim_type_properties[0]);
}

void ndt::fixed_dim_type::get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties,
                                                       size_t *out_count) const
{
  *out_properties = m_array_properties.empty() ? NULL : &m_array_properties[0];
  *out_count = (int)m_array_properties.size();
}

void ndt::fixed_dim_type::get_dynamic_array_functions(const std::pair<std::string, gfunc::callable> **out_functions,
                                                      size_t *out_count) const
{
  *out_functions = m_array_functions.empty() ? NULL : &m_array_functions[0];
  *out_count = (int)m_array_functions.size();
}

ndt::type ndt::fixed_dim_type::with_element_type(const type &element_tp) const
{
  return make_fixed_dim(m_dim_size, element_tp);
}
