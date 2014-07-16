//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/func/make_callable.hpp>
#include <dynd/types/builtin_type_properties.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>

using namespace std;
using namespace dynd;

strided_dim_type::strided_dim_type(const ndt::type& element_tp)
    : base_dim_type(strided_dim_type_id, element_tp, 0, element_tp.get_data_alignment(),
                    sizeof(strided_dim_type_arrmeta), type_flag_none, true)
{
  // Propagate the inherited flags from the element
  m_members.flags |=
      (element_tp.get_flags() &
       ((type_flags_operand_inherited | type_flags_value_inherited) &
        ~type_flag_scalar));
}

strided_dim_type::~strided_dim_type()
{
}

size_t strided_dim_type::get_default_data_size(intptr_t ndim,
                                               const intptr_t *shape) const
{
    if (ndim == 0) {
        throw std::runtime_error("the strided_dim type requires a shape be "
                                 "specified for default construction");
    } else if (shape[0] < 0) {
        throw std::runtime_error("the strided_dim type requires a non-negative "
                                 "shape to be specified for default "
                                 "construction");
    }

    if (!m_element_tp.is_builtin()) {
        return shape[0] * m_element_tp.extended()->get_default_data_size(
                              ndim - 1, shape + 1);
    } else {
        return shape[0] * m_element_tp.get_data_size();
    }
}

void strided_dim_type::print_data(std::ostream &o, const char *arrmeta,
                                  const char *data) const
{
  const strided_dim_type_arrmeta *md =
      reinterpret_cast<const strided_dim_type_arrmeta *>(arrmeta);
  strided_array_summarized(o, get_element_type(),
                           arrmeta + sizeof(strided_dim_type_arrmeta), data,
                           md->dim_size, md->stride);
}

void strided_dim_type::print_type(std::ostream& o) const
{
  o << "strided * " << m_element_tp;
}

bool strided_dim_type::is_expression() const
{
  return m_element_tp.is_expression();
}

bool strided_dim_type::is_unique_data_owner(const char *arrmeta) const
{
    if (m_element_tp.is_builtin()) {
        return true;
    } else {
        return m_element_tp.extended()->is_unique_data_owner(
            arrmeta + sizeof(strided_dim_type_arrmeta));
    }
}

void strided_dim_type::transform_child_types(type_transform_fn_t transform_fn,
                                             void *extra,
                                             ndt::type &out_transformed_tp,
                                             bool &out_was_transformed) const
{
    ndt::type tmp_tp;
    bool was_transformed = false;
    transform_fn(m_element_tp, extra, tmp_tp, was_transformed);
    if (was_transformed) {
        out_transformed_tp = ndt::type(new strided_dim_type(tmp_tp), false);
        out_was_transformed = true;
    } else {
        out_transformed_tp = ndt::type(this, true);
    }
}


ndt::type strided_dim_type::get_canonical_type() const
{
    return ndt::type(new strided_dim_type(m_element_tp.get_canonical_type()),
                     false);
}

ndt::type strided_dim_type::apply_linear_index(intptr_t nindices,
                                               const irange *indices,
                                               size_t current_i,
                                               const ndt::type &root_tp,
                                               bool leading_dimension) const
{
    if (nindices == 0) {
        return ndt::type(this, true);
    } else if (nindices == 1) {
        if (indices->step() == 0) {
            if (leading_dimension && !m_element_tp.is_builtin()) {
                // For leading dimensions, need to give the next type a chance
                // to collapse itself even though indexing doesn't continue
                // further.
                return m_element_tp.extended()->apply_linear_index(
                    0, NULL, current_i, root_tp, true);
            } else {
                return m_element_tp;
            }
        } else {
            return ndt::type(this, true);
        }
    } else {
        if (indices->step() == 0) {
            return m_element_tp.apply_linear_index(nindices-1, indices+1,
                            current_i+1, root_tp, leading_dimension);
        } else {
            return ndt::type(
                new strided_dim_type(m_element_tp.apply_linear_index(
                    nindices - 1, indices + 1, current_i + 1, root_tp, false)),
                false);
        }
    }
}

intptr_t strided_dim_type::apply_linear_index(
    intptr_t nindices, const irange *indices, const char *arrmeta,
    const ndt::type &result_tp, char *out_arrmeta,
    memory_block_data *embedded_reference, size_t current_i,
    const ndt::type &root_tp, bool leading_dimension, char **inout_data,
    memory_block_data **inout_dataref) const
{
    const strided_dim_type_arrmeta *md =
        reinterpret_cast<const strided_dim_type_arrmeta *>(arrmeta);
    strided_dim_type_arrmeta *out_md =
        reinterpret_cast<strided_dim_type_arrmeta *>(out_arrmeta);
    if (nindices == 0) {
        // If there are no more indices, copy the rest verbatim
        arrmeta_copy_construct(out_arrmeta, arrmeta, embedded_reference);
        return 0;
    } else {
        bool remove_dimension;
        intptr_t start_index, index_stride, dimension_size;
        apply_single_linear_index(*indices, md->dim_size, current_i, &root_tp,
                        remove_dimension, start_index, index_stride, dimension_size);
        if (remove_dimension) {
            // Apply the strided offset and continue applying the index
            intptr_t offset = md->stride * start_index;
            if (!m_element_tp.is_builtin()) {
                if (leading_dimension) {
                    // In the case of a leading dimension, first bake the offset into
                    // the data pointer, so that it's pointing at the right element
                    // for the collapsing of leading dimensions to work correctly.
                    *inout_data += offset;
                    offset = m_element_tp.extended()->apply_linear_index(nindices - 1, indices + 1,
                                    arrmeta + sizeof(strided_dim_type_arrmeta),
                                    result_tp, out_arrmeta,
                                    embedded_reference, current_i + 1, root_tp,
                                    true, inout_data, inout_dataref);
                } else {
                    offset += m_element_tp.extended()->apply_linear_index(nindices - 1, indices + 1,
                                    arrmeta + sizeof(strided_dim_type_arrmeta),
                                    result_tp, out_arrmeta,
                                    embedded_reference, current_i + 1, root_tp,
                                    false, NULL, NULL);
                }
            }
            return offset;
        } else {
            // Produce the new offset data, stride, and size for the resulting array
            intptr_t offset = md->stride * start_index;
            out_md->stride = md->stride * index_stride;
            out_md->dim_size = dimension_size;
            if (!m_element_tp.is_builtin()) {
                const strided_dim_type *result_etp = result_tp.tcast<strided_dim_type>();
                offset += m_element_tp.extended()->apply_linear_index(nindices - 1, indices + 1,
                                arrmeta + sizeof(strided_dim_type_arrmeta),
                                result_etp->m_element_tp, out_arrmeta + sizeof(strided_dim_type_arrmeta),
                                embedded_reference, current_i + 1, root_tp,
                                false, NULL, NULL);
            }
            return offset;
        }
    }
}

ndt::type strided_dim_type::at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const
{
    if (inout_arrmeta) {
        const strided_dim_type_arrmeta *md = reinterpret_cast<const strided_dim_type_arrmeta *>(*inout_arrmeta);
        // Bounds-checking of the index
        i0 = apply_single_index(i0, md->dim_size, NULL);
        // Modify the arrmeta
        *inout_arrmeta += sizeof(strided_dim_type_arrmeta);
        // If requested, modify the data
        if (inout_data) {
            *inout_data += i0 * md->stride;
        }
    }
    return m_element_tp;
}

ndt::type strided_dim_type::get_type_at_dimension(char **inout_arrmeta,
                                                  intptr_t i,
                                                  intptr_t total_ndim) const
{
    if (i == 0) {
        return ndt::type(this, true);
    } else {
        if (inout_arrmeta) {
            *inout_arrmeta += sizeof(strided_dim_type_arrmeta);
        }
        return m_element_tp.get_type_at_dimension(inout_arrmeta, i - 1,
                                                  total_ndim + 1);
    }
}

intptr_t strided_dim_type::get_dim_size(const char *arrmeta,
                                        const char *DYND_UNUSED(data)) const
{
  if (arrmeta != NULL) {
    return reinterpret_cast<const strided_dim_type_arrmeta *>(arrmeta)
        ->dim_size;
  } else {
    return -1;
  }
}

void strided_dim_type::get_shape(intptr_t ndim, intptr_t i,
                intptr_t *out_shape, const char *arrmeta, const char *data) const
{
    if (arrmeta) {
        const strided_dim_type_arrmeta *md = reinterpret_cast<const strided_dim_type_arrmeta *>(arrmeta);
        out_shape[i] = md->dim_size;
        if (md->dim_size != 1) {
            data = NULL;
        }
    } else {
        out_shape[i] = -1;
        data = NULL;
    }

    // Process the later shape values
    if (i+1 < ndim) {
        if (!m_element_tp.is_builtin()) {
            m_element_tp.extended()->get_shape(ndim, i+1, out_shape,
                            arrmeta ? (arrmeta + sizeof(strided_dim_type_arrmeta)) : NULL,
                            data);
        } else {
            stringstream ss;
            ss << "requested too many dimensions from type " << ndt::type(this, true);
            throw runtime_error(ss.str());
        }
    }
}

void strided_dim_type::get_strides(size_t i, intptr_t *out_strides, const char *arrmeta) const
{
    const strided_dim_type_arrmeta *md = reinterpret_cast<const strided_dim_type_arrmeta *>(arrmeta);

    out_strides[i] = md->stride;

    // Process the later shape values
    if (!m_element_tp.is_builtin()) {
        m_element_tp.extended()->get_strides(i+1, out_strides, arrmeta + sizeof(strided_dim_type_arrmeta));
    }
}

axis_order_classification_t strided_dim_type::classify_axis_order(const char *arrmeta) const
{
    const strided_dim_type_arrmeta *md = reinterpret_cast<const strided_dim_type_arrmeta *>(arrmeta);
    if (m_element_tp.get_ndim() > 0) {
        if (md->stride != 0) {
            // Call the helper function to do the classification
            return classify_strided_axis_order(md->stride >= 0 ? md->stride : -md->stride, m_element_tp,
                            arrmeta + sizeof(strided_dim_type_arrmeta));
        } else {
            // Use the classification of the element type
            return m_element_tp.extended()->classify_axis_order(
                            arrmeta + sizeof(strided_dim_type_arrmeta));
        }
    } else {
        return axis_order_none;
    }
}

bool strided_dim_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    if (dst_tp.extended() == this) {
        if (src_tp.extended() == this) {
            return true;
        } else if (src_tp.get_type_id() == strided_dim_type_id) {
            return *dst_tp.extended() == *src_tp.extended();
        }
    }

    return false;
}

bool strided_dim_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != strided_dim_type_id) {
        return false;
    } else {
        const strided_dim_type *dt = static_cast<const strided_dim_type*>(&rhs);
        return m_element_tp == dt->m_element_tp;
    }
}

void strided_dim_type::arrmeta_default_construct(char *arrmeta, intptr_t ndim, const intptr_t* shape) const
{
    // Validate that the shape is ok
    if (ndim == 0 || shape[0] < 0) {
        throw std::runtime_error("the strided_dim type requires a shape be specified for default construction");
    }
    size_t element_size = m_element_tp.is_builtin() ? m_element_tp.get_data_size()
                                                     : m_element_tp.extended()->get_default_data_size(ndim-1, shape+1);

    strided_dim_type_arrmeta *md = reinterpret_cast<strided_dim_type_arrmeta *>(arrmeta);
    md->dim_size = shape[0];
    if (shape[0] > 1) {
        md->stride = element_size;
    } else {
        md->stride = 0;
    }
    if (!m_element_tp.is_builtin()) {
        m_element_tp.extended()->arrmeta_default_construct(arrmeta + sizeof(strided_dim_type_arrmeta), ndim-1, shape+1);
    }
}

void strided_dim_type::arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta, memory_block_data *embedded_reference) const
{
    const strided_dim_type_arrmeta *src_md = reinterpret_cast<const strided_dim_type_arrmeta *>(src_arrmeta);
    strided_dim_type_arrmeta *dst_md = reinterpret_cast<strided_dim_type_arrmeta *>(dst_arrmeta);
    dst_md->dim_size = src_md->dim_size;
    dst_md->stride = src_md->stride;
    if (!m_element_tp.is_builtin()) {
        m_element_tp.extended()->arrmeta_copy_construct(dst_arrmeta + sizeof(strided_dim_type_arrmeta),
                        src_arrmeta + sizeof(strided_dim_type_arrmeta), embedded_reference);
    }
}

size_t strided_dim_type::arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                memory_block_data *DYND_UNUSED(embedded_reference)) const
{
    const strided_dim_type_arrmeta *src_md = reinterpret_cast<const strided_dim_type_arrmeta *>(src_arrmeta);
    strided_dim_type_arrmeta *dst_md = reinterpret_cast<strided_dim_type_arrmeta *>(dst_arrmeta);
    dst_md->dim_size = src_md->dim_size;
    dst_md->stride = src_md->stride;
    return sizeof(strided_dim_type_arrmeta);
}

void strided_dim_type::arrmeta_reset_buffers(char *arrmeta) const
{
    if (m_element_tp.get_arrmeta_size() > 0) {
        m_element_tp.extended()->arrmeta_reset_buffers(
                        arrmeta + sizeof(strided_dim_type_arrmeta));
    }
}

void strided_dim_type::arrmeta_finalize_buffers(char *arrmeta) const
{
    if (!m_element_tp.is_builtin()) {
        m_element_tp.extended()->arrmeta_finalize_buffers(arrmeta + sizeof(strided_dim_type_arrmeta));
    }
}

void strided_dim_type::arrmeta_destruct(char *arrmeta) const
{
    if (!m_element_tp.is_builtin()) {
        m_element_tp.extended()->arrmeta_destruct(arrmeta + sizeof(strided_dim_type_arrmeta));
    }
}

void strided_dim_type::arrmeta_debug_print(const char *arrmeta, std::ostream &o,
                                           const std::string &indent) const
{
  const strided_dim_type_arrmeta *md =
      reinterpret_cast<const strided_dim_type_arrmeta *>(arrmeta);
  o << indent << "strided_dim arrmeta\n";
  o << indent << " size: " << md->dim_size << "\n";
  o << indent << " stride: " << md->stride << "\n";
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_debug_print(
        arrmeta + sizeof(strided_dim_type_arrmeta), o, indent + " ");
  }
}

size_t strided_dim_type::get_iterdata_size(intptr_t ndim) const
{
    if (ndim == 0) {
        return 0;
    } else if (ndim == 1) {
        return sizeof(strided_dim_type_iterdata);
    } else {
        return m_element_tp.get_iterdata_size(ndim - 1) + sizeof(strided_dim_type_iterdata);
    }
}

// Does one iterator increment for this type
static char *iterdata_incr(iterdata_common *iterdata, intptr_t level)
{
    strided_dim_type_iterdata *id = reinterpret_cast<strided_dim_type_iterdata *>(iterdata);
    if (level == 0) {
        id->data += id->stride;
        return id->data;
    } else {
        id->data = (id + 1)->common.incr(&(id + 1)->common, level - 1);
        return id->data;
    }
}

static char *iterdata_adv(iterdata_common *iterdata, intptr_t level, intptr_t i)
{
    strided_dim_type_iterdata *id = reinterpret_cast<strided_dim_type_iterdata *>(iterdata);
    if (level == 0) {
        id->data += i * id->stride;
        return id->data;
    } else {
        id->data = (id + 1)->common.adv(&(id + 1)->common, level - 1, i);
        return id->data;
    }
}

static char *iterdata_reset(iterdata_common *iterdata, char *data, intptr_t ndim)
{
    strided_dim_type_iterdata *id = reinterpret_cast<strided_dim_type_iterdata *>(iterdata);
    if (ndim == 1) {
        id->data = data;
        return data;
    } else {
        id->data = (id + 1)->common.reset(&(id + 1)->common, data, ndim - 1);
        return id->data;
    }
}

size_t strided_dim_type::iterdata_construct(iterdata_common *iterdata, const char **inout_arrmeta, intptr_t ndim, const intptr_t* shape, ndt::type& out_uniform_tp) const
{
    const strided_dim_type_arrmeta *md = reinterpret_cast<const strided_dim_type_arrmeta *>(*inout_arrmeta);
    *inout_arrmeta += sizeof(strided_dim_type_arrmeta);
    size_t inner_size = 0;
    if (ndim > 1) {
        // Place any inner iterdata earlier than the outer iterdata
        inner_size = m_element_tp.extended()->iterdata_construct(iterdata, inout_arrmeta,
                        ndim - 1, shape + 1, out_uniform_tp);
        iterdata = reinterpret_cast<iterdata_common *>(reinterpret_cast<char *>(iterdata) + inner_size);
    } else {
        out_uniform_tp = m_element_tp;
    }

    strided_dim_type_iterdata *id = reinterpret_cast<strided_dim_type_iterdata *>(iterdata);

    id->common.incr = &iterdata_incr;
    id->common.adv = &iterdata_adv;
    id->common.reset = &iterdata_reset;
    id->data = NULL;
    id->stride = md->stride;

    return inner_size + sizeof(strided_dim_type_iterdata);
}

size_t strided_dim_type::iterdata_destruct(iterdata_common *iterdata, intptr_t ndim) const
{
    size_t inner_size = 0;
    if (ndim > 1) {
        inner_size = m_element_tp.extended()->iterdata_destruct(iterdata, ndim - 1);
    }
    // No dynamic data to free
    return inner_size + sizeof(strided_dim_type_iterdata);
}

void strided_dim_type::data_destruct(const char *arrmeta, char *data) const
{
    const strided_dim_type_arrmeta *md = reinterpret_cast<const strided_dim_type_arrmeta *>(arrmeta);
    m_element_tp.extended()->data_destruct_strided(
                    arrmeta + sizeof(strided_dim_type_arrmeta),
                    data, md->stride, md->dim_size);
}

void strided_dim_type::data_destruct_strided(const char *arrmeta, char *data,
                intptr_t stride, size_t count) const
{
    const strided_dim_type_arrmeta *md = reinterpret_cast<const strided_dim_type_arrmeta *>(arrmeta);
    arrmeta += sizeof(strided_dim_type_arrmeta);
    intptr_t child_stride = md->stride;
    size_t child_size = md->dim_size;

    for (size_t i = 0; i != count; ++i, data += stride) {
        m_element_tp.extended()->data_destruct_strided(
                        arrmeta, data, child_stride, child_size);
    }
}

size_t strided_dim_type::make_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx) const
{
  if (this == dst_tp.extended()) {
    const strided_dim_type_arrmeta *dst_md =
        reinterpret_cast<const strided_dim_type_arrmeta *>(dst_arrmeta);
    intptr_t src_size, src_stride;
    ndt::type src_el_tp;
    const char *src_el_arrmeta;

    if (src_tp.get_ndim() < dst_tp.get_ndim()) {
      kernels::strided_assign_ck *self =
          kernels::strided_assign_ck::create(ckb, kernreq, ckb_offset);
      self->m_size = dst_md->dim_size;
      self->m_dst_stride = dst_md->stride;
      // If the src has fewer dimensions, broadcast it across this one
      self->m_src_stride = 0;
      return ::make_assignment_kernel(
          ckb, ckb_offset, m_element_tp,
          dst_arrmeta + sizeof(strided_dim_type_arrmeta), src_tp, src_arrmeta,
          kernel_request_strided, ectx);
    } else if (src_tp.get_as_strided(src_arrmeta, &src_size, &src_stride,
                                         &src_el_tp, &src_el_arrmeta)) {
      kernels::strided_assign_ck *self =
          kernels::strided_assign_ck::create(ckb, kernreq, ckb_offset);
      self->m_size = dst_md->dim_size;
      self->m_dst_stride = dst_md->stride;
      self->m_src_stride = src_stride;
      // Check for a broadcasting error
      if (src_size != 1 && dst_md->dim_size != src_size) {
        throw broadcast_error(dst_tp, dst_arrmeta, src_tp, src_arrmeta);
      }

      return ::make_assignment_kernel(
          ckb, ckb_offset, m_element_tp,
          dst_arrmeta + sizeof(strided_dim_type_arrmeta), src_el_tp,
          src_el_arrmeta, kernel_request_strided, ectx);
    } else if (!src_tp.is_builtin()) {
      // Give the src type a chance to make a kernel
      return src_tp.extended()->make_assignment_kernel(
          ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
          kernreq, ectx);
    } else {
      stringstream ss;
      ss << "Cannot assign from " << src_tp << " to " << dst_tp;
      throw dynd::type_error(ss.str());
    }
  } else if (dst_tp.get_kind() == string_kind) {
    return make_any_to_string_assignment_kernel(ckb, ckb_offset, dst_tp,
                                                dst_arrmeta, src_tp,
                                                src_arrmeta, kernreq, ectx);
  } else if (dst_tp.get_ndim() < src_tp.get_ndim()) {
    throw broadcast_error(dst_tp, dst_arrmeta, src_tp, src_arrmeta);
  } else {
    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw dynd::type_error(ss.str());
  }
}

void strided_dim_type::foreach_leading(const char *arrmeta, char *data,
                                       foreach_fn_t callback,
                                       void *callback_data) const
{
    const strided_dim_type_arrmeta *md = reinterpret_cast<const strided_dim_type_arrmeta *>(arrmeta);
    const char *child_arrmeta = arrmeta + sizeof(strided_dim_type_arrmeta);
    intptr_t stride = md->stride;
    for (intptr_t i = 0, i_end = md->dim_size; i < i_end; ++i, data += stride) {
        callback(m_element_tp, child_arrmeta, data, callback_data);
    }
}

void strided_dim_type::reorder_default_constructed_strides(char *dst_arrmeta,
                const ndt::type& src_tp, const char *src_arrmeta) const
{
    if (m_element_tp.get_type_id() != strided_dim_type_id) {
        // Nothing to do if there's just one reorderable dimension
        return;
    }

    if (get_ndim() > src_tp.get_ndim()) {
        // If the destination has more dimensions than the source,
        // do the reordering starting from where they match, to
        // follow the broadcasting rules.
        if (m_element_tp.get_type_id() == strided_dim_type_id) {
            const strided_dim_type *sdd = m_element_tp.tcast<strided_dim_type>();
            sdd->reorder_default_constructed_strides(
                            dst_arrmeta + sizeof(strided_dim_type_arrmeta),
                            src_tp, src_arrmeta);
        }
        return;
    }

    // Find the total number of dimensions we might be reordering, then process
    // them all at once. This code handles a whole chain of strided_dim_type
    // instances at once.
    size_t ndim = 1;
    ndt::type last_dt = m_element_tp;
    do {
        ++ndim;
        last_dt = last_dt.tcast<strided_dim_type>()->get_element_type();
    } while (last_dt.get_type_id() == strided_dim_type_id);

    dimvector strides(ndim);
    ndt::type last_src_tp = src_tp;
    intptr_t previous_stride = 0;
    size_t ndim_partial = 0;
    // Get representative strides from all the strided source dimensions
    bool c_order = true;
    for (size_t i = 0; i < ndim; ++i) {
        intptr_t stride;
        switch (last_src_tp.get_type_id()) {
            case cfixed_dim_type_id: {
                const cfixed_dim_type *fdd = last_src_tp.tcast<cfixed_dim_type>();
                stride = fdd->get_fixed_stride();
                last_src_tp = fdd->get_element_type();
                src_arrmeta += sizeof(cfixed_dim_type_arrmeta);
                break;
            }
            case strided_dim_type_id: {
                const strided_dim_type *sdd = last_src_tp.tcast<strided_dim_type>();
                const strided_dim_type_arrmeta *md = reinterpret_cast<const strided_dim_type_arrmeta *>(src_arrmeta);
                stride = md->stride;
                last_src_tp = sdd->get_element_type();
                src_arrmeta += sizeof(strided_dim_type_arrmeta);
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
        strided_dim_type_arrmeta *md =
                        reinterpret_cast<strided_dim_type_arrmeta *>(dst_arrmeta);
        intptr_t stride = md[ndim_partial-1].stride;
        if (stride == 0) {
            // Because of the rule that size one dimensions have
            // zero stride, may have to look further
            intptr_t i = ndim_partial-2;
            do {
                stride = md[i].stride;
            } while (stride == 0 && i >= 0);
        }
        for (size_t i = 0; i < ndim_partial; ++i) {
            int i_perm = axis_perm[i];
            strided_dim_type_arrmeta& i_md = md[i_perm];
            intptr_t dim_size = i_md.dim_size;
            i_md.stride = dim_size > 1 ? stride : 0;
            stride *= dim_size;
        }
    }

    // If that didn't cover all the dimensions, then get the
    // axis order classification to handle the rest
    if (ndim_partial < ndim && !last_src_tp.is_builtin()) {
        axis_order_classification_t aoc = last_src_tp.extended()->classify_axis_order(src_arrmeta);
        // TODO: Allow user control by adding a "default axis order" to the evaluation context
        if (aoc == axis_order_f) {
            // If it's F-order, reverse the ordering of the strides
            strided_dim_type_arrmeta *md =
                            reinterpret_cast<strided_dim_type_arrmeta *>(dst_arrmeta);
            intptr_t stride = md[ndim-1].stride;
            if (stride == 0) {
                // Because of the rule that size one dimensions have
                // zero stride, may have to look further
                intptr_t i = ndim-2;
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

static ndt::type get_element_type(const ndt::type& dt) {
    return dt.tcast<strided_dim_type>()->get_element_type();
}

void strided_dim_type::get_dynamic_type_properties(
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    static pair<string, gfunc::callable> strided_dim_type_properties[] = {
        pair<string, gfunc::callable>(
            "element_type", gfunc::make_callable(&::get_element_type, "self"))};

    *out_properties = strided_dim_type_properties;
    *out_count = sizeof(strided_dim_type_properties) / sizeof(strided_dim_type_properties[0]);
}

void strided_dim_type::get_dynamic_array_properties(
    const std::pair<std::string, gfunc::callable> **out_properties,
    size_t *out_count) const
{
    if (m_element_tp.is_builtin()) {
        get_builtin_type_dynamic_array_properties(m_element_tp.get_type_id(),
                                                  out_properties, out_count);
    } else {
        m_element_tp.extended()->get_dynamic_array_properties(out_properties, out_count);
    }
}

void strided_dim_type::get_dynamic_array_functions(
    const std::pair<std::string, gfunc::callable> **out_functions,
    size_t *out_count) const
{
    if (m_element_tp.is_builtin()) {
        // TODO
    } else {
        m_element_tp.extended()->get_dynamic_array_functions(out_functions, out_count);
    }
}

namespace {
    // TODO: use the PP meta stuff, but DYND_PP_LEN_MAX is set to 8 right now,
    // would need to be 19
    struct static_strided_dims {
        strided_dim_type bt1;
        strided_dim_type bt2;
        strided_dim_type bt3;
        strided_dim_type bt4;
        strided_dim_type bt5;
        strided_dim_type bt6;
        strided_dim_type bt7;
        strided_dim_type bt8;
        strided_dim_type bt9;
        strided_dim_type bt10;
        strided_dim_type bt11;
        strided_dim_type bt12;
        strided_dim_type bt13;
        strided_dim_type bt14;
        strided_dim_type bt15;
        strided_dim_type bt16;
        strided_dim_type bt17;
        strided_dim_type bt18;

        ndt::type static_builtins_instance[builtin_type_id_count];

        static_strided_dims()
            : bt1(ndt::type((type_id_t)1)),
              bt2(ndt::type((type_id_t)2)),
              bt3(ndt::type((type_id_t)3)),
              bt4(ndt::type((type_id_t)4)),
              bt5(ndt::type((type_id_t)5)),
              bt6(ndt::type((type_id_t)6)),
              bt7(ndt::type((type_id_t)7)),
              bt8(ndt::type((type_id_t)8)),
              bt9(ndt::type((type_id_t)9)),
              bt10(ndt::type((type_id_t)10)),
              bt11(ndt::type((type_id_t)11)),
              bt12(ndt::type((type_id_t)12)),
              bt13(ndt::type((type_id_t)13)),
              bt14(ndt::type((type_id_t)14)),
              bt15(ndt::type((type_id_t)15)),
              bt16(ndt::type((type_id_t)16)),
              bt17(ndt::type((type_id_t)17)),
              bt18(ndt::type((type_id_t)18))
        {
            static_builtins_instance[1] = ndt::type(&bt1, true);
            static_builtins_instance[2] = ndt::type(&bt2, true);
            static_builtins_instance[3] = ndt::type(&bt3, true);
            static_builtins_instance[4] = ndt::type(&bt4, true);
            static_builtins_instance[5] = ndt::type(&bt5, true);
            static_builtins_instance[6] = ndt::type(&bt6, true);
            static_builtins_instance[7] = ndt::type(&bt7, true);
            static_builtins_instance[8] = ndt::type(&bt8, true);
            static_builtins_instance[9] = ndt::type(&bt9, true);
            static_builtins_instance[10] = ndt::type(&bt10, true);
            static_builtins_instance[11] = ndt::type(&bt11, true);
            static_builtins_instance[12] = ndt::type(&bt12, true);
            static_builtins_instance[13] = ndt::type(&bt13, true);
            static_builtins_instance[14] = ndt::type(&bt14, true);
            static_builtins_instance[15] = ndt::type(&bt15, true);
            static_builtins_instance[16] = ndt::type(&bt16, true);
            static_builtins_instance[17] = ndt::type(&bt17, true);
            static_builtins_instance[18] = ndt::type(&bt18, true);
        }
    };
} // anonymous namespace

ndt::type ndt::make_strided_dim(const ndt::type& element_tp)
{
    // Static instances of the types, which have a reference
    // count > 0 for the lifetime of the program. This static
    // construction is inside a function to ensure correct creation
    // order during startup.
    static static_strided_dims ssd;

    if (element_tp.is_builtin()) {
        return ssd.static_builtins_instance[element_tp.get_type_id()];
    } else {
        return ndt::type(new strided_dim_type(element_tp), false);
    }
}

