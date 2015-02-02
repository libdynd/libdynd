//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/shortvector.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/func/make_callable.hpp>

using namespace std;
using namespace dynd;

cfixed_dim_type::cfixed_dim_type(size_t dimension_size, const ndt::type& element_tp)
    : base_dim_type(cfixed_dim_type_id, element_tp, 0, element_tp.get_data_alignment(),
                    sizeof(cfixed_dim_type_arrmeta), type_flag_none, true),
            m_dim_size(dimension_size)
{
  size_t child_element_size = element_tp.get_data_size();
  if (child_element_size == 0) {
    stringstream ss;
    ss << "Cannot create dynd cfixed_dim type with element type " << element_tp;
    ss << ", as it does not have a fixed size";
    throw dynd::type_error(ss.str());
  }
  m_stride = m_dim_size > 1 ? element_tp.get_data_size() : 0;
  m_members.data_size = m_stride * (m_dim_size - 1) + child_element_size;
  // Propagate the inherited flags from the element
  m_members.flags |=
      (element_tp.get_flags() &
       ((type_flags_operand_inherited | type_flags_value_inherited) &
        ~type_flag_scalar));

  // Copy nd::array properties and functions from the first non-array dimension
  get_scalar_properties_and_functions(m_array_properties, m_array_functions);
}

cfixed_dim_type::cfixed_dim_type(size_t dimension_size,
                                 const ndt::type &element_tp, intptr_t stride)
    : base_dim_type(cfixed_dim_type_id, element_tp, 0,
                            element_tp.get_data_alignment(), 0, type_flag_none,
                            true),
      m_stride(stride), m_dim_size(dimension_size)
{
    size_t child_element_size = element_tp.get_data_size();
    if (child_element_size == 0) {
        stringstream ss;
        ss << "Cannot create dynd cfixed_dim type with element type " << element_tp;
        ss << ", as it does not have a fixed size";
        throw dynd::type_error(ss.str());
    }
    if (dimension_size <= 1 && stride != 0) {
        stringstream ss;
        ss << "Cannot create dynd cfixed_dim type with size " << dimension_size;
        ss << " and stride " << stride << ", as the stride must be zero when the dimension size is 1";
        throw dynd::type_error(ss.str());
    }
    if (dimension_size > 1 && stride == 0) {
        stringstream ss;
        ss << "Cannot create dynd cfixed_dim type with size " << dimension_size;
        ss << " and stride 0, as the stride must be non-zero when the dimension size is > 1";
        throw dynd::type_error(ss.str());
    }
    m_members.data_size = m_stride * (m_dim_size-1) + child_element_size;
    // Propagate the zeroinit flag from the element
    m_members.flags |= (element_tp.get_flags()&type_flag_zeroinit);

    // Copy nd::array properties and functions from the first non-array dimension
    get_scalar_properties_and_functions(m_array_properties, m_array_functions);
}

cfixed_dim_type::~cfixed_dim_type()
{
}

void cfixed_dim_type::print_data(std::ostream &o, const char *arrmeta,
                                 const char *data) const
{
  strided_array_summarized(o, get_element_type(),
                           arrmeta + sizeof(cfixed_dim_type_arrmeta), data,
                           m_dim_size, m_stride);
}

void cfixed_dim_type::print_type(std::ostream& o) const
{
    o << "cfixed[";
    o << m_dim_size;
    if ((size_t)m_stride != m_element_tp.get_data_size() &&
            m_dim_size != 1) {
        o << ", stride=" << m_stride;
    }
    o << "] * " << m_element_tp;
}

bool cfixed_dim_type::is_expression() const
{
  return m_element_tp.is_expression();
}

bool cfixed_dim_type::is_unique_data_owner(const char *arrmeta) const
{
  if (m_element_tp.is_builtin()) {
    return true;
  } else {
    return m_element_tp.extended()->is_unique_data_owner(
        arrmeta + sizeof(cfixed_dim_type_arrmeta));
  }
}

void cfixed_dim_type::transform_child_types(type_transform_fn_t transform_fn,
                                            intptr_t arrmeta_offset,
                                            void *extra,
                                            ndt::type &out_transformed_tp,
                                            bool &out_was_transformed) const
{
    ndt::type tmp_tp;
    bool was_transformed = false;
    transform_fn(m_element_tp, arrmeta_offset + sizeof(cfixed_dim_type_arrmeta),
                 extra, tmp_tp, was_transformed);
    if (was_transformed) {
        if (tmp_tp.get_data_size() != 0) {
          out_transformed_tp =
              ndt::type(new cfixed_dim_type(m_dim_size, tmp_tp), false);
        } else {
          out_transformed_tp =
              ndt::type(new fixed_dim_type(m_dim_size, tmp_tp), false);
        }
        out_was_transformed = true;
    } else {
        out_transformed_tp = ndt::type(this, true);
    }
}


ndt::type cfixed_dim_type::get_canonical_type() const
{
    ndt::type canonical_element_dt = m_element_tp.get_canonical_type();
    // The transformed type may no longer have a fixed size, so check whether
    // we have to switch to the more flexible fixed_dim_type
    if (canonical_element_dt.get_data_size() != 0) {
        return ndt::type(new cfixed_dim_type(m_dim_size, canonical_element_dt), false);
    } else {
        return ndt::type(new fixed_dim_type(m_dim_size, canonical_element_dt), false);
    }
}

ndt::type cfixed_dim_type::apply_linear_index(intptr_t nindices,
                                              const irange *indices,
                                              size_t current_i,
                                              const ndt::type &root_tp,
                                              bool leading_dimension) const
{
  if (nindices == 0) {
    return ndt::type(this, true);
  }
  else {
    bool remove_dimension;
    intptr_t start_index, index_stride, dimension_size;
    apply_single_linear_index(*indices, m_dim_size, current_i, &root_tp,
                              remove_dimension, start_index, index_stride,
                              dimension_size);
    if (remove_dimension) {
      return m_element_tp.apply_linear_index(
          nindices - 1, indices + 1, current_i + 1, root_tp, leading_dimension);
    }
    else {
      return ndt::make_fixed_dim(
          dimension_size,
          m_element_tp.apply_linear_index(nindices - 1, indices + 1,
                                          current_i + 1, root_tp, false));
    }
  }
}

intptr_t cfixed_dim_type::apply_linear_index(
    intptr_t nindices, const irange *indices, const char *arrmeta,
    const ndt::type &result_tp, char *out_arrmeta,
    memory_block_data *embedded_reference, size_t current_i,
    const ndt::type &root_tp, bool leading_dimension, char **inout_data,
    memory_block_data **inout_dataref) const
{
    if (nindices == 0) {
        // If there are no more indices, copy the arrmeta verbatim
        arrmeta_copy_construct(out_arrmeta, arrmeta, embedded_reference);
        return 0;
    } else {
        bool remove_dimension;
        intptr_t start_index, index_stride, dimension_size;
        apply_single_linear_index(*indices, m_dim_size, current_i, &root_tp,
                        remove_dimension, start_index, index_stride, dimension_size);
        if (remove_dimension) {
            // Apply the strided offset and continue applying the index
            intptr_t offset = m_stride * start_index;
            if (!m_element_tp.is_builtin()) {
                if (leading_dimension) {
                    // In the case of a leading dimension, first bake the offset into
                    // the data pointer, so that it's pointing at the right element
                    // for the collapsing of leading dimensions to work correctly.
                    *inout_data += offset;
                    offset = m_element_tp.extended()->apply_linear_index(
                        nindices - 1, indices + 1,
                        arrmeta + sizeof(cfixed_dim_type_arrmeta), result_tp,
                        out_arrmeta, embedded_reference, current_i + 1, root_tp,
                        true, inout_data, inout_dataref);
                } else {
                  offset += m_element_tp.extended()->apply_linear_index(
                      nindices - 1, indices + 1,
                      arrmeta + sizeof(cfixed_dim_type_arrmeta), result_tp,
                      out_arrmeta, embedded_reference, current_i + 1, root_tp,
                      false, NULL, NULL);
                }
            }
            return offset;
        } else {
            fixed_dim_type_arrmeta *out_md = reinterpret_cast<fixed_dim_type_arrmeta *>(out_arrmeta);
            // Produce the new offset data, stride, and size for the resulting array,
            // which is now a fixed_dim instead of a cfixed_dim
            intptr_t offset = m_stride * start_index;
            out_md->stride = m_stride * index_stride;
            out_md->dim_size = dimension_size;
            if (!m_element_tp.is_builtin()) {
                const fixed_dim_type *result_etp = result_tp.extended<fixed_dim_type>();
                offset += m_element_tp.extended()->apply_linear_index(
                    nindices - 1, indices + 1,
                    arrmeta + sizeof(cfixed_dim_type_arrmeta),
                    result_etp->get_element_type(),
                    out_arrmeta + sizeof(fixed_dim_type_arrmeta),
                    embedded_reference, current_i + 1, root_tp, false, NULL,
                    NULL);
            }
            return offset;
        }
    }
}

ndt::type cfixed_dim_type::at_single(intptr_t i0, const char **inout_arrmeta,
                                     const char **inout_data) const
{
    // Bounds-checking of the index
    i0 = apply_single_index(i0, m_dim_size, NULL);
    if (inout_arrmeta) {
      *inout_arrmeta += sizeof(cfixed_dim_type_arrmeta);
    }
    // The cfixed_dim type has no arrmeta
    // If requested, modify the data
    if (inout_data) {
        *inout_data += i0 * m_stride;
    }
    return m_element_tp;
}

ndt::type cfixed_dim_type::get_type_at_dimension(char **inout_arrmeta,
                                                 intptr_t i,
                                                 intptr_t total_ndim) const
{
    if (i == 0) {
        return ndt::type(this, true);
    } else {
      if (inout_arrmeta) {
        *inout_arrmeta += sizeof(cfixed_dim_type_arrmeta);
      }
      return m_element_tp.get_type_at_dimension(inout_arrmeta, i - 1,
                                                total_ndim + 1);
    }
}

intptr_t cfixed_dim_type::get_dim_size(const char *DYND_UNUSED(arrmeta),
                                       const char *DYND_UNUSED(data)) const
{
    return m_dim_size;
}

void cfixed_dim_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                                const char *arrmeta, const char *data) const
{
  out_shape[i] = m_dim_size;

  // Process the later shape values
  if (i + 1 < ndim) {
    if (!m_element_tp.is_builtin()) {
      m_element_tp.extended()->get_shape(
          ndim, i + 1, out_shape, arrmeta + sizeof(cfixed_dim_type_arrmeta),
          (m_dim_size == 1) ? data : NULL);
    } else {
      stringstream ss;
      ss << "requested too many dimensions from type " << ndt::type(this, true);
      throw runtime_error(ss.str());
    }
  }
}

void cfixed_dim_type::get_strides(size_t i, intptr_t *out_strides,
                                  const char *arrmeta) const
{
  out_strides[i] = m_stride;

  // Process the later shape values
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->get_strides(
        i + 1, out_strides, arrmeta + sizeof(cfixed_dim_type_arrmeta));
  }
}

axis_order_classification_t
cfixed_dim_type::classify_axis_order(const char *arrmeta) const
{
  if (m_element_tp.get_ndim() > 0) {
    if (m_stride != 0) {
      // Call the helper function to do the classification
      return classify_strided_axis_order(
          m_stride, m_element_tp, arrmeta + sizeof(cfixed_dim_type_arrmeta));
    } else {
      // Use the classification of the element type
      return m_element_tp.extended()->classify_axis_order(
          arrmeta + sizeof(cfixed_dim_type_arrmeta));
    }
  } else {
    return axis_order_none;
  }
}

bool cfixed_dim_type::is_lossless_assignment(const ndt::type &dst_tp,
                                             const ndt::type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    } else if (src_tp.get_type_id() == cfixed_dim_type_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

bool cfixed_dim_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != cfixed_dim_type_id) {
    return false;
  } else {
    const cfixed_dim_type *dt = static_cast<const cfixed_dim_type *>(&rhs);
    return m_element_tp == dt->m_element_tp && m_dim_size == dt->m_dim_size &&
           m_stride == dt->m_stride;
  }
}

void cfixed_dim_type::arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const
{
  cfixed_dim_type_arrmeta *md =
      reinterpret_cast<cfixed_dim_type_arrmeta *>(arrmeta);
  md->dim_size = get_fixed_dim_size();
  md->stride = get_fixed_stride();
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_default_construct(
        arrmeta + sizeof(cfixed_dim_type_arrmeta), blockref_alloc);
  }
}

void cfixed_dim_type::arrmeta_copy_construct(
    char *dst_arrmeta, const char *src_arrmeta,
    memory_block_data *embedded_reference) const
{
  const cfixed_dim_type_arrmeta *src_md =
      reinterpret_cast<const cfixed_dim_type_arrmeta *>(src_arrmeta);
  cfixed_dim_type_arrmeta *dst_md =
      reinterpret_cast<cfixed_dim_type_arrmeta *>(dst_arrmeta);
  *dst_md = *src_md;
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_copy_construct(
        dst_arrmeta + sizeof(cfixed_dim_type_arrmeta),
        src_arrmeta + sizeof(cfixed_dim_type_arrmeta), embedded_reference);
  }
}

size_t cfixed_dim_type::arrmeta_copy_construct_onedim(
    char *dst_arrmeta, const char *src_arrmeta,
    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
  const cfixed_dim_type_arrmeta *src_md =
      reinterpret_cast<const cfixed_dim_type_arrmeta *>(src_arrmeta);
  cfixed_dim_type_arrmeta *dst_md =
      reinterpret_cast<cfixed_dim_type_arrmeta *>(dst_arrmeta);
  *dst_md = *src_md;
  return sizeof(cfixed_dim_type_arrmeta);
}

void cfixed_dim_type::arrmeta_reset_buffers(char *arrmeta) const
{
  if (m_element_tp.get_arrmeta_size() > 0) {
    m_element_tp.extended()->arrmeta_reset_buffers(arrmeta);
  }
}

void cfixed_dim_type::arrmeta_finalize_buffers(char *arrmeta) const
{
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_finalize_buffers(
        arrmeta + sizeof(cfixed_dim_type_arrmeta));
  }
}

void cfixed_dim_type::arrmeta_destruct(char *arrmeta) const
{
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_destruct(arrmeta +
                                              sizeof(cfixed_dim_type_arrmeta));
  }
}

void cfixed_dim_type::arrmeta_debug_print(const char *arrmeta, std::ostream &o,
                                         const std::string &indent) const
{
  const cfixed_dim_type_arrmeta *md =
      reinterpret_cast<const cfixed_dim_type_arrmeta *>(arrmeta);
  o << indent << "cfixed_dim arrmeta\n";
  o << indent << " size: " << md->dim_size;
  if (md->dim_size != get_fixed_dim_size()) {
    o << " INTERNAL INCONSISTENCY, type size: " << get_fixed_dim_size();
  }
  o << "\n";
  o << indent << " stride: " << md->stride;
  if (md->stride != get_fixed_stride()) {
    o << " INTERNAL INCONSISTENCY, type stride: " << get_fixed_stride();
  }
  o << "\n";
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_debug_print(
        arrmeta + sizeof(cfixed_dim_type_arrmeta), o, indent + " ");
  }
}

size_t cfixed_dim_type::get_iterdata_size(intptr_t ndim) const
{
    if (ndim == 0) {
        return 0;
    } else if (ndim == 1) {
        return sizeof(cfixed_dim_type_iterdata);
    } else {
        return m_element_tp.get_iterdata_size(ndim - 1) + sizeof(cfixed_dim_type_iterdata);
    }
}

// Does one iterator increment for this type
static char *iterdata_incr(iterdata_common *iterdata, intptr_t level)
{
    cfixed_dim_type_iterdata *id = reinterpret_cast<cfixed_dim_type_iterdata *>(iterdata);
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
    cfixed_dim_type_iterdata *id = reinterpret_cast<cfixed_dim_type_iterdata *>(iterdata);
    if (ndim == 1) {
        id->data = data;
        return data;
    } else {
        id->data = (id + 1)->common.reset(&(id + 1)->common, data, ndim - 1);
        return id->data;
    }
}

size_t cfixed_dim_type::iterdata_construct(iterdata_common *iterdata,
                                           const char **inout_arrmeta,
                                           intptr_t ndim, const intptr_t *shape,
                                           ndt::type &out_uniform_tp) const
{
    size_t inner_size = 0;
    if (ndim > 1) {
        *inout_arrmeta += sizeof(cfixed_dim_type_arrmeta);
        // Place any inner iterdata earlier than the outer iterdata
        inner_size = m_element_tp.extended()->iterdata_construct(iterdata, inout_arrmeta,
                        ndim - 1, shape + 1, out_uniform_tp);
        iterdata = reinterpret_cast<iterdata_common *>(reinterpret_cast<char *>(iterdata) + inner_size);
    } else {
        out_uniform_tp = m_element_tp;
    }

    if (m_dim_size != 1 && shape[0] != m_dim_size) {
        stringstream ss;
        ss << "Cannot construct dynd iterator of type " << ndt::type(this, true);
        ss << " with dimension size " << shape[0] << ", the size must be " << m_dim_size;
        throw runtime_error(ss.str());
    }

    cfixed_dim_type_iterdata *id = reinterpret_cast<cfixed_dim_type_iterdata *>(iterdata);

    id->common.incr = &iterdata_incr;
    id->common.reset = &iterdata_reset;
    id->data = NULL;
    id->stride = m_stride;

    return inner_size + sizeof(cfixed_dim_type_iterdata);
}

size_t cfixed_dim_type::iterdata_destruct(iterdata_common *iterdata, intptr_t ndim) const
{
    size_t inner_size = 0;
    if (ndim > 1) {
        inner_size = m_element_tp.extended()->iterdata_destruct(iterdata, ndim - 1);
    }
    // No dynamic data to free
    return inner_size + sizeof(cfixed_dim_type_iterdata);
}

void cfixed_dim_type::data_destruct(const char *arrmeta, char *data) const
{
  m_element_tp.extended()->data_destruct_strided(
      arrmeta + sizeof(cfixed_dim_type_arrmeta), data, m_stride, m_dim_size);
}

void cfixed_dim_type::data_destruct_strided(const char *arrmeta, char *data,
                                            intptr_t stride, size_t count) const
{
  intptr_t child_stride = m_stride;
  size_t child_size = m_dim_size;

  for (size_t i = 0; i != count; ++i, data += stride) {
    m_element_tp.extended()->data_destruct_strided(
        arrmeta + sizeof(cfixed_dim_type_arrmeta), data, child_stride,
        child_size);
  }
}

intptr_t cfixed_dim_type::make_assignment_kernel(
    const arrfunc_type_data *self, const arrfunc_type *af_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds) const
{
  if (this == dst_tp.extended()) {
    intptr_t src_size, src_stride;
    ndt::type src_el_tp;
    const char *src_el_arrmeta;

    if (src_tp.get_ndim() < dst_tp.get_ndim()) {
      kernels::strided_assign_ck *ck_self =
          kernels::strided_assign_ck::create(ckb, kernreq, ckb_offset);
      ck_self->m_size = get_fixed_dim_size();
      ck_self->m_dst_stride = get_fixed_stride();
      // If the src has fewer dimensions, broadcast it across this one
      ck_self->m_src_stride = 0;
      return ::make_assignment_kernel(
          self, af_tp, ckb, ckb_offset, m_element_tp,
          dst_arrmeta + sizeof(cfixed_dim_type_arrmeta), src_tp, src_arrmeta,
          kernel_request_strided, ectx, kwds);
    } else if (src_tp.get_as_strided(src_arrmeta, &src_size, &src_stride,
                                         &src_el_tp, &src_el_arrmeta)) {
      kernels::strided_assign_ck *ck_self =
          kernels::strided_assign_ck::create(ckb, kernreq, ckb_offset);
      ck_self->m_size = get_fixed_dim_size();
      ck_self->m_dst_stride = get_fixed_stride();
      ck_self->m_src_stride = src_stride;
      // Check for a broadcasting error
      if (src_size != 1 && get_fixed_dim_size() != src_size) {
        throw broadcast_error(dst_tp, dst_arrmeta, src_tp, src_arrmeta);
      }

      return ::make_assignment_kernel(
          self, af_tp, ckb, ckb_offset, m_element_tp,
          dst_arrmeta + sizeof(cfixed_dim_type_arrmeta), src_el_tp,
          src_el_arrmeta, kernel_request_strided, ectx, kwds);
    } else if (!src_tp.is_builtin()) {
      // Give the src type a chance to make a kernel
      return src_tp.extended()->make_assignment_kernel(
          self, af_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
          kernreq, ectx, kwds);
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

void cfixed_dim_type::foreach_leading(const char *arrmeta, char *data,
                                      foreach_fn_t callback,
                                      void *callback_data) const
{
  intptr_t stride = m_stride;
  for (intptr_t i = 0, i_end = m_dim_size; i < i_end; ++i, data += stride) {
    callback(m_element_tp, arrmeta + sizeof(cfixed_dim_type_arrmeta), data, callback_data);
  }
}

bool cfixed_dim_type::matches(const char *arrmeta, const ndt::type &other,
                             std::map<nd::string, ndt::type> &tp_vars) const
{
  switch (other.get_type_id()) {
  case fixed_dim_type_id:
    return get_fixed_dim_size() ==
               other.extended<fixed_dim_type>()->get_fixed_dim_size() &&
           m_element_tp.matches(
               arrmeta, other.extended<base_dim_type>()->get_element_type(),
               tp_vars);
  case cfixed_dim_type_id:
    return get_fixed_dim_size() ==
               other.extended<cfixed_dim_type>()->get_fixed_dim_size() &&
           get_fixed_stride() ==
               other.extended<cfixed_dim_type>()->get_fixed_stride() &&
           m_element_tp.matches(arrmeta,
                                other.extended<cfixed_dim_type>()->m_element_tp,
                                tp_vars);
  default:
    return false;
  }
}

ndt::type dynd::ndt::make_cfixed_dim(intptr_t ndim, const intptr_t *shape,
                const ndt::type& uniform_tp, const int *axis_perm)
{
    if (axis_perm == NULL) {
        // Build a C-order fixed array type
        ndt::type result = uniform_tp;
        for (ptrdiff_t i = (ptrdiff_t)ndim-1; i >= 0; --i) {
            result = ndt::make_cfixed_dim(shape[i], result);
        }
        return result;
    } else {
        // Create strides with the axis permutation
        dimvector strides(ndim);
        intptr_t stride = uniform_tp.get_data_size();
        for (intptr_t i = 0; i < ndim; ++i) {
            int i_perm = axis_perm[i];
            size_t dim_size = shape[i_perm];
            strides[i_perm] = dim_size > 1 ? stride : 0;
            stride *= dim_size;
        }
        // Build the fixed array type
        ndt::type result = uniform_tp;
        for (ptrdiff_t i = (ptrdiff_t)ndim-1; i >= 0; --i) {
            result = ndt::make_cfixed_dim(shape[i], result, strides[i]);
        }
        return result;
    }
}

static intptr_t get_fixed_dim_size(const ndt::type& dt) {
    return  dt.extended<cfixed_dim_type>()->get_fixed_dim_size();
}

static intptr_t get_fixed_dim_stride(const ndt::type& dt) {
    return dt.extended<cfixed_dim_type>()->get_fixed_stride();
}

static ndt::type get_element_type(const ndt::type& dt) {
    return dt.extended<cfixed_dim_type>()->get_element_type();
}

void cfixed_dim_type::get_dynamic_type_properties(
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    static pair<string, gfunc::callable> cfixed_dim_type_properties[] = {
        pair<string, gfunc::callable>(
            "fixed_dim_size",
            gfunc::make_callable(&::get_fixed_dim_size, "self")),
        pair<string, gfunc::callable>(
            "fixed_dim_stride",
            gfunc::make_callable(&get_fixed_dim_stride, "self")),
        pair<string, gfunc::callable>(
            "element_type", gfunc::make_callable(&::get_element_type, "self"))};

    *out_properties = cfixed_dim_type_properties;
    *out_count = sizeof(cfixed_dim_type_properties) / sizeof(cfixed_dim_type_properties[0]);
}

void cfixed_dim_type::get_dynamic_array_properties(
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    *out_properties = m_array_properties.empty() ? NULL : &m_array_properties[0];
    *out_count = (int)m_array_properties.size();
}

void cfixed_dim_type::get_dynamic_array_functions(
                const std::pair<std::string, gfunc::callable> **out_functions,
                size_t *out_count) const
{
    *out_functions = m_array_functions.empty() ? NULL : &m_array_functions[0];
    *out_count = (int)m_array_functions.size();
}
