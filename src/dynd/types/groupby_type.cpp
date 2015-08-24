//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <vector>

#include <dynd/types/groupby_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/c_contiguous_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/categorical_type.hpp>
#include <dynd/gfunc/make_callable.hpp>

using namespace std;
using namespace dynd;

ndt::groupby_type::groupby_type(const type &data_values_tp,
                                const type &by_values_tp)
    : base_expr_type(groupby_type_id, expr_kind, sizeof(groupby_type_data),
                     sizeof(void *), type_flag_none, 0,
                     1 + data_values_tp.get_ndim())
{
  m_groups_type = by_values_tp.at_single(0).value_type();
  if (m_groups_type.get_type_id() != categorical_type_id) {
    stringstream ss;
    ss << "to construct a groupby type, the by type, "
       << by_values_tp.at_single(0);
    ss << ", must have a categorical value type";
    throw runtime_error(ss.str());
  }
  if (data_values_tp.get_ndim() < 1) {
    throw runtime_error("to construct a groupby type, its values type must "
                        "have at least one array dimension");
  }
  if (by_values_tp.get_ndim() < 1) {
    throw runtime_error("to construct a groupby type, its values type must "
                        "have at least one array dimension");
  }
  m_operand_type =
      struct_type::make({"data", "by"}, {pointer_type::make(data_values_tp),
                                         pointer_type::make(by_values_tp)});
  m_members.arrmeta_size = m_operand_type.get_arrmeta_size();
  const categorical_type *cd = m_groups_type.extended<categorical_type>();
  m_value_type =
      make_fixed_dim(cd->get_category_count(),
                     var_dim_type::make(data_values_tp.at_single(0)));
  m_members.flags =
      inherited_flags(m_value_type.get_flags(), m_operand_type.get_flags());
}

ndt::groupby_type::~groupby_type()
{
}

void ndt::groupby_type::print_data(std::ostream &DYND_UNUSED(o),
                                   const char *DYND_UNUSED(arrmeta),
                                   const char *DYND_UNUSED(data)) const
{
  throw runtime_error(
      "internal error: groupby_type::print_data isn't supposed to be called");
}

ndt::type ndt::groupby_type::get_data_values_type() const
{
  const pointer_type *pd =
      static_cast<const pointer_type *>(m_operand_type.at_single(0).extended());
  return pd->get_target_type();
}

ndt::type ndt::groupby_type::get_by_values_type() const
{
  const pointer_type *pd =
      static_cast<const pointer_type *>(m_operand_type.at_single(1).extended());
  return pd->get_target_type();
}

void ndt::groupby_type::print_type(std::ostream &o) const
{
  o << "groupby<values=" << get_data_values_type();
  o << ", by=" << get_by_values_type() << ">";
}

void ndt::groupby_type::get_shape(intptr_t ndim, intptr_t i,
                                  intptr_t *out_shape, const char *arrmeta,
                                  const char *DYND_UNUSED(data)) const
{
  // The first dimension is the groups, the second variable-sized
  out_shape[i] = reinterpret_cast<const categorical_type *>(
      m_groups_type.extended())->get_category_count();
  if (i + 1 < ndim) {
    out_shape[i + 1] = -1;
  }

  // Get the rest of the shape if necessary
  if (i + 2 < ndim) {
    // Get the type for a single data_value element, and its corresponding
    // arrmeta
    type data_values_tp =
        m_operand_type.at_single(0, arrmeta ? &arrmeta : NULL);
    data_values_tp = data_values_tp.at_single(0, arrmeta ? &arrmeta : NULL);
    // Use this to get the rest of the shape
    data_values_tp.extended()->get_shape(ndim, i + 2, out_shape, arrmeta, NULL);
  }
}

bool ndt::groupby_type::is_lossless_assignment(const type &dst_tp,
                                               const type &src_tp) const
{
  // Treat this type as the value type for whether assignment is always lossless
  if (src_tp.extended() == this) {
    return ::dynd::is_lossless_assignment(dst_tp, m_value_type);
  } else {
    return ::dynd::is_lossless_assignment(m_value_type, src_tp);
  }
}

bool ndt::groupby_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != groupby_type_id) {
    return false;
  } else {
    const groupby_type *dt = static_cast<const groupby_type *>(&rhs);
    return m_value_type == dt->m_value_type &&
           m_operand_type == dt->m_operand_type;
  }
}

namespace {
// Assign from a categorical type to some other type
template <typename UIntType>
struct groupby_to_value_assign_kernel
    : nd::base_kernel<groupby_to_value_assign_kernel<UIntType>, 1> {
  typedef groupby_to_value_assign_kernel extra_type;

  // The groupby type
  const ndt::groupby_type *src_groupby_tp;
  const char *dst_arrmeta, *src_arrmeta;

  groupby_to_value_assign_kernel(const ndt::groupby_type *src_groupby_tp,
                                 const char *dst_arrmeta,
                                 const char *src_arrmeta)
      : src_groupby_tp(src_groupby_tp), dst_arrmeta(dst_arrmeta),
        src_arrmeta(src_arrmeta)
  {
    // The kernel type owns a reference to this type
    base_type_incref(this->src_groupby_tp);
  }

  void single(char *dst, char *const *src)
  {
    const ndt::groupby_type *gd = src_groupby_tp;

    // Get the data_values raw nd::array
    ndt::type data_values_tp = gd->get_operand_type();
    const char *data_values_arrmeta = src_arrmeta;
    char *data_values_data = src[0];
    data_values_tp = data_values_tp.extended()->at_single(
        0, &data_values_arrmeta, const_cast<const char **>(&data_values_data));
    data_values_tp =
        data_values_tp.extended<ndt::pointer_type>()->get_target_type();
    data_values_arrmeta += sizeof(pointer_type_arrmeta);
    data_values_data = *reinterpret_cast<char **>(data_values_data);

    // Get the by_values raw nd::array
    ndt::type by_values_tp = gd->get_operand_type();
    const char *by_values_arrmeta = src_arrmeta;
    char *by_values_data = src[0];
    by_values_tp = by_values_tp.extended()->at_single(
        1, &by_values_arrmeta, const_cast<const char **>(&by_values_data));
    by_values_tp =
        by_values_tp.extended<ndt::pointer_type>()->get_target_type();
    by_values_arrmeta += sizeof(pointer_type_arrmeta);
    by_values_data = *reinterpret_cast<char **>(by_values_data);

    // If by_values is an expression, evaluate it since we're doing two
    // passes through it
    nd::array by_values_tmp;
    if (by_values_tp.is_expression()) {
      by_values_tmp =
          nd::eval_raw_copy(by_values_tp, by_values_arrmeta, by_values_data);
      by_values_tp = by_values_tmp.get_type();
      by_values_arrmeta = by_values_tmp.get_arrmeta();
      by_values_data =
          const_cast<char *>(by_values_tmp.get_readonly_originptr());
    }

    // Get a strided representation of by_values for processing
    intptr_t by_values_stride, by_values_size;
    if (!by_values_tp.get_as_strided(by_values_arrmeta, &by_values_size,
                                     &by_values_stride, &by_values_tp,
                                     &by_values_arrmeta)) {
      throw runtime_error(
          "groupby: failed to get by_values as a strided array");
    }

    const ndt::type &result_tp = gd->get_value_type();
    const ndt::fixed_dim_type *fad = result_tp.extended<ndt::fixed_dim_type>();
    intptr_t fad_stride =
        reinterpret_cast<const size_stride_t *>(dst_arrmeta)->stride;
    const ndt::var_dim_type *vad = static_cast<const ndt::var_dim_type *>(
        fad->get_element_type().extended());
    const var_dim_type_arrmeta *vad_md =
        reinterpret_cast<const var_dim_type_arrmeta *>(
            dst_arrmeta + sizeof(fixed_dim_type_arrmeta));
    if (vad_md->offset != 0) {
      throw runtime_error("dynd groupby: destination var_dim offset "
                          "must be zero to allocate output");
    }
    intptr_t vad_stride = vad_md->stride;

    // Do a pass through by_values to get the size of each variable-sized
    // dimension
    vector<size_t> cat_sizes(fad->get_fixed_dim_size());
    const char *by_values_ptr = by_values_data;
    for (intptr_t i = 0; i < by_values_size;
         ++i, by_values_ptr += by_values_stride) {
      UIntType value = *reinterpret_cast<const UIntType *>(by_values_ptr);
      if (value >= cat_sizes.size()) {
        stringstream ss;
        ss << "dynd groupby: 'by' array contains an out of bounds value "
           << (uint32_t)value;
        ss << ", range is [0, " << cat_sizes.size() << ")";
        throw runtime_error(ss.str());
      }
      ++cat_sizes[value];
    }

    // Allocate the output, and create a vector of pointers to the start
    // of each group's output
    memory_block_pod_allocator_api *allocator =
        get_memory_block_pod_allocator_api(vad_md->blockref);
    char *out_begin = NULL, *out_end = NULL;
    allocator->allocate(vad_md->blockref, by_values_size * vad_stride,
                        vad->get_element_type().get_data_alignment(),
                        &out_begin, &out_end);
    vector<char *> cat_pointers(cat_sizes.size());
    for (size_t i = 0, i_end = cat_pointers.size(); i != i_end; ++i) {
      cat_pointers[i] = out_begin;
      reinterpret_cast<var_dim_type_data *>(dst + i *fad_stride)->begin =
          out_begin;
      size_t csize = cat_sizes[i];
      reinterpret_cast<var_dim_type_data *>(dst + i *fad_stride)->size = csize;
      out_begin += csize * vad_stride;
    }

    // Loop through both by_values and data_values,
    // copying the data to the right place in the output
    ckernel_prefix *echild = this->get_child_ckernel();
    expr_single_t opchild = echild->get_function<expr_single_t>();
    intptr_t dvit_dim_size, dvit_stride;
    ndt::type dvit_el_tp;
    const char *dvit_el_arrmeta;
    if (!data_values_tp.get_as_strided(data_values_arrmeta, &dvit_dim_size,
                                       &dvit_stride, &dvit_el_tp,
                                       &dvit_el_arrmeta)) {
      throw runtime_error(
          "groupby: failed to get data_values as a strided array");
    }
    char *dvit_data = data_values_data;
    by_values_ptr = by_values_data;
    for (intptr_t i = 0; i < dvit_dim_size; ++i) {
      UIntType value = *reinterpret_cast<const UIntType *>(by_values_ptr);
      char *&cp = cat_pointers[value];
      opchild(echild, cp, &dvit_data);
      // Advance the pointer inside the cat_pointers array
      cp += vad_stride;
      by_values_ptr += by_values_stride;
      dvit_data += dvit_stride;
    }
  }

  static void destruct(ckernel_prefix *self)
  {
    extra_type *e = reinterpret_cast<extra_type *>(self);
    if (e->src_groupby_tp != NULL) {
      base_type_decref(e->src_groupby_tp);
    }
    self->get_child_ckernel(sizeof(extra_type))->destroy();
  }
};
} // anonymous namespace

size_t ndt::groupby_type::make_operand_to_value_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
    const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx) const
{
  const categorical_type *cd = m_groups_type.extended<categorical_type>();
  switch (cd->get_storage_type().get_type_id()) {
  case uint8_type_id: {
    groupby_to_value_assign_kernel<uint8_t>::make(
        ckb, kernreq, ckb_offset, this, dst_arrmeta, src_arrmeta);
  } break;
  case uint16_type_id: {
    groupby_to_value_assign_kernel<uint16_t>::make(
        ckb, kernreq, ckb_offset, this, dst_arrmeta, src_arrmeta);
  } break;
  case uint32_type_id: {
    groupby_to_value_assign_kernel<uint32_t>::make(
        ckb, kernreq, ckb_offset, this, dst_arrmeta, src_arrmeta);
  } break;
  default:
    throw runtime_error(
        "internal error in groupby_type::get_operand_to_value_kernel");
  }

  // The following is the setup for copying a single 'data' value to the output
  // The destination element type and arrmeta
  const type &dst_element_tp = static_cast<const var_dim_type *>(
      m_value_type.extended<fixed_dim_type>()->get_element_type().extended())
                                   ->get_element_type();
  const char *dst_element_arrmeta = dst_arrmeta +
                                    sizeof(fixed_dim_type_arrmeta) +
                                    sizeof(var_dim_type_arrmeta);
  // Get source element type and arrmeta
  type src_element_tp = m_operand_type;
  const char *src_element_arrmeta = src_arrmeta;
  src_element_tp =
      src_element_tp.extended()->at_single(0, &src_element_arrmeta, NULL);
  src_element_tp = src_element_tp.extended<pointer_type>()->get_target_type();
  src_element_arrmeta += sizeof(pointer_type_arrmeta);
  src_element_tp =
      src_element_tp.extended()->at_single(0, &src_element_arrmeta, NULL);

  return ::make_assignment_kernel(
      ckb, ckb_offset, dst_element_tp, dst_element_arrmeta, src_element_tp,
      src_element_arrmeta, kernel_request_single, ectx);
}

size_t ndt::groupby_type::make_value_to_operand_assignment_kernel(
    void *DYND_UNUSED(ckb), intptr_t DYND_UNUSED(ckb_offset),
    const char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    kernel_request_t DYND_UNUSED(kernreq),
    const eval::eval_context *DYND_UNUSED(ectx)) const
{
  throw runtime_error("Cannot assign to a dynd groupby object value");
}

ndt::type ndt::groupby_type::with_replaced_storage_type(
    const type &DYND_UNUSED(replacement_type)) const
{
  throw runtime_error(
      "TODO: implement groupby_type::with_replaced_storage_type");
}

///////// properties on the nd::array

static nd::array property_ndo_get_groups(const nd::array &n)
{
  ndt::type d = n.get_type();
  while (d.get_type_id() != groupby_type_id) {
    d = d.at_single(0);
  }
  const ndt::groupby_type *gd = d.extended<ndt::groupby_type>();
  return gd->get_groups_type().p("categories");
}

void ndt::groupby_type::get_dynamic_array_properties(
    const std::pair<std::string, gfunc::callable> **out_properties,
    size_t *out_count) const
{
  static pair<string, gfunc::callable> groupby_array_properties[] = {
      pair<string, gfunc::callable>(
          "groups", gfunc::make_callable(&property_ndo_get_groups, "self")), };

  *out_properties = groupby_array_properties;
  *out_count =
      sizeof(groupby_array_properties) / sizeof(groupby_array_properties[0]);
}
