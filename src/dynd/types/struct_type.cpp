//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/types/adapt_type.hpp>
#include <dynd/types/str_util.hpp>
#include <dynd/kernels/get_then_copy_kernel.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/kernels/tuple_assignment_kernels.hpp>
#include <dynd/kernels/struct_assignment_kernels.hpp>
#include <dynd/func/assignment.hpp>

using namespace std;
using namespace dynd;

ndt::struct_type::struct_type(const std::vector<std::string> &field_names, const std::vector<type> &field_types,
                              bool variadic)
    : tuple_type(struct_id, field_types, type_flag_none, true, variadic), m_field_names(field_names)
{
  /*
    if (!nd::ensure_immutable_contig<std::string>(m_field_names)) {
      stringstream ss;
      ss << "dynd struct field names requires an array of strings, got an "
            "array with type " << m_field_names.get_type();
      throw invalid_argument(ss.str());
    }
  */

  // Make sure that the number of names matches
  uintptr_t name_count = field_names.size();
  if (name_count != (uintptr_t)m_field_count) {
    stringstream ss;
    ss << "dynd struct type requires that the number of names, " << name_count << " matches the number of types, "
       << m_field_count;
    throw invalid_argument(ss.str());
  }

  this->kind = variadic ? kind_kind : struct_kind;

  create_array_properties();
}

ndt::struct_type::~struct_type() {}

intptr_t ndt::struct_type::get_field_index(const std::string &name) const
{
  auto it = std::find(m_field_names.begin(), m_field_names.end(), name);
  if (it != m_field_names.end()) {
    return it - m_field_names.begin();
  }

  return -1;
}

void ndt::struct_type::print_type(std::ostream &o) const
{
  // Use the record datashape syntax
  o << "{";
  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    if (i != 0) {
      o << ", ";
    }
    const std::string &name = m_field_names[i];
    if (is_simple_identifier_name(name)) {
      o << name;
    }
    else {
      print_escaped_utf8_string(o, name, true);
    }
    o << ": " << get_field_type(i);
  }
  if (m_variadic) {
    o << ", ...}";
  }
  else {
    o << "}";
  }
}

void ndt::struct_type::transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                                             type &out_transformed_tp, bool &out_was_transformed) const
{
  std::vector<type> tmp_field_types(m_field_count);
  bool was_transformed = false;

  for (intptr_t i = 0; i < m_field_count; ++i) {
    tmp_field_types[i] = make_type<type_type>();
  }

  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    transform_fn(get_field_type(i), arrmeta_offset + get_arrmeta_offset(i), extra, tmp_field_types[i], was_transformed);
  }
  if (was_transformed) {
    out_transformed_tp = struct_type::make(m_field_names, tmp_field_types, m_variadic);
    out_was_transformed = true;
  }
  else {
    out_transformed_tp = type(this, true);
  }
}

ndt::type ndt::struct_type::get_canonical_type() const
{
  std::vector<type> tmp_field_types(m_field_count);

  for (intptr_t i = 0; i < m_field_count; ++i) {
    tmp_field_types[i] = make_type<type_type>();
  }

  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    tmp_field_types[i] = get_field_type(i).get_canonical_type();
  }

  return struct_type::make(m_field_names, tmp_field_types, m_variadic);
}

ndt::type ndt::struct_type::at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const
{
  // Bounds-checking of the index
  i0 = apply_single_index(i0, m_field_count, NULL);
  if (inout_arrmeta) {
    char *arrmeta = const_cast<char *>(*inout_arrmeta);
    // Modify the arrmeta
    *inout_arrmeta += m_arrmeta_offsets[i0];
    // If requested, modify the data
    if (inout_data) {
      *inout_data += get_arrmeta_data_offsets(arrmeta)[i0];
    }
  }
  return get_field_type(i0);
}

bool ndt::struct_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    }
    else if (src_tp.get_id() == struct_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

bool ndt::struct_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_id() != struct_id) {
    return false;
  }
  else {
    const struct_type *dt = static_cast<const struct_type *>(&rhs);
    return get_data_alignment() == dt->get_data_alignment() && m_field_types == dt->m_field_types &&
           m_field_names == dt->m_field_names && m_variadic == dt->m_variadic;
  }
}

void ndt::struct_type::arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const
{
  const size_t *offsets = reinterpret_cast<const size_t *>(arrmeta);
  o << indent << "struct arrmeta\n";
  o << indent << " field offsets: ";
  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    o << offsets[i];
    if (i != i_end - 1) {
      o << ", ";
    }
  }
  o << "\n";
  for (intptr_t i = 0; i < m_field_count; ++i) {
    const type &field_dt = get_field_type(i);
    if (!field_dt.is_builtin() && field_dt.extended()->get_arrmeta_size() > 0) {
      o << indent << " field " << i << " (name ";
      o << m_field_names[i];
      o << ") arrmeta:\n";
      field_dt.extended()->arrmeta_debug_print(arrmeta + m_arrmeta_offsets[i], o, indent + "  ");
    }
  }
}

ndt::type ndt::struct_type::apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i,
                                               const type &root_tp, bool leading_dimension) const
{
  if (nindices == 0) {
    return type(this, true);
  }
  else {
    bool remove_dimension;
    intptr_t start_index, index_stride, dimension_size;
    apply_single_linear_index(*indices, m_field_count, current_i, &root_tp, remove_dimension, start_index, index_stride,
                              dimension_size);
    if (remove_dimension) {
      return get_field_type(start_index)
          .apply_linear_index(nindices - 1, indices + 1, current_i + 1, root_tp, leading_dimension);
    }
    else if (nindices == 1 && start_index == 0 && index_stride == 1 && dimension_size == m_field_count) {
      // This is a do-nothing index, keep the same type
      return type(this, true);
    }
    else {
      // Take the subset of the fields in-place
      std::vector<type> tmp_field_types(dimension_size);
      std::vector<std::string> tmp_field_names(dimension_size);

      for (intptr_t i = 0; i < dimension_size; ++i) {
        intptr_t idx = start_index + i * index_stride;
        tmp_field_types[i] =
            get_field_type(idx).apply_linear_index(nindices - 1, indices + 1, current_i + 1, root_tp, false);
        tmp_field_names[i] = m_field_names[idx];
      }

      return struct_type::make(tmp_field_names, tmp_field_types);
    }
  }
}

intptr_t ndt::struct_type::apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta,
                                              const type &result_tp, char *out_arrmeta,
                                              const intrusive_ptr<memory_block_data> &embedded_reference,
                                              size_t current_i, const type &root_tp, bool leading_dimension,
                                              char **inout_data, intrusive_ptr<memory_block_data> &inout_dataref) const
{
  if (nindices == 0) {
    // If there are no more indices, copy the arrmeta verbatim
    arrmeta_copy_construct(out_arrmeta, arrmeta, embedded_reference);
    return 0;
  }
  else {
    const uintptr_t *offsets = get_data_offsets(arrmeta);
    bool remove_dimension;
    intptr_t start_index, index_stride, dimension_size;
    apply_single_linear_index(*indices, m_field_count, current_i, &root_tp, remove_dimension, start_index, index_stride,
                              dimension_size);
    if (remove_dimension) {
      const type &dt = get_field_type(start_index);
      intptr_t offset = offsets[start_index];
      if (!dt.is_builtin()) {
        if (leading_dimension) {
          // In the case of a leading dimension, first bake the offset into
          // the data pointer, so that it's pointing at the right element
          // for the collapsing of leading dimensions to work correctly.
          *inout_data += offset;
          offset = dt.extended()->apply_linear_index(
              nindices - 1, indices + 1, arrmeta + m_arrmeta_offsets[start_index], result_tp, out_arrmeta,
              embedded_reference, current_i + 1, root_tp, true, inout_data, inout_dataref);
        }
        else {
          intrusive_ptr<memory_block_data> tmp;
          offset += dt.extended()->apply_linear_index(nindices - 1, indices + 1,
                                                      arrmeta + m_arrmeta_offsets[start_index], result_tp, out_arrmeta,
                                                      embedded_reference, current_i + 1, root_tp, false, NULL, tmp);
        }
      }
      return offset;
    }
    else {
      intrusive_ptr<memory_block_data> tmp;
      intptr_t *out_offsets = reinterpret_cast<intptr_t *>(out_arrmeta);
      const struct_type *result_e_dt = result_tp.extended<struct_type>();
      for (intptr_t i = 0; i < dimension_size; ++i) {
        intptr_t idx = start_index + i * index_stride;
        out_offsets[i] = offsets[idx];
        const type &dt = result_e_dt->get_field_type(i);
        if (!dt.is_builtin()) {
          out_offsets[i] +=
              dt.extended()->apply_linear_index(nindices - 1, indices + 1, arrmeta + m_arrmeta_offsets[idx], dt,
                                                out_arrmeta + result_e_dt->get_arrmeta_offset(i), embedded_reference,
                                                current_i + 1, root_tp, false, NULL, tmp);
        }
      }
      return 0;
    }
  }
}

std::map<std::string, nd::callable> ndt::struct_type::get_dynamic_type_properties() const
{
  std::map<std::string, nd::callable> properties;
  properties["field_types"] =
      nd::callable::make<nd::get_then_copy_kernel<const std::vector<type> &, tuple_type, &tuple_type::get_field_types>>(
          ndt::callable_type::make(get_type(), ndt::tuple_type::make(),
                                   ndt::struct_type::make({"self"}, {ndt::make_type<ndt::type_type>()})));
  properties["metadata_offsets"] = nd::callable::make<
      nd::get_then_copy_kernel<const std::vector<uintptr_t> &, tuple_type, &tuple_type::get_arrmeta_offsets>>(
      ndt::callable_type::make(ndt::type_for(m_arrmeta_offsets), ndt::tuple_type::make(),
                               ndt::struct_type::make({"self"}, {ndt::make_type<ndt::type_type>()})));
  properties["field_names"] = nd::callable::make<
      nd::get_then_copy_kernel<const std::vector<std::string> &, struct_type, &struct_type::get_field_names>>(
      ndt::callable_type::make(ndt::type_for(m_field_names), ndt::tuple_type::make(),
                               ndt::struct_type::make({"self"}, {ndt::make_type<ndt::type_type>()})));

  return properties;
}

namespace dynd {
namespace nd {

  struct get_array_field_kernel : nd::base_kernel<get_array_field_kernel> {
    array self;
    intptr_t i;

    get_array_field_kernel(const array &self, intptr_t i) : self(self), i(i) {}

    void call(array *dst, const array *DYND_UNUSED(src))
    {
      array res = helper(self, i);
      *dst = res;
    }

    static void resolve_dst_type(char *static_data, char *DYND_UNUSED(data), ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const array *kwds,
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = helper(kwds[0], *reinterpret_cast<intptr_t *>(static_data)).get_type();
    }

    static void instantiate(char *static_data, char *DYND_UNUSED(data), nd::kernel_builder *ckb,
                            const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                            intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                            const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                            intptr_t DYND_UNUSED(nkwd), const array *kwds,
                            const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      ckb->emplace_back<get_array_field_kernel>(kernreq, kwds[0], *reinterpret_cast<intptr_t *>(static_data));
    }

    static array helper(const array &n, intptr_t i)
    {
      // Get the nd::array 'self' parameter
      intptr_t undim = n.get_ndim();
      ndt::type udt = n.get_dtype();
      if (udt.get_kind() == expr_kind) {
        std::string field_name = udt.value_type().extended<ndt::struct_type>()->get_field_name(i);
        return n.replace_dtype(ndt::make_type<ndt::adapt_type>(
            udt.value_type().extended<ndt::struct_type>()->get_field_type(i), udt, nd::callable(), nd::callable()));
      }
      else {
        if (undim == 0) {
          return n(i);
        }
        else {
          shortvector<irange> idx(undim + 1);
          idx[undim] = irange(i);
          return n.at_array(undim + 1, idx.get());
        }
      }
    }
  };

} // namespace dynd::nd

} // namespace dynd

static std::vector<ndt::type> make_self_types()
{
  std::vector<ndt::type> result(1);
  result[0] = ndt::any_kind_type::make();
  return result;
}

ndt::struct_type::struct_type(int, int) : tuple_type(struct_id, make_self_types(), type_flag_none, false, false)
{
  // Equivalent to ndt::struct_type::make(ndt::make_ndarrayarg(), "self");
  // but hardcoded to break the dependency of struct_type::array_parameters_type
  m_arrmeta_offsets = {0};
  // The data offsets also consist of one zero
  //    m_data_offsets = m_arrmeta_offsets;
  // Inherit any operand flags from the fields
  this->flags |= (ndt::any_kind_type::make().get_flags() & type_flags_operand_inherited);
  this->data_alignment = sizeof(void *);
  this->arrmeta_size = 0;
  this->data_size = sizeof(void *);
  // Leave m_array_properties so there is no reference loop
}

void ndt::struct_type::create_array_properties()
{
  type array_parameters_type(new struct_type(0, 0), false);

  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    // TODO: Transform the name into a valid Python symbol?
    m_array_properties[get_field_name(i)] = nd::callable::make<nd::get_array_field_kernel>(
        callable_type::make(type("Any"), tuple_type::make(), array_parameters_type), i);
  }
}

std::map<std::string, nd::callable> ndt::struct_type::get_dynamic_array_properties() const
{
  return m_array_properties;
}

bool ndt::struct_type::match(const char *arrmeta, const type &candidate_tp, const char *candidate_arrmeta,
                             std::map<std::string, type> &tp_vars) const
{
  intptr_t candidate_field_count = candidate_tp.extended<struct_type>()->get_field_count();
  bool candidate_variadic = candidate_tp.extended<tuple_type>()->is_variadic();

  if ((m_field_count == candidate_field_count && !candidate_variadic) ||
      ((candidate_field_count >= m_field_count) && m_variadic)) {
    // Compare the field names
    const std::vector<std::string> &candidate_names = candidate_tp.extended<struct_type>()->get_field_names();
    if (!std::equal(m_field_names.begin(), m_field_names.end(), candidate_names.begin())) {
      return false;
    }

    // Compare the field types
    const std::vector<type> &candidate_fields = candidate_tp.extended<struct_type>()->get_field_types();
    for (intptr_t i = 0; i < m_field_count; ++i) {
      if (!m_field_types[i].match(arrmeta, candidate_fields[i], candidate_arrmeta, tp_vars)) {
        return false;
      }
    }
    return true;
  }

  return false;
}
