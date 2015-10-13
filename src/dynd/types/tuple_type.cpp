//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/tuple_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/kernels/tuple_assignment_kernels.hpp>
#include <dynd/kernels/tuple_comparison_kernels.hpp>
#include <dynd/kernels/base_property_kernel.hpp>

using namespace std;
using namespace dynd;

ndt::tuple_type::~tuple_type()
{
}

void ndt::tuple_type::print_type(std::ostream &o) const
{
  // Use the tuple datashape syntax
  o << "(";
  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    if (i != 0) {
      o << ", ";
    }
    o << get_field_type(i);
  }
  if (m_variadic) {
    o << ", ...)";
  } else {
    o << ")";
  }
}

void ndt::tuple_type::transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                                            type &out_transformed_tp, bool &out_was_transformed) const
{
  nd::array tmp_field_types(nd::empty(m_field_count, make_type()));
  type *tmp_field_types_raw = reinterpret_cast<type *>(tmp_field_types.get_readwrite_originptr());

  bool was_transformed = false;
  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    transform_fn(get_field_type(i), arrmeta_offset + get_arrmeta_offset(i), extra, tmp_field_types_raw[i],
                 was_transformed);
  }
  if (was_transformed) {
    tmp_field_types.flag_as_immutable();
    out_transformed_tp = make(tmp_field_types, m_variadic);
    out_was_transformed = true;
  } else {
    out_transformed_tp = type(this, true);
  }
}

ndt::type ndt::tuple_type::get_canonical_type() const
{
  nd::array tmp_field_types(nd::empty(m_field_count, make_type()));
  type *tmp_field_types_raw = reinterpret_cast<type *>(tmp_field_types.get_readwrite_originptr());

  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    tmp_field_types_raw[i] = get_field_type(i).get_canonical_type();
  }

  tmp_field_types.flag_as_immutable();
  return make(tmp_field_types, m_variadic);
}

bool ndt::tuple_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    } else if (src_tp.get_type_id() == tuple_type_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

intptr_t ndt::tuple_type::make_assignment_kernel(void *ckb, intptr_t ckb_offset, const type &dst_tp,
                                                 const char *dst_arrmeta, const type &src_tp, const char *src_arrmeta,
                                                 kernel_request_t kernreq, const eval::eval_context *ectx) const
{
  if (this == dst_tp.extended()) {
    if (this == src_tp.extended()) {
      return make_tuple_identical_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_arrmeta, kernreq, ectx);
    } else if (src_tp.get_kind() == tuple_kind || src_tp.get_kind() == struct_kind) {
      return make_tuple_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta, kernreq, ectx);
    } else if (src_tp.is_builtin()) {
      return make_broadcast_to_tuple_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
                                                       kernreq, ectx);
    } else {
      return src_tp.extended()->make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
                                                       kernreq, ectx);
    }
  }

  stringstream ss;
  ss << "Cannot assign from " << src_tp << " to " << dst_tp;
  throw dynd::type_error(ss.str());
}

size_t ndt::tuple_type::make_comparison_kernel(void *ckb, intptr_t ckb_offset, const type &src0_tp,
                                               const char *src0_arrmeta, const type &src1_tp, const char *src1_arrmeta,
                                               comparison_type_t comptype, const eval::eval_context *ectx) const
{
  if (this == src0_tp.extended()) {
    if (*this == *src1_tp.extended()) {
      return make_tuple_comparison_kernel(ckb, ckb_offset, src0_tp, src0_arrmeta, src1_arrmeta, comptype, ectx);
    } else if (src1_tp.get_kind() == tuple_kind) {
      // TODO
    }
  }

  throw not_comparable_error(src0_tp, src1_tp, comptype);
}

bool ndt::tuple_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != tuple_type_id) {
    return false;
  } else {
    const tuple_type *dt = static_cast<const tuple_type *>(&rhs);
    return get_data_alignment() == dt->get_data_alignment() && m_field_types.equals_exact(dt->m_field_types) &&
           m_variadic == dt->m_variadic;
  }
}

void ndt::tuple_type::arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const
{
  const size_t *data_offsets = reinterpret_cast<const size_t *>(arrmeta);
  o << indent << "tuple arrmeta\n";
  o << indent << " field offsets: ";
  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    o << data_offsets[i];
    if (i != i_end - 1) {
      o << ", ";
    }
  }
  o << "\n";
  const uintptr_t *arrmeta_offsets = get_arrmeta_offsets_raw();
  for (intptr_t i = 0; i < m_field_count; ++i) {
    const type &field_dt = get_field_type(i);
    if (!field_dt.is_builtin() && field_dt.extended()->get_arrmeta_size() > 0) {
      o << indent << " field " << i << " arrmeta:\n";
      field_dt.extended()->arrmeta_debug_print(arrmeta + arrmeta_offsets[i], o, indent + "  ");
    }
  }
}

/*
static nd::array property_get_field_types(const ndt::type &tp)
{
  return tp.extended<ndt::tuple_type>()->get_field_types();
}

static nd::array property_get_arrmeta_offsets(const ndt::type &tp)
{
  return tp.extended<ndt::tuple_type>()->get_arrmeta_offsets();
}
*/

void ndt::tuple_type::get_dynamic_type_properties(const std::pair<std::string, nd::callable> **out_properties,
                                                  size_t *out_count) const
{
  struct field_types_kernel : nd::base_property_kernel<field_types_kernel> {
    field_types_kernel(const ndt::type &tp, const ndt::type &dst_tp, const char *dst_arrmeta)
        : base_property_kernel<field_types_kernel>(tp, dst_tp, dst_arrmeta)
    {
    }

    void single(char *dst, char *const *DYND_UNUSED(src))
    {
      typed_data_copy(dst_tp, dst_arrmeta, dst, tp.extended<tuple_type>()->m_field_types.get_arrmeta(),
                      tp.extended<tuple_type>()->m_field_types.get_data());
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size), char *data,
                                 ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const dynd::nd::array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      const type &tp = *reinterpret_cast<const ndt::type *>(data);
      dst_tp = tp.extended<tuple_type>()->m_field_types.get_type();
    }
  };

  struct arrmeta_offsets_kernel : nd::base_property_kernel<arrmeta_offsets_kernel> {
    arrmeta_offsets_kernel(const ndt::type &tp, const ndt::type &dst_tp, const char *dst_arrmeta)
        : base_property_kernel<arrmeta_offsets_kernel>(tp, dst_tp, dst_arrmeta)
    {
    }

    void single(char *dst, char *const *DYND_UNUSED(src))
    {
      typed_data_copy(dst_tp, dst_arrmeta, dst, tp.extended<tuple_type>()->m_arrmeta_offsets.get_arrmeta(),
                      tp.extended<tuple_type>()->m_arrmeta_offsets.get_data());
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size), char *data,
                                 ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const dynd::nd::array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      const type &tp = *reinterpret_cast<const ndt::type *>(data);
      dst_tp = tp.extended<tuple_type>()->m_arrmeta_offsets.get_type();
    }
  };

  static pair<std::string, nd::callable> type_properties[] = {
      pair<std::string, nd::callable>("field_types",
                                      nd::callable::make<field_types_kernel>(type("(self: type) -> Any"))),
      pair<std::string, nd::callable>("arrmeta_offsets",
                                      nd::callable::make<arrmeta_offsets_kernel>(type("(self: type) -> Any"))), };

  *out_properties = type_properties;
  *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}

nd::array ndt::pack(intptr_t field_count, const nd::array *field_vals)
{
  if (field_count == 0) {
    return nd::array();
  }

  vector<type> field_types(field_count);
  for (intptr_t i = 0; i < field_count; ++i) {
    field_types[i] = field_vals[i].get_type();
  }

  nd::array res = nd::empty(ndt::tuple_type::make(field_types));
  for (intptr_t i = 0; i < field_count; ++i) {
    res.vals_at(i) = field_vals[i];
  }

  return res;
}
