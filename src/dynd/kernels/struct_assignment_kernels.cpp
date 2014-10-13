//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>
#include <algorithm>

#include <dynd/type.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/struct_assignment_kernels.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/func/copy_arrfunc.hpp>

using namespace std;
using namespace dynd;

namespace {
struct tuple_unary_op_item {
  size_t child_kernel_offset;
  size_t dst_data_offset;
  size_t src_data_offset;
};

struct tuple_unary_op_ck : public kernels::unary_ck<tuple_unary_op_ck> {
  vector<tuple_unary_op_item> m_fields;

  inline void single(char *dst, char *src)
  {
    const tuple_unary_op_item *fi = &m_fields[0];
    intptr_t field_count = m_fields.size();
    ckernel_prefix *child;
    expr_single_t child_fn;

    for (intptr_t i = 0; i < field_count; ++i) {
      const tuple_unary_op_item &item = fi[i];
      child = get_child_ckernel(item.child_kernel_offset);
      child_fn = child->get_function<expr_single_t>();
      char *child_src = src + item.src_data_offset;
      child_fn(dst + item.dst_data_offset, &child_src, child);
    }
  }

  inline void destruct_children()
  {
    for (size_t i = 0; i < m_fields.size(); ++i) {
      base.destroy_child_ckernel(m_fields[i].child_kernel_offset);
    }
  }
};
} // anonymous namespace

intptr_t dynd::make_tuple_unary_op_ckernel(
    const arrfunc_type_data *af, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, intptr_t field_count, const uintptr_t *dst_offsets,
    const ndt::type *dst_tp, const char *const *dst_arrmeta,
    const uintptr_t *src_offsets, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  intptr_t root_ckb_offset = ckb_offset;
  tuple_unary_op_ck *self = tuple_unary_op_ck::create(ckb, kernreq, ckb_offset);
  self->m_fields.resize(field_count);
  for (intptr_t i = 0; i < field_count; ++i) {
    ckb->ensure_capacity(ckb_offset);
    self = ckb->get_at<tuple_unary_op_ck>(root_ckb_offset);
    tuple_unary_op_item &field = self->m_fields[i];
    field.child_kernel_offset = ckb_offset - root_ckb_offset;
    field.dst_data_offset = dst_offsets[i];
    field.src_data_offset = src_offsets[i];
    ckb_offset = af->instantiate(af, ckb, ckb_offset, dst_tp[i], dst_arrmeta[i],
                                 &src_tp[i], &src_arrmeta[i],
                                 kernel_request_single, nd::array(), ectx);
  }
  return ckb_offset;
}

intptr_t dynd::make_tuple_unary_op_ckernel(
    const arrfunc_type_data *const *af, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, intptr_t field_count, const uintptr_t *dst_offsets,
    const ndt::type *dst_tp, const char *const *dst_arrmeta,
    const uintptr_t *src_offsets, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  intptr_t root_ckb_offset = ckb_offset;
  tuple_unary_op_ck *self = tuple_unary_op_ck::create(ckb, kernreq, ckb_offset);
  self->m_fields.resize(field_count);
  for (intptr_t i = 0; i < field_count; ++i) {
    ckb->ensure_capacity(ckb_offset);
    self = ckb->get_at<tuple_unary_op_ck>(root_ckb_offset);
    tuple_unary_op_item &field = self->m_fields[i];
    field.child_kernel_offset = ckb_offset - root_ckb_offset;
    field.dst_data_offset = dst_offsets[i];
    field.src_data_offset = src_offsets[i];
    ckb_offset = af[i]->instantiate(af[i], ckb, ckb_offset, dst_tp[i],
                                    dst_arrmeta[i], &src_tp[i], &src_arrmeta[i],
                                    kernel_request_single, nd::array(), ectx);
  }
  return ckb_offset;
}

/////////////////////////////////////////
// struct to identical struct assignment

size_t dynd::make_struct_identical_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &val_struct_tp,
    const char *dst_arrmeta, const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  if (val_struct_tp.get_kind() != struct_kind) {
    stringstream ss;
    ss << "make_struct_identical_assignment_kernel: provided type "
       << val_struct_tp << " is not of struct kind";
    throw runtime_error(ss.str());
  }
  if (val_struct_tp.is_pod()) {
    // For POD structs, get a trivial memory copy kernel
    return make_pod_typed_data_assignment_kernel(
        ckb, ckb_offset, val_struct_tp.get_data_size(),
        val_struct_tp.get_data_alignment(), kernreq);
  }

  const base_struct_type *sd = val_struct_tp.tcast<base_struct_type>();
  intptr_t field_count = sd->get_field_count();
  const uintptr_t *arrmeta_offsets = sd->get_arrmeta_offsets_raw();
  shortvector<const char *> dst_fields_arrmeta(field_count);
  for (intptr_t i = 0; i != field_count; ++i) {
    dst_fields_arrmeta[i] = dst_arrmeta + arrmeta_offsets[i];
  }
  shortvector<const char *> src_fields_arrmeta(field_count);
  for (intptr_t i = 0; i != field_count; ++i) {
    src_fields_arrmeta[i] = src_arrmeta + arrmeta_offsets[i];
  }

  return make_tuple_unary_op_ckernel(
      make_copy_arrfunc().get(), ckb, ckb_offset, field_count,
      sd->get_data_offsets(dst_arrmeta), sd->get_field_types_raw(),
      dst_fields_arrmeta.get(), sd->get_data_offsets(src_arrmeta),
      sd->get_field_types_raw(), src_fields_arrmeta.get(), kernreq, ectx);
}

/////////////////////////////////////////
// struct to different struct assignment

size_t dynd::make_struct_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_struct_tp,
    const char *dst_arrmeta, const ndt::type &src_struct_tp,
    const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  if (src_struct_tp.get_kind() != struct_kind) {
    stringstream ss;
    ss << "make_struct_assignment_kernel: provided source type "
       << src_struct_tp << " is not of struct kind";
    throw runtime_error(ss.str());
  }
  if (dst_struct_tp.get_kind() != struct_kind) {
    stringstream ss;
    ss << "make_struct_assignment_kernel: provided destination type "
       << dst_struct_tp << " is not of struct kind";
    throw runtime_error(ss.str());
  }
  const base_struct_type *dst_sd = dst_struct_tp.tcast<base_struct_type>();
  const base_struct_type *src_sd = src_struct_tp.tcast<base_struct_type>();
  intptr_t field_count = dst_sd->get_field_count();

  if (field_count != src_sd->get_field_count()) {
    stringstream ss;
    ss << "cannot assign dynd struct " << src_struct_tp << " to "
       << dst_struct_tp;
    ss << " because they have different numbers of fields";
    throw runtime_error(ss.str());
  }

  const ndt::type *src_fields_tp_orig = src_sd->get_field_types_raw();
  const uintptr_t *src_arrmeta_offsets_orig = src_sd->get_arrmeta_offsets_raw();
  const uintptr_t *src_data_offsets_orig = src_sd->get_data_offsets(src_arrmeta);
  vector<ndt::type> src_fields_tp(field_count);
  shortvector<uintptr_t> src_data_offsets(field_count);
  shortvector<const char *> src_fields_arrmeta(field_count);

  // Match up the fields
  for (intptr_t i = 0; i != field_count; ++i) {
    const string_type_data &dst_name = dst_sd->get_field_name_raw(i);
    intptr_t src_i = src_sd->get_field_index(dst_name.begin, dst_name.end);
    if (src_i < 0) {
      stringstream ss;
      ss << "cannot assign dynd struct " << src_struct_tp << " to "
         << dst_struct_tp;
      ss << " because they have different field names";
      throw runtime_error(ss.str());
    }
    src_fields_tp[i] = src_fields_tp_orig[src_i];
    src_data_offsets[i] = src_data_offsets_orig[src_i];
    src_fields_arrmeta[i] = src_arrmeta + src_arrmeta_offsets_orig[src_i];
  }

  const uintptr_t *dst_arrmeta_offsets = dst_sd->get_arrmeta_offsets_raw();
  shortvector<const char *> dst_fields_arrmeta(field_count);
  for (intptr_t i = 0; i != field_count; ++i) {
    dst_fields_arrmeta[i] = dst_arrmeta + dst_arrmeta_offsets[i];
  }

  return make_tuple_unary_op_ckernel(
      make_copy_arrfunc().get(), ckb, ckb_offset, field_count,
      dst_sd->get_data_offsets(dst_arrmeta), dst_sd->get_field_types_raw(),
      dst_fields_arrmeta.get(), src_data_offsets.get(),
      &src_fields_tp[0], src_fields_arrmeta.get(), kernreq, ectx);
}

/////////////////////////////////////////
// value to each struct field assignment

size_t dynd::make_broadcast_to_struct_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_struct_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
  // This implementation uses the same struct to struct kernel, just with
  // an offset of 0 for each source value. A kernel tailored to this
  // case can be made if better performance is needed.

  if (dst_struct_tp.get_kind() != struct_kind) {
    stringstream ss;
    ss << "make_struct_assignment_kernel: provided destination type "
       << dst_struct_tp << " is not of struct kind";
    throw runtime_error(ss.str());
  }
  const base_struct_type *dst_sd = dst_struct_tp.tcast<base_struct_type>();
  intptr_t field_count = dst_sd->get_field_count();

  const uintptr_t *dst_arrmeta_offsets = dst_sd->get_arrmeta_offsets_raw();
  shortvector<const char *> dst_fields_arrmeta(field_count);
  for (intptr_t i = 0; i != field_count; ++i) {
    dst_fields_arrmeta[i] = dst_arrmeta + dst_arrmeta_offsets[i];
  }
  vector<ndt::type> src_fields_tp(field_count, src_tp);
  vector<const char *> src_fields_arrmeta(field_count, src_arrmeta);
  vector<uintptr_t> src_data_offsets(field_count, 0);

  return make_tuple_unary_op_ckernel(
      make_copy_arrfunc().get(), ckb, ckb_offset, field_count,
      dst_sd->get_data_offsets(dst_arrmeta), dst_sd->get_field_types_raw(),
      dst_fields_arrmeta.get(), &src_data_offsets[0], &src_fields_tp[0],
      &src_fields_arrmeta[0], kernreq, ectx);
}
