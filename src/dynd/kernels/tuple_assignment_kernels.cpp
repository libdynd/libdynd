//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>
#include <algorithm>

#include <dynd/type.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/tuple_assignment_kernels.hpp>
#include <dynd/func/copy.hpp>

using namespace std;
using namespace dynd;

namespace {
struct tuple_unary_op_item {
  size_t child_kernel_offset;
  size_t dst_data_offset;
  size_t src_data_offset;
};

struct tuple_unary_op_ck : nd::base_kernel<tuple_unary_op_ck, 1> {
  vector<tuple_unary_op_item> m_fields;

  ~tuple_unary_op_ck()
  {
    for (size_t i = 0; i < m_fields.size(); ++i) {
      get_child_ckernel(m_fields[i].child_kernel_offset)->destroy();
    }
  }

  void single(char *dst, char *const *src)
  {
    const tuple_unary_op_item *fi = &m_fields[0];
    intptr_t field_count = m_fields.size();
    ckernel_prefix *child;
    expr_single_t child_fn;

    for (intptr_t i = 0; i < field_count; ++i) {
      const tuple_unary_op_item &item = fi[i];
      child = get_child_ckernel(item.child_kernel_offset);
      child_fn = child->get_function<expr_single_t>();
      char *child_src = src[0] + item.src_data_offset;
      child_fn(child, dst + item.dst_data_offset, &child_src);
    }
  }
};
} // anonymous namespace

intptr_t dynd::make_tuple_unary_op_ckernel(
    const callable_type_data *af, const ndt::callable_type *DYND_UNUSED(af_tp),
    void *ckb, intptr_t ckb_offset, intptr_t field_count,
    const uintptr_t *dst_offsets, const ndt::type *dst_tp,
    const char *const *dst_arrmeta, const uintptr_t *src_offsets,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
  intptr_t root_ckb_offset = ckb_offset;
  tuple_unary_op_ck *self = tuple_unary_op_ck::make(ckb, kernreq, ckb_offset);
  self->m_fields.resize(field_count);
  for (intptr_t i = 0; i < field_count; ++i) {
    reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
        ->reserve(ckb_offset + sizeof(ckernel_prefix));
    self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
               ->get_at<tuple_unary_op_ck>(root_ckb_offset);
    tuple_unary_op_item &field = self->m_fields[i];
    field.child_kernel_offset = ckb_offset - root_ckb_offset;
    field.dst_data_offset = dst_offsets[i];
    field.src_data_offset = src_offsets[i];
    ckb_offset = af->instantiate(NULL, 0, NULL, ckb, ckb_offset, dst_tp[i],
                                 dst_arrmeta[i], 1, &src_tp[i], &src_arrmeta[i],
                                 kernel_request_single, ectx, 0, NULL,
                                 std::map<std::string, ndt::type>());
  }
  return ckb_offset;
}

intptr_t dynd::make_tuple_unary_op_ckernel(
    const callable_type_data *const *af,
    const ndt::callable_type *const *DYND_UNUSED(af_tp), void *ckb,
    intptr_t ckb_offset, intptr_t field_count, const uintptr_t *dst_offsets,
    const ndt::type *dst_tp, const char *const *dst_arrmeta,
    const uintptr_t *src_offsets, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  intptr_t root_ckb_offset = ckb_offset;
  tuple_unary_op_ck *self = tuple_unary_op_ck::make(ckb, kernreq, ckb_offset);
  self->m_fields.resize(field_count);
  for (intptr_t i = 0; i < field_count; ++i) {
    reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
        ->reserve(ckb_offset + sizeof(ckernel_prefix));
    self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
               ->get_at<tuple_unary_op_ck>(root_ckb_offset);
    tuple_unary_op_item &field = self->m_fields[i];
    field.child_kernel_offset = ckb_offset - root_ckb_offset;
    field.dst_data_offset = dst_offsets[i];
    field.src_data_offset = src_offsets[i];
    ckb_offset = af[i]->instantiate(
        NULL, 0, NULL, ckb, ckb_offset, dst_tp[i], dst_arrmeta[i], 1,
        &src_tp[i], &src_arrmeta[i], kernel_request_single, ectx, 0, NULL,
        std::map<std::string, ndt::type>());
  }
  return ckb_offset;
}

/////////////////////////////////////////
// tuple/struct to identical tuple/struct assignment

size_t dynd::make_tuple_identical_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const ndt::type &val_tup_tp,
    const char *dst_arrmeta, const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  if (val_tup_tp.get_kind() != tuple_kind &&
      val_tup_tp.get_kind() != struct_kind) {
    stringstream ss;
    ss << "make_tuple_identical_assignment_kernel: provided type " << val_tup_tp
       << " is not of tuple or struct kind";
    throw runtime_error(ss.str());
  }
  if (val_tup_tp.is_pod()) {
    // For POD structs, get a trivial memory copy kernel
    return make_pod_typed_data_assignment_kernel(
        ckb, ckb_offset, val_tup_tp.get_data_size(),
        val_tup_tp.get_data_alignment(), kernreq);
  }

  auto sd = val_tup_tp.extended<ndt::base_tuple_type>();
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
      nd::copy::get().get(), nd::copy::get().get_type(), ckb, ckb_offset,
      field_count, sd->get_data_offsets(dst_arrmeta), sd->get_field_types_raw(),
      dst_fields_arrmeta.get(), sd->get_data_offsets(src_arrmeta),
      sd->get_field_types_raw(), src_fields_arrmeta.get(), kernreq, ectx);
}

/////////////////////////////////////////
// struct/tuple to different struct/tuple assignment
// (matches up fields by number, not name in struct case)

size_t dynd::make_tuple_assignment_kernel(void *ckb, intptr_t ckb_offset,
                                          const ndt::type &dst_tuple_tp,
                                          const char *dst_arrmeta,
                                          const ndt::type &src_tuple_tp,
                                          const char *src_arrmeta,
                                          kernel_request_t kernreq,
                                          const eval::eval_context *ectx)
{
  if (src_tuple_tp.get_kind() != tuple_kind &&
      src_tuple_tp.get_kind() != struct_kind) {
    stringstream ss;
    ss << "make_tuple_assignment_kernel: provided source type " << src_tuple_tp
       << " is not of tuple or struct kind";
    throw runtime_error(ss.str());
  }
  if (dst_tuple_tp.get_kind() != tuple_kind &&
      dst_tuple_tp.get_kind() != struct_kind) {
    stringstream ss;
    ss << "make_tuple_assignment_kernel: provided destination type "
       << dst_tuple_tp << " is not of tuple or struct kind";
    throw runtime_error(ss.str());
  }
  auto dst_sd = dst_tuple_tp.extended<ndt::base_tuple_type>();
  auto src_sd = src_tuple_tp.extended<ndt::base_tuple_type>();
  intptr_t field_count = dst_sd->get_field_count();

  if (field_count != src_sd->get_field_count()) {
    stringstream ss;
    ss << "cannot assign dynd " << src_tuple_tp << " to " << dst_tuple_tp
       << " because they have different numbers of fields";
    throw type_error(ss.str());
  }

  const uintptr_t *src_arrmeta_offsets = src_sd->get_arrmeta_offsets_raw();
  shortvector<const char *> src_fields_arrmeta(field_count);
  for (intptr_t i = 0; i != field_count; ++i) {
    src_fields_arrmeta[i] = src_arrmeta + src_arrmeta_offsets[i];
  }

  const uintptr_t *dst_arrmeta_offsets = dst_sd->get_arrmeta_offsets_raw();
  shortvector<const char *> dst_fields_arrmeta(field_count);
  for (intptr_t i = 0; i != field_count; ++i) {
    dst_fields_arrmeta[i] = dst_arrmeta + dst_arrmeta_offsets[i];
  }

  return make_tuple_unary_op_ckernel(
      nd::copy::get().get(), nd::copy::get().get_type(), ckb, ckb_offset,
      field_count, dst_sd->get_data_offsets(dst_arrmeta),
      dst_sd->get_field_types_raw(), dst_fields_arrmeta.get(),
      src_sd->get_data_offsets(src_arrmeta), src_sd->get_field_types_raw(),
      src_fields_arrmeta.get(), kernreq, ectx);
}

/////////////////////////////////////////
// value to each tuple/struct field assignment

size_t dynd::make_broadcast_to_tuple_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tuple_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
  // This implementation uses the same struct to struct kernel, just with
  // an offset of 0 for each source value. A kernel tailored to this
  // case can be made if better performance is needed.

  if (dst_tuple_tp.get_kind() != tuple_kind &&
      dst_tuple_tp.get_kind() != struct_kind) {
    stringstream ss;
    ss << "make_tuple_assignment_kernel: provided destination type "
       << dst_tuple_tp << " is not of tuple or struct kind";
    throw runtime_error(ss.str());
  }
  auto dst_sd = dst_tuple_tp.extended<ndt::base_tuple_type>();
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
      nd::copy::get().get(), nd::copy::get().get_type(), ckb, ckb_offset,
      field_count, dst_sd->get_data_offsets(dst_arrmeta),
      dst_sd->get_field_types_raw(), dst_fields_arrmeta.get(),
      &src_data_offsets[0], &src_fields_tp[0], &src_fields_arrmeta[0], kernreq,
      ectx);
}
