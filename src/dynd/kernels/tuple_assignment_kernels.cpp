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

struct tuple_unary_op_ck : nd::base_strided_kernel<tuple_unary_op_ck, 1> {
  vector<tuple_unary_op_item> m_fields;

  ~tuple_unary_op_ck()
  {
    for (size_t i = 0; i < m_fields.size(); ++i) {
      get_child(m_fields[i].child_kernel_offset)->destroy();
    }
  }

  void single(char *dst, char *const *src)
  {
    const tuple_unary_op_item *fi = &m_fields[0];
    intptr_t field_count = m_fields.size();
    kernel_prefix *child;
    kernel_single_t child_fn;

    for (intptr_t i = 0; i < field_count; ++i) {
      const tuple_unary_op_item &item = fi[i];
      child = get_child(item.child_kernel_offset);
      child_fn = child->get_function<kernel_single_t>();
      char *child_src = src[0] + item.src_data_offset;
      child_fn(child, dst + item.dst_data_offset, &child_src);
    }
  }
};
} // anonymous namespace

void dynd::make_tuple_unary_op_ckernel(const nd::base_callable *af, const ndt::callable_type *DYND_UNUSED(af_tp),
                                       nd::kernel_builder *ckb, intptr_t field_count, const uintptr_t *dst_offsets,
                                       const ndt::type *dst_tp, const char *const *dst_arrmeta,
                                       const uintptr_t *src_offsets, const ndt::type *src_tp,
                                       const char *const *src_arrmeta, kernel_request_t kernreq)
{
  intptr_t self_offset = ckb->size();
  ckb->emplace_back<tuple_unary_op_ck>(kernreq);
  tuple_unary_op_ck *self = ckb->get_at<tuple_unary_op_ck>(self_offset);
  self->m_fields.resize(field_count);
  for (intptr_t i = 0; i < field_count; ++i) {
    self = ckb->get_at<tuple_unary_op_ck>(self_offset);
    tuple_unary_op_item &field = self->m_fields[i];
    field.child_kernel_offset = ckb->size() - self_offset;
    field.dst_data_offset = dst_offsets[i];
    field.src_data_offset = src_offsets[i];
    nd::array error_mode = ndt::traits<assign_error_mode>::na();
    const_cast<nd::base_callable *>(af)->instantiate(NULL, NULL, ckb, dst_tp[i], dst_arrmeta[i], 1, &src_tp[i],
                                                     &src_arrmeta[i], kernel_request_single, 1, &error_mode,
                                                     std::map<std::string, ndt::type>());
  }
}

/////////////////////////////////////////
// tuple/struct to identical tuple/struct assignment

void dynd::make_tuple_identical_assignment_kernel(nd::kernel_builder *ckb, const ndt::type &val_tup_tp,
                                                  const char *dst_arrmeta, const char *src_arrmeta,
                                                  kernel_request_t kernreq)
{
  if (val_tup_tp.get_id() != tuple_id && val_tup_tp.get_id() != struct_id) {
    stringstream ss;
    ss << "make_tuple_identical_assignment_kernel: provided type " << val_tup_tp << " is not of tuple or struct kind";
    throw runtime_error(ss.str());
  }
  if (val_tup_tp.is_pod()) {
    // For POD structs, get a trivial memory copy kernel
    make_pod_typed_data_assignment_kernel(ckb, val_tup_tp.get_data_size(), val_tup_tp.get_data_alignment(), kernreq);
    return;
  }

  auto sd = val_tup_tp.extended<ndt::tuple_type>();
  intptr_t field_count = sd->get_field_count();
  const std::vector<uintptr_t> &arrmeta_offsets = sd->get_arrmeta_offsets();
  shortvector<const char *> dst_fields_arrmeta(field_count);
  for (intptr_t i = 0; i != field_count; ++i) {
    dst_fields_arrmeta[i] = dst_arrmeta + arrmeta_offsets[i];
  }
  shortvector<const char *> src_fields_arrmeta(field_count);
  for (intptr_t i = 0; i != field_count; ++i) {
    src_fields_arrmeta[i] = src_arrmeta + arrmeta_offsets[i];
  }

  make_tuple_unary_op_ckernel(nd::copy::get().get(), nd::copy::get().get_type(), ckb, field_count,
                              sd->get_data_offsets(dst_arrmeta), sd->get_field_types().data(), dst_fields_arrmeta.get(),
                              sd->get_data_offsets(src_arrmeta), sd->get_field_types().data(), src_fields_arrmeta.get(),
                              kernreq);
}

/////////////////////////////////////////
// struct/tuple to different struct/tuple assignment
// (matches up fields by number, not name in struct case)

void dynd::make_tuple_assignment_kernel(nd::kernel_builder *ckb, const ndt::type &dst_tuple_tp, const char *dst_arrmeta,
                                        const ndt::type &src_tuple_tp, const char *src_arrmeta,
                                        kernel_request_t kernreq)
{
  if (src_tuple_tp.get_id() != tuple_id && src_tuple_tp.get_id() != struct_id) {
    stringstream ss;
    ss << "make_tuple_assignment_kernel: provided source type " << src_tuple_tp << " is not of tuple or struct kind";
    throw runtime_error(ss.str());
  }
  if (dst_tuple_tp.get_id() != tuple_id && dst_tuple_tp.get_id() != struct_id) {
    stringstream ss;
    ss << "make_tuple_assignment_kernel: provided destination type " << dst_tuple_tp
       << " is not of tuple or struct kind";
    throw runtime_error(ss.str());
  }
  auto dst_sd = dst_tuple_tp.extended<ndt::tuple_type>();
  auto src_sd = src_tuple_tp.extended<ndt::tuple_type>();
  intptr_t field_count = dst_sd->get_field_count();

  if (field_count != src_sd->get_field_count()) {
    stringstream ss;
    ss << "cannot assign dynd " << src_tuple_tp << " to " << dst_tuple_tp
       << " because they have different numbers of fields";
    throw type_error(ss.str());
  }

  const std::vector<uintptr_t> &src_arrmeta_offsets = src_sd->get_arrmeta_offsets();
  shortvector<const char *> src_fields_arrmeta(field_count);
  for (intptr_t i = 0; i != field_count; ++i) {
    src_fields_arrmeta[i] = src_arrmeta + src_arrmeta_offsets[i];
  }

  const std::vector<uintptr_t> &dst_arrmeta_offsets = dst_sd->get_arrmeta_offsets();
  shortvector<const char *> dst_fields_arrmeta(field_count);
  for (intptr_t i = 0; i != field_count; ++i) {
    dst_fields_arrmeta[i] = dst_arrmeta + dst_arrmeta_offsets[i];
  }

  make_tuple_unary_op_ckernel(nd::copy::get().get(), nd::copy::get().get_type(), ckb, field_count,
                              dst_sd->get_data_offsets(dst_arrmeta), dst_sd->get_field_types().data(),
                              dst_fields_arrmeta.get(), src_sd->get_data_offsets(src_arrmeta),
                              src_sd->get_field_types().data(), src_fields_arrmeta.get(), kernreq);
}

/////////////////////////////////////////
// value to each tuple/struct field assignment

void dynd::make_broadcast_to_tuple_assignment_kernel(nd::kernel_builder *ckb, const ndt::type &dst_tuple_tp,
                                                     const char *dst_arrmeta, const ndt::type &src_tp,
                                                     const char *src_arrmeta, kernel_request_t kernreq)
{
  // This implementation uses the same struct to struct kernel, just with
  // an offset of 0 for each source value. A kernel tailored to this
  // case can be made if better performance is needed.

  if (dst_tuple_tp.get_id() != tuple_id && dst_tuple_tp.get_id() != struct_id) {
    stringstream ss;
    ss << "make_tuple_assignment_kernel: provided destination type " << dst_tuple_tp
       << " is not of tuple or struct kind";
    throw runtime_error(ss.str());
  }
  auto dst_sd = dst_tuple_tp.extended<ndt::tuple_type>();
  intptr_t field_count = dst_sd->get_field_count();

  const std::vector<uintptr_t> &dst_arrmeta_offsets = dst_sd->get_arrmeta_offsets();
  shortvector<const char *> dst_fields_arrmeta(field_count);
  for (intptr_t i = 0; i != field_count; ++i) {
    dst_fields_arrmeta[i] = dst_arrmeta + dst_arrmeta_offsets[i];
  }
  vector<ndt::type> src_fields_tp(field_count, src_tp);
  vector<const char *> src_fields_arrmeta(field_count, src_arrmeta);
  vector<uintptr_t> src_data_offsets(field_count, 0);

  make_tuple_unary_op_ckernel(nd::copy::get().get(), nd::copy::get().get_type(), ckb, field_count,
                              dst_sd->get_data_offsets(dst_arrmeta), dst_sd->get_field_types().data(),
                              dst_fields_arrmeta.get(), &src_data_offsets[0], &src_fields_tp[0], &src_fields_arrmeta[0],
                              kernreq);
}
