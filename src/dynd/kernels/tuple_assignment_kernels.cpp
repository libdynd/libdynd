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
#include <dynd/assignment.hpp>

using namespace std;
using namespace dynd;

namespace {
} // anonymous namespace

void dynd::make_tuple_unary_op_ckernel(const nd::base_callable *af, const ndt::callable_type *DYND_UNUSED(af_tp),
                                       nd::kernel_builder *ckb, intptr_t field_count, const uintptr_t *dst_offsets,
                                       const ndt::type *dst_tp, const char *const *dst_arrmeta,
                                       const uintptr_t *src_offsets, const ndt::type *src_tp,
                                       const char *const *src_arrmeta, kernel_request_t kernreq) {
  intptr_t self_offset = ckb->size();
  ckb->emplace_back<nd::tuple_unary_op_ck>(kernreq);
  nd::tuple_unary_op_ck *self = ckb->get_at<nd::tuple_unary_op_ck>(self_offset);
  self->m_fields.resize(field_count);
  for (intptr_t i = 0; i < field_count; ++i) {
    self = ckb->get_at<nd::tuple_unary_op_ck>(self_offset);
    nd::tuple_unary_op_item &field = self->m_fields[i];
    field.child_kernel_offset = ckb->size() - self_offset;
    field.dst_data_offset = dst_offsets[i];
    field.src_data_offset = src_offsets[i];
    nd::array error_mode = ndt::traits<assign_error_mode>::na();
    const_cast<nd::base_callable *>(af)->instantiate(NULL, NULL, ckb, dst_tp[i], dst_arrmeta[i], 1, &src_tp[i],
                                                     &src_arrmeta[i], kernel_request_single, 1, &error_mode,
                                                     std::map<std::string, ndt::type>());
  }
}
