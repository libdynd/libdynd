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
#include <dynd/kernels/struct_assignment_kernels.hpp>
#include <dynd/func/copy.hpp>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// struct to different struct assignment

size_t dynd::make_struct_assignment_kernel(void *ckb, intptr_t ckb_offset,
                                           const ndt::type &dst_struct_tp,
                                           const char *dst_arrmeta,
                                           const ndt::type &src_struct_tp,
                                           const char *src_arrmeta,
                                           kernel_request_t kernreq,
                                           const eval::eval_context *ectx)
{
  if (src_struct_tp.get_kind() != struct_kind) {
    stringstream ss;
    ss << "struct_type::make_assignment_kernel: provided source type "
       << src_struct_tp << " is not of struct kind";
    throw runtime_error(ss.str());
  }
  if (dst_struct_tp.get_kind() != struct_kind) {
    stringstream ss;
    ss << "struct_type::make_assignment_kernel: provided destination type "
       << dst_struct_tp << " is not of struct kind";
    throw runtime_error(ss.str());
  }
  const ndt::base_struct_type *dst_sd =
      dst_struct_tp.extended<ndt::base_struct_type>();
  const ndt::base_struct_type *src_sd =
      src_struct_tp.extended<ndt::base_struct_type>();
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
  const uintptr_t *src_data_offsets_orig =
      src_sd->get_data_offsets(src_arrmeta);
  vector<ndt::type> src_fields_tp(field_count);
  shortvector<uintptr_t> src_data_offsets(field_count);
  shortvector<const char *> src_fields_arrmeta(field_count);

  // Match up the fields
  for (intptr_t i = 0; i != field_count; ++i) {
    const string &dst_name = dst_sd->get_field_name_raw(i);
    intptr_t src_i = src_sd->get_field_index(dst_name.begin(), dst_name.end());
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
      nd::copy::get().get(), nd::copy::get().get_type(), ckb, ckb_offset,
      field_count, dst_sd->get_data_offsets(dst_arrmeta),
      dst_sd->get_field_types_raw(), dst_fields_arrmeta.get(),
      src_data_offsets.get(), &src_fields_tp[0], src_fields_arrmeta.get(),
      kernreq, ectx);
}
