//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/bytes_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/bytes_assignment_kernels.hpp>
#include <dynd/types/fixed_bytes_type.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/func/apply.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

ndt::bytes_type::bytes_type(size_t alignment)
    : base_bytes_type(bytes_type_id, bytes_kind, sizeof(bytes), alignof(bytes),
                      type_flag_zeroinit | type_flag_destructor, 0),
      m_alignment(alignment)
{
  if (alignment != 1 && alignment != 2 && alignment != 4 && alignment != 8 && alignment != 16) {
    std::stringstream ss;
    ss << "Cannot make a dynd bytes type with alignment " << alignment << ", it must be a small power of two";
    throw std::runtime_error(ss.str());
  }
}

ndt::bytes_type::~bytes_type()
{
}

void ndt::bytes_type::get_bytes_range(const char **out_begin, const char **out_end, const char *DYND_UNUSED(arrmeta),
                                      const char *data) const
{
  *out_begin = reinterpret_cast<const bytes_type_data *>(data)->begin();
  *out_end = reinterpret_cast<const bytes_type_data *>(data)->end();
}

void ndt::bytes_type::set_bytes_data(const char *DYND_UNUSED(arrmeta), char *data, const char *bytes_begin,
                                     const char *bytes_end) const
{
  bytes_type_data *d = reinterpret_cast<bytes_type_data *>(data);
  if (d->begin() != NULL) {
    throw runtime_error("assigning to a bytes data element requires that it be "
                        "initialized to NULL");
  }

  // Allocate the output array data, then copy it
  d->assign(bytes_begin, bytes_end - bytes_begin);
}

void ndt::bytes_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
  const char *begin = reinterpret_cast<const char *const *>(data)[0];
  const char *end = reinterpret_cast<const char *const *>(data)[1];

  // Print as hexadecimal
  hexadecimal_print_summarized(o, begin, end - begin, 80);
}

void ndt::bytes_type::print_type(std::ostream &o) const
{
  o << "bytes";
  if (m_alignment != 1) {
    o << "[align=" << m_alignment << "]";
  }
}

bool ndt::bytes_type::is_unique_data_owner(const char *DYND_UNUSED(arrmeta)) const
{
  return true;
}

ndt::type ndt::bytes_type::get_canonical_type() const
{
  return type(this, true);
}

void ndt::bytes_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *DYND_UNUSED(arrmeta),
                                const char *data) const
{
  if (data == NULL) {
    out_shape[i] = -1;
  } else {
    const bytes_type_data *d = reinterpret_cast<const bytes_type_data *>(data);
    out_shape[i] = d->end() - d->begin();
  }
  if (i + 1 < ndim) {
    stringstream ss;
    ss << "requested too many dimensions from type " << type(this, true);
    throw runtime_error(ss.str());
  }
}

bool ndt::bytes_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.get_kind() == bytes_kind) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

intptr_t ndt::bytes_type::make_assignment_kernel(void *ckb, intptr_t ckb_offset, const type &dst_tp,
                                                 const char *dst_arrmeta, const type &src_tp, const char *src_arrmeta,
                                                 kernel_request_t kernreq, const eval::eval_context *ectx) const
{
  if (this == dst_tp.extended()) {
    switch (src_tp.get_type_id()) {
    case bytes_type_id: {
      return make_blockref_bytes_assignment_kernel(ckb, ckb_offset, get_data_alignment(), dst_arrmeta,
                                                   src_tp.get_data_alignment(), src_arrmeta, kernreq, ectx);
    }
    case fixed_bytes_type_id: {
      return make_fixed_bytes_to_blockref_bytes_assignment_kernel(ckb, ckb_offset, get_data_alignment(), dst_arrmeta,
                                                                  src_tp.get_data_size(), src_tp.get_data_alignment(),
                                                                  kernreq, ectx);
    }
    default: {
      if (!src_tp.is_builtin()) {
        src_tp.extended()->make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta, kernreq,
                                                  ectx);
      }
      break;
    }
    }
  }

  stringstream ss;
  ss << "Cannot assign from " << src_tp << " to " << dst_tp;
  throw runtime_error(ss.str());
}

bool ndt::bytes_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != bytes_type_id) {
    return false;
  } else {
    const bytes_type *dt = static_cast<const bytes_type *>(&rhs);
    return m_alignment == dt->m_alignment;
  }
}

void ndt::bytes_type::data_destruct(const char *DYND_UNUSED(arrmeta), char *data) const
{
  reinterpret_cast<bytes *>(data)->~bytes();
}

void ndt::bytes_type::data_destruct_strided(const char *DYND_UNUSED(arrmeta), char *data, intptr_t stride,
                                            size_t count) const
{
  for (size_t i = 0; i != count; ++i) {
    reinterpret_cast<bytes *>(data)->~bytes();
    data += stride;
  }
}

void ndt::bytes_type::get_dynamic_type_properties(const std::pair<std::string, nd::callable> **out_properties,
                                                  size_t *out_count) const
{
  static pair<std::string, nd::callable> type_properties[] = {pair<std::string, nd::callable>(
      "target_alignment",
      nd::functional::apply([](type self) { return self.extended<bytes_type>()->get_target_alignment(); }, "self"))};

  *out_properties = type_properties;
  *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}
