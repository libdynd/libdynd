//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

ndt::type_type::type_type()
    : base_type(type_type_id, type_kind, sizeof(const base_type *), sizeof(const base_type *),
                type_flag_zeroinit | type_flag_destructor, 0, 0, 0)
{
}

ndt::type_type::type_type(const type &pattern_tp)
    : base_type(type_type_id, type_kind, sizeof(const base_type *), sizeof(const base_type *),
                type_flag_zeroinit | type_flag_destructor, 0, 0, 0),
      m_pattern_tp(pattern_tp)
{
  if (!m_pattern_tp.is_symbolic()) {
    throw type_error("type_type must have a symbolic type for a pattern");
  }
}

ndt::type_type::~type_type()
{
}

void ndt::type_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
  const type_type_data *ddd = reinterpret_cast<const type_type_data *>(data);
  // This tests avoids the atomic increment/decrement of
  // always constructing a type object
  if (is_builtin_type(ddd->tp)) {
    o << type(ddd->tp, true);
  } else {
    ddd->tp->print_type(o);
  }
}

void ndt::type_type::print_type(std::ostream &o) const
{
  o << "type";
  if (!m_pattern_tp.is_null()) {
    o << " | " << m_pattern_tp;
  }
}

bool ndt::type_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != type_type_id) {
    return false;
  } else {
    return m_pattern_tp == static_cast<const type_type *>(&rhs)->m_pattern_tp;
  }
}

void ndt::type_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta), bool DYND_UNUSED(blockref_alloc)) const
{
}

void ndt::type_type::arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
                                            memory_block_data *DYND_UNUSED(embedded_reference)) const
{
}

void ndt::type_type::arrmeta_reset_buffers(char *DYND_UNUSED(arrmeta)) const
{
}

void ndt::type_type::arrmeta_finalize_buffers(char *DYND_UNUSED(arrmeta)) const
{
}

void ndt::type_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const
{
}

void ndt::type_type::data_destruct(const char *DYND_UNUSED(arrmeta), char *data) const
{
  const base_type *bd = reinterpret_cast<type_type_data *>(data)->tp;
  if (!is_builtin_type(bd)) {
    base_type_decref(bd);
  }
}

void ndt::type_type::data_destruct_strided(const char *DYND_UNUSED(arrmeta), char *data, intptr_t stride,
                                           size_t count) const
{
  for (size_t i = 0; i != count; ++i, data += stride) {
    const base_type *bd = reinterpret_cast<type_type_data *>(data)->tp;
    if (!is_builtin_type(bd)) {
      base_type_decref(bd);
    }
  }
}

namespace {

struct typed_data_assignment_kernel : nd::base_kernel<typed_data_assignment_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    // Free the destination reference
    base_type_xdecref(reinterpret_cast<const type_type_data *>(dst)->tp);
    // Copy the pointer and count the reference
    const ndt::base_type *bd = (*reinterpret_cast<type_type_data *const *>(src))->tp;
    reinterpret_cast<type_type_data *>(dst)->tp = bd;
    base_type_xincref(bd);
  }
};

struct string_to_type_kernel : nd::base_kernel<string_to_type_kernel, 1> {
  const ndt::base_string_type *src_string_dt;
  const char *src_arrmeta;
  assign_error_mode errmode;

  ~string_to_type_kernel()
  {
    base_type_xdecref(src_string_dt);
  }

  void single(char *dst, char *const *src)
  {
    const std::string &s = src_string_dt->get_utf8_string(src_arrmeta, src[0], errmode);
    ndt::type(s).swap(reinterpret_cast<type_type_data *>(dst)->tp);
  }
};

struct type_to_string_kernel : nd::base_kernel<type_to_string_kernel, 1> {
  const ndt::base_string_type *dst_string_dt;
  const char *dst_arrmeta;
  eval::eval_context ectx;

  ~type_to_string_kernel()
  {
    base_type_xdecref(dst_string_dt);
  }

  void single(char *dst, char *const *src)
  {
    const ndt::base_type *bd = (*reinterpret_cast<type_type_data *const *>(src))->tp;
    stringstream ss;
    if (is_builtin_type(bd)) {
      ss << ndt::type(bd, true);
    } else {
      bd->print_type(ss);
    }
    dst_string_dt->set_from_utf8_string(dst_arrmeta, dst, ss.str(), &ectx);
  }
};

} // anonymous namespace

intptr_t ndt::type_type::make_assignment_kernel(void *ckb, intptr_t ckb_offset, const type &dst_tp,
                                                const char *dst_arrmeta, const type &src_tp, const char *src_arrmeta,
                                                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
  if (this == dst_tp.extended()) {
    if (src_tp.get_type_id() == type_type_id) {
      typed_data_assignment_kernel::make(ckb, kernreq, ckb_offset);
      return ckb_offset;
    } else if (src_tp.get_kind() == string_kind) {
      // String to type
      string_to_type_kernel *e = string_to_type_kernel::make(ckb, kernreq, ckb_offset);
      // The kernel data owns a reference to this type
      e->src_string_dt = static_cast<const base_string_type *>(type(src_tp).release());
      e->src_arrmeta = src_arrmeta;
      e->errmode = ectx->errmode;
      return ckb_offset;
    } else if (!src_tp.is_builtin()) {
      return src_tp.extended()->make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
                                                       kernreq, ectx);
    }
  } else {
    if (dst_tp.get_kind() == string_kind) {
      // Type to string
      type_to_string_kernel *e = type_to_string_kernel::make(ckb, kernreq, ckb_offset);
      // The kernel data owns a reference to this type
      e->dst_string_dt = static_cast<const base_string_type *>(type(dst_tp).release());
      e->dst_arrmeta = dst_arrmeta;
      e->ectx = *ectx;
      return ckb_offset;
    }
  }

  stringstream ss;
  ss << "Cannot assign from " << src_tp << " to " << dst_tp;
  throw dynd::type_error(ss.str());
}
