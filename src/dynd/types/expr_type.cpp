//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/expr_type.hpp>
#include <dynd/shortvector.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/builtin_type_properties.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>

using namespace std;
using namespace dynd;

ndt::expr_type::expr_type(const type &value_type, const type &operand_type,
                          const expr_kernel_generator *kgen)
    : base_expr_type(
          expr_type_id, expr_kind, operand_type.get_data_size(),
          operand_type.get_data_alignment(),
          inherited_flags(value_type.get_flags(), operand_type.get_flags()),
          operand_type.get_arrmeta_size(), value_type.get_ndim()),
      m_value_type(value_type), m_operand_type(operand_type), m_kgen(kgen)
{
  if (operand_type.get_type_id() != tuple_type_id) {
    stringstream ss;
    ss << "expr_type can only be constructed with a tuple as its operand, "
          "given ";
    ss << operand_type;
    throw runtime_error(ss.str());
  }
  const tuple_type *fsd = operand_type.extended<tuple_type>();
  size_t field_count = fsd->get_field_count();
  if (field_count == 1) {
    throw runtime_error("expr_type is for 2 or more operands, use "
                        "unary_expr_type for 1 operand");
  }
  for (size_t i = 0; i != field_count; ++i) {
    const type &ft = fsd->get_field_type(i);
    if (ft.get_type_id() != pointer_type_id) {
      stringstream ss;
      ss << "each field of the expr_type's operand must be a pointer, field "
         << i;
      ss << " is " << ft;
      throw runtime_error(ss.str());
    }
  }
}

ndt::expr_type::~expr_type()
{
  expr_kernel_generator_decref(m_kgen);
}

void ndt::expr_type::print_data(std::ostream &DYND_UNUSED(o),
                                const char *DYND_UNUSED(arrmeta),
                                const char *DYND_UNUSED(data)) const
{
  throw runtime_error(
      "internal error: expr_type::print_data isn't supposed to be called");
}

void ndt::expr_type::print_type(std::ostream &o) const
{
  const tuple_type *fsd = m_operand_type.extended<tuple_type>();
  size_t field_count = fsd->get_field_count();
  o << "expr<";
  o << m_value_type;
  for (size_t i = 0; i != field_count; ++i) {
    const pointer_type *pd =
        static_cast<const pointer_type *>(fsd->get_field_type(i).extended());
    o << ", op" << i << "=" << pd->get_target_type();
  }
  o << ", expr=";
  m_kgen->print_type(o);
  o << ">";
}

ndt::type
ndt::expr_type::apply_linear_index(intptr_t nindices, const irange *indices,
                                   size_t current_i, const type &root_tp,
                                   bool DYND_UNUSED(leading_dimension)) const
{
  if (m_kgen->is_elwise()) {
    intptr_t undim = get_ndim();
    const tuple_type *fsd = m_operand_type.extended<tuple_type>();
    size_t field_count = fsd->get_field_count();

    type result_value_dt = m_value_type.apply_linear_index(
        nindices, indices, current_i, root_tp, true);
    vector<type> result_src_dt(field_count);
    // Apply the portion of the indexing to each of the src operand types
    for (size_t i = 0; i != field_count; ++i) {
      const type &dt = fsd->get_field_type(i);
      intptr_t field_undim = dt.get_ndim();
      if (nindices + field_undim <= undim) {
        result_src_dt[i] = dt;
      } else {
        size_t index_offset = undim - field_undim;
        result_src_dt[i] = dt.apply_linear_index(nindices - index_offset,
                                                 indices + index_offset,
                                                 current_i, root_tp, false);
      }
    }
    type result_operand_type = ndt::tuple_type::make(result_src_dt);
    expr_kernel_generator_incref(m_kgen);
    return make_expr(result_value_dt, result_operand_type, m_kgen);
  } else {
    throw runtime_error("expr_type::apply_linear_index is only implemented for "
                        "elwise kernel generators");
  }
}

intptr_t ndt::expr_type::apply_linear_index(
    intptr_t nindices, const irange *indices, const char *arrmeta,
    const type &result_tp, char *out_arrmeta,
    memory_block_data *embedded_reference, size_t current_i,
    const type &root_tp, bool DYND_UNUSED(leading_dimension),
    char **DYND_UNUSED(inout_data),
    memory_block_data **DYND_UNUSED(inout_dataref)) const
{
  if (m_kgen->is_elwise()) {
    intptr_t undim = get_ndim();
    const expr_type *out_ed = result_tp.extended<expr_type>();
    const tuple_type *fsd = m_operand_type.extended<tuple_type>();
    const tuple_type *out_fsd =
        static_cast<const tuple_type *>(out_ed->m_operand_type.extended());
    const size_t *arrmeta_offsets = fsd->get_arrmeta_offsets_raw();
    const size_t *out_arrmeta_offsets = out_fsd->get_arrmeta_offsets_raw();
    size_t field_count = fsd->get_field_count();
    // Apply the portion of the indexing to each of the src operand types
    for (size_t i = 0; i != field_count; ++i) {
      const pointer_type *pd = fsd->get_field_type(i).extended<pointer_type>();
      intptr_t field_undim = pd->get_ndim();
      if (nindices + field_undim <= undim) {
        pd->arrmeta_copy_construct(out_arrmeta + out_arrmeta_offsets[i],
                                   arrmeta + arrmeta_offsets[i],
                                   embedded_reference);
      } else {
        size_t index_offset = undim - field_undim;
        intptr_t offset = pd->apply_linear_index(
            nindices - index_offset, indices + index_offset,
            arrmeta + arrmeta_offsets[i], out_fsd->get_field_type(i),
            out_arrmeta + out_arrmeta_offsets[i], embedded_reference, current_i,
            root_tp, false, NULL, NULL);
        if (offset != 0) {
          throw runtime_error(
              "internal error: expr_type::apply_linear_index"
              " expected 0 offset from pointer_type::apply_linear_index");
        }
      }
    }
    return 0;
  } else {
    throw runtime_error("expr_type::apply_linear_index is only implemented for "
                        "elwise kernel generators");
  }
}

void ndt::expr_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                               const char *arrmeta,
                               const char *DYND_UNUSED(data)) const
{
  intptr_t undim = get_ndim();
  // Initialize the shape to all ones
  dimvector bcast_shape(undim);
  for (intptr_t j = 0; j < undim; ++j) {
    bcast_shape[j] = 1;
  }

  // Get each operand shape, and broadcast them together
  dimvector shape(undim);
  const tuple_type *fsd = m_operand_type.extended<tuple_type>();
  const uintptr_t *arrmeta_offsets = fsd->get_arrmeta_offsets_raw();
  size_t field_count = fsd->get_field_count();
  for (size_t fi = 0; fi != field_count; ++fi) {
    const type &dt = fsd->get_field_type(fi);
    size_t field_undim = dt.get_ndim();
    if (field_undim > 0) {
      dt.extended()->get_shape(field_undim, 0, shape.get(),
                               arrmeta ? (arrmeta + arrmeta_offsets[fi]) : NULL,
                               NULL);
      incremental_broadcast(undim, bcast_shape.get(), field_undim, shape.get());
    }
  }

  // Copy this shape to the output
  memcpy(out_shape + i, bcast_shape.get(),
         min(undim, ndim - i) * sizeof(intptr_t));

  // If more shape is requested, get it from the value type
  if (ndim - i > undim) {
    const type &dt = m_value_type.get_dtype();
    if (!dt.is_builtin()) {
      dt.extended()->get_shape(ndim, i + undim, out_shape, NULL, NULL);
    } else {
      stringstream ss;
      ss << "requested too many dimensions from type " << type(this, true);
      throw runtime_error(ss.str());
    }
  }
}

bool
ndt::expr_type::is_lossless_assignment(const type &DYND_UNUSED(dst_tp),
                                       const type &DYND_UNUSED(src_tp)) const
{
  return false;
}

bool ndt::expr_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != expr_type_id) {
    return false;
  } else {
    const expr_type *dt = static_cast<const expr_type *>(&rhs);
    return m_value_type == dt->m_value_type &&
           m_operand_type == dt->m_operand_type && m_kgen == dt->m_kgen;
  }
}

namespace {
template <int N>
struct expr_type_offset_applier_extra
    : nd::base_kernel<expr_type_offset_applier_extra<N>, 1> {
  typedef expr_type_offset_applier_extra<N> extra_type;

  size_t offsets[N];

  // Only the single kernel is needed for this one
  void single(char *dst, char *const *src)
  {
    char *src_modified[N];
    for (int i = 0; i < N; ++i) {
      src_modified[i] = src[i] + offsets[i];
    }
    ckernel_prefix *echild = this->get_child_ckernel();
    expr_single_t opchild = echild->get_function<expr_single_t>();
    opchild(echild, dst, src_modified);
  }

  static void destruct(ckernel_prefix *self)
  {
    self->get_child_ckernel(sizeof(extra_type))->destroy();
  }
};

struct expr_type_offset_applier_general_extra
    : nd::base_kernel<expr_type_offset_applier_general_extra, 1> {
  typedef expr_type_offset_applier_general_extra extra_type;

  size_t src_count;
  // After this are src_count size_t offsets

  // Only the single kernel is needed for this one
  void single(char *dst, char *const *src)
  {
    const size_t *offsets = reinterpret_cast<const size_t *>(this + 1);
    shortvector<char *> src_modified(src_count);
    for (size_t i = 0; i != src_count; ++i) {
      src_modified[i] = src[i] + offsets[i];
    }
    ckernel_prefix *echild = this->get_child_ckernel(
        sizeof(extra_type) + src_count * sizeof(size_t));
    expr_single_t opchild = echild->get_function<expr_single_t>();
    opchild(echild, dst, src_modified.get());
  }

  static void destruct(ckernel_prefix *self)
  {
    extra_type *e = reinterpret_cast<extra_type *>(self);
    self->get_child_ckernel(sizeof(extra_type) + e->src_count * sizeof(size_t))
        ->destroy();
  }
};
} // anonymous namespace

static size_t make_expr_type_offset_applier(void *ckb, kernel_request_t kernreq,
                                            intptr_t ckb_offset,
                                            size_t src_count,
                                            const intptr_t *src_data_offsets)
{
  // A few specializations with fixed size, and a general case version
  // NOTE: src_count == 1 must never happen here, it is handled by the
  // unary_expr type
  switch (src_count) {
  case 2: {
    expr_type_offset_applier_extra<2> *e =
        expr_type_offset_applier_extra<2>::make(ckb, kernreq, ckb_offset);
    memcpy(e->offsets, src_data_offsets, sizeof(e->offsets));
    return ckb_offset;
  }
  case 3: {
    expr_type_offset_applier_extra<3> *e =
        expr_type_offset_applier_extra<3>::make(ckb, kernreq, ckb_offset);
    memcpy(e->offsets, src_data_offsets, sizeof(e->offsets));
    return ckb_offset;
  }
  case 4: {
    expr_type_offset_applier_extra<4> *e =
        expr_type_offset_applier_extra<4>::make(ckb, kernreq, ckb_offset);
    memcpy(e->offsets, src_data_offsets, sizeof(e->offsets));
    return ckb_offset;
  }
  default: {
    expr_type_offset_applier_general_extra *e =
        expr_type_offset_applier_general_extra::make(ckb, kernreq, ckb_offset);
    e->src_count = src_count;
    size_t *out_offsets = reinterpret_cast<size_t *>(e + 1);
    memcpy(out_offsets, src_data_offsets, src_count * sizeof(size_t));
    return ckb_offset;
  }
  }
}

static void src_deref_single(ckernel_prefix *self, char *dst, char *const *src)
{
  ckernel_prefix *child = self->get_child_ckernel(sizeof(ckernel_prefix));
  expr_single_t child_fn = child->get_function<expr_single_t>();
  child_fn(child, dst, reinterpret_cast<char *const *>(*src));
}

static size_t make_src_deref_ckernel(void *ckb, intptr_t ckb_offset)
{
  ckernel_prefix *self =
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
          ->alloc_ck<ckernel_prefix>(ckb_offset);
  self->set_function<expr_single_t>(&src_deref_single);
  self->destructor = &kernels::destroy_trivial_parent_ckernel;
  return ckb_offset;
}

size_t ndt::expr_type::make_operand_to_value_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
    const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx) const
{
  const tuple_type *fsd = m_operand_type.extended<tuple_type>();

  ckb_offset = make_src_deref_ckernel(ckb, ckb_offset);
  size_t input_count = fsd->get_field_count();
  const uintptr_t *arrmeta_offsets = fsd->get_arrmeta_offsets_raw();
  shortvector<const char *> src_arrmeta_array(input_count);
  dimvector src_data_offsets(input_count);
  bool nonzero_offsets = false;

  vector<type> src_dt(input_count);
  for (size_t i = 0; i != input_count; ++i) {
    const pointer_type *pd =
        static_cast<const pointer_type *>(fsd->get_field_type(i).extended());
    src_dt[i] = pd->get_target_type();
  }
  for (size_t i = 0; i != input_count; ++i) {
    const char *ptr_arrmeta = src_arrmeta + arrmeta_offsets[i];
    intptr_t offset =
        reinterpret_cast<const pointer_type_arrmeta *>(ptr_arrmeta)->offset;
    if (offset != 0) {
      nonzero_offsets = true;
    }
    src_data_offsets[i] = offset;
    src_arrmeta_array[i] = ptr_arrmeta + sizeof(pointer_type_arrmeta);
  }
  // If there were any non-zero pointer offsets, we need to add a kernel
  // adapter which applies those offsets.
  if (nonzero_offsets) {
    ckb_offset = make_expr_type_offset_applier(
        ckb, kernreq, ckb_offset, input_count, src_data_offsets.get());
  }
  return m_kgen->make_expr_kernel(
      ckb, ckb_offset, m_value_type, dst_arrmeta, input_count, &src_dt[0],
      src_arrmeta_array.get(), kernel_request_single, ectx);
}

size_t ndt::expr_type::make_value_to_operand_assignment_kernel(
    void *DYND_UNUSED(ckb), intptr_t DYND_UNUSED(ckb_offset),
    const char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    kernel_request_t DYND_UNUSED(kernreq),
    const eval::eval_context *DYND_UNUSED(ectx)) const
{
  throw runtime_error("Cannot assign to a dynd expr object value");
}

ndt::type ndt::expr_type::with_replaced_storage_type(
    const type &DYND_UNUSED(replacement_type)) const
{
  throw runtime_error("TODO: implement expr_type::with_replaced_storage_type");
}

void ndt::expr_type::get_dynamic_array_properties(
    const std::pair<std::string, gfunc::callable> **out_properties,
    size_t *out_count) const
{
  const type &udt = m_value_type.get_dtype();
  if (!udt.is_builtin()) {
    udt.extended()->get_dynamic_array_properties(out_properties, out_count);
  } else {
    get_builtin_type_dynamic_array_properties(udt.get_type_id(), out_properties,
                                              out_count);
  }
}

void ndt::expr_type::get_dynamic_array_functions(
    const std::pair<std::string, gfunc::callable> **out_functions,
    size_t *out_count) const
{
  const type &udt = m_value_type.get_dtype();
  if (!udt.is_builtin()) {
    udt.extended()->get_dynamic_array_functions(out_functions, out_count);
  } else {
    // get_builtin_type_dynamic_array_functions(udt.get_type_id(),
    // out_functions, out_count);
  }
}
