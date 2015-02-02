//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/fixed_dimsym_type.hpp>
#include <dynd/types/pow_dimsym_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/func/make_callable.hpp>
#include <dynd/types/builtin_type_properties.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>

using namespace std;
using namespace dynd;

fixed_dimsym_type::fixed_dimsym_type(const ndt::type &element_tp)
    : base_dim_type(fixed_dimsym_type_id, element_tp, 0,
                    element_tp.get_data_alignment(), sizeof(size_stride_t),
                    type_flag_symbolic, true)
{
  // Propagate the inherited flags from the element
  m_members.flags |=
      (element_tp.get_flags() &
       ((type_flags_operand_inherited | type_flags_value_inherited) &
        ~type_flag_scalar));
}

fixed_dimsym_type::~fixed_dimsym_type()
{
}

size_t fixed_dimsym_type::get_default_data_size() const
{
  stringstream ss;
  ss << "Cannot get default data size of type " << ndt::type(this, true);
  throw runtime_error(ss.str());
}

void fixed_dimsym_type::print_data(std::ostream &DYND_UNUSED(o),
                                    const char *DYND_UNUSED(arrmeta),
                                    const char *DYND_UNUSED(data)) const
{
  throw type_error("Cannot store data of symbolic fixed_dim type");
}

void fixed_dimsym_type::print_type(std::ostream &o) const
{
  o << "Fixed * " << m_element_tp;
}

bool fixed_dimsym_type::is_expression() const
{
  return m_element_tp.is_expression();
}

bool fixed_dimsym_type::is_unique_data_owner(const char *DYND_UNUSED(arrmeta)) const
{
  return false;
}

void fixed_dimsym_type::transform_child_types(type_transform_fn_t transform_fn,
                                               intptr_t arrmeta_offset,
                                               void *extra,
                                               ndt::type &out_transformed_tp,
                                               bool &out_was_transformed) const
{
  ndt::type tmp_tp;
  bool was_transformed = false;
  transform_fn(m_element_tp, arrmeta_offset, extra, tmp_tp, was_transformed);
  if (was_transformed) {
    out_transformed_tp = ndt::type(new fixed_dimsym_type(tmp_tp), false);
    out_was_transformed = true;
  }
  else {
    out_transformed_tp = ndt::type(this, true);
  }
}

ndt::type fixed_dimsym_type::get_canonical_type() const
{
  return ndt::type(new fixed_dimsym_type(m_element_tp.get_canonical_type()),
                   false);
}

ndt::type
fixed_dimsym_type::at_single(intptr_t DYND_UNUSED(i0),
                              const char **DYND_UNUSED(inout_arrmeta),
                              const char **DYND_UNUSED(inout_data)) const
{
  return m_element_tp;
}

ndt::type
fixed_dimsym_type::get_type_at_dimension(char **DYND_UNUSED(inout_arrmeta),
                                         intptr_t i, intptr_t total_ndim) const
{
  if (i == 0) {
    return ndt::type(this, true);
  }
  else {
    return m_element_tp.get_type_at_dimension(NULL, i - 1, total_ndim + 1);
  }
}

intptr_t fixed_dimsym_type::get_dim_size(const char *DYND_UNUSED(arrmeta),
                                          const char *DYND_UNUSED(data)) const
{
  return -1;
}

void fixed_dimsym_type::get_shape(intptr_t ndim, intptr_t i,
                                   intptr_t *out_shape,
                                   const char *DYND_UNUSED(arrmeta),
                                   const char *DYND_UNUSED(data)) const
{
  out_shape[i] = -1;

  // Process the later shape values
  if (i + 1 < ndim) {
    if (!m_element_tp.is_builtin()) {
      m_element_tp.extended()->get_shape(ndim, i + 1, out_shape, NULL, NULL);
    }
    else {
      stringstream ss;
      ss << "requested too many dimensions from type " << ndt::type(this, true);
      throw runtime_error(ss.str());
    }
  }
}

bool fixed_dimsym_type::is_lossless_assignment(const ndt::type &DYND_UNUSED(dst_tp),
                                                const ndt::type &DYND_UNUSED(src_tp)) const
{
  return false;
}

bool fixed_dimsym_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_type_id() != fixed_dimsym_type_id) {
    return false;
  }
  else {
    const fixed_dimsym_type *dt =
        static_cast<const fixed_dimsym_type *>(&rhs);
    return m_element_tp == dt->m_element_tp;
  }
}

void fixed_dimsym_type::arrmeta_default_construct(
    char *DYND_UNUSED(arrmeta), bool DYND_UNUSED(blockref_alloc)) const
{
  stringstream ss;
  ss << "Cannot default construct arrmeta for symbolic type "
     << ndt::type(this, true);
  throw runtime_error(ss.str());
}

void fixed_dimsym_type::arrmeta_copy_construct(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
  stringstream ss;
  ss << "Cannot copy construct arrmeta for symbolic type "
     << ndt::type(this, true);
  throw runtime_error(ss.str());
}

size_t fixed_dimsym_type::arrmeta_copy_construct_onedim(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
  stringstream ss;
  ss << "Cannot copy construct arrmeta for symbolic type "
     << ndt::type(this, true);
  throw runtime_error(ss.str());
}

void fixed_dimsym_type::arrmeta_reset_buffers(char *DYND_UNUSED(arrmeta)) const
{
}

void
fixed_dimsym_type::arrmeta_finalize_buffers(char *DYND_UNUSED(arrmeta)) const
{
}

void fixed_dimsym_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {}

void fixed_dimsym_type::arrmeta_debug_print(
    const char *DYND_UNUSED(arrmeta), std::ostream &DYND_UNUSED(o),
    const std::string &DYND_UNUSED(indent)) const
{
  stringstream ss;
  ss << "Cannot have arrmeta for symbolic type " << ndt::type(this, true);
  throw runtime_error(ss.str());
}

void fixed_dimsym_type::data_destruct(const char *DYND_UNUSED(arrmeta),
                                       char *DYND_UNUSED(data)) const
{
  stringstream ss;
  ss << "Cannot have data for symbolic type " << ndt::type(this, true);
  throw runtime_error(ss.str());
}

void fixed_dimsym_type::data_destruct_strided(const char *DYND_UNUSED(arrmeta),
                                               char *DYND_UNUSED(data),
                                               intptr_t DYND_UNUSED(stride),
                                               size_t DYND_UNUSED(count)) const
{
  stringstream ss;
  ss << "Cannot have data for symbolic type " << ndt::type(this, true);
  throw runtime_error(ss.str());
}

bool fixed_dimsym_type::matches(const char *arrmeta, const ndt::type &other,
                                std::map<nd::string, ndt::type> &tp_vars) const
{
  switch (other.get_type_id()) {
    case fixed_dim_type_id:
    case cfixed_dim_type_id:
    case fixed_dimsym_type_id:
      return m_element_tp.matches(arrmeta, other.extended<base_dim_type>()->get_element_type(), tp_vars);
    case any_sym_type_id:
      return true;
    case pow_dimsym_type_id:
      return matches(arrmeta, other.extended<pow_dimsym_type>()->get_base_type(), tp_vars);
    default:
      return false;
  }
}

static ndt::type get_element_type(const ndt::type& dt) {
    return dt.extended<fixed_dimsym_type>()->get_element_type();
}

void fixed_dimsym_type::get_dynamic_type_properties(
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    static pair<string, gfunc::callable> fixed_dimsym_type_properties[] = {
        pair<string, gfunc::callable>(
            "element_type", gfunc::make_callable(&::get_element_type, "self"))};

    *out_properties = fixed_dimsym_type_properties;
    *out_count = sizeof(fixed_dimsym_type_properties) / sizeof(fixed_dimsym_type_properties[0]);
}

void fixed_dimsym_type::get_dynamic_array_properties(
    const std::pair<std::string, gfunc::callable> **out_properties,
    size_t *out_count) const
{
    if (m_element_tp.is_builtin()) {
        get_builtin_type_dynamic_array_properties(m_element_tp.get_type_id(),
                                                  out_properties, out_count);
    } else {
        m_element_tp.extended()->get_dynamic_array_properties(out_properties, out_count);
    }
}

void fixed_dimsym_type::get_dynamic_array_functions(
    const std::pair<std::string, gfunc::callable> **out_functions,
    size_t *out_count) const
{
    if (m_element_tp.is_builtin()) {
        // TODO
    } else {
        m_element_tp.extended()->get_dynamic_array_functions(out_functions, out_count);
    }
}

namespace {
    // TODO: use the PP meta stuff, but DYND_PP_LEN_MAX is set to 8 right now,
    // would need to be 19
    struct static_strided_dims {
        fixed_dimsym_type bt1;
        fixed_dimsym_type bt2;
        fixed_dimsym_type bt3;
        fixed_dimsym_type bt4;
        fixed_dimsym_type bt5;
        fixed_dimsym_type bt6;
        fixed_dimsym_type bt7;
        fixed_dimsym_type bt8;
        fixed_dimsym_type bt9;
        fixed_dimsym_type bt10;
        fixed_dimsym_type bt11;
        fixed_dimsym_type bt12;
        fixed_dimsym_type bt13;
        fixed_dimsym_type bt14;
        fixed_dimsym_type bt15;
        fixed_dimsym_type bt16;
        fixed_dimsym_type bt17;
        fixed_dimsym_type bt18;

        ndt::type static_builtins_instance[builtin_type_id_count];

        static_strided_dims()
            : bt1(ndt::type((type_id_t)1)),
              bt2(ndt::type((type_id_t)2)),
              bt3(ndt::type((type_id_t)3)),
              bt4(ndt::type((type_id_t)4)),
              bt5(ndt::type((type_id_t)5)),
              bt6(ndt::type((type_id_t)6)),
              bt7(ndt::type((type_id_t)7)),
              bt8(ndt::type((type_id_t)8)),
              bt9(ndt::type((type_id_t)9)),
              bt10(ndt::type((type_id_t)10)),
              bt11(ndt::type((type_id_t)11)),
              bt12(ndt::type((type_id_t)12)),
              bt13(ndt::type((type_id_t)13)),
              bt14(ndt::type((type_id_t)14)),
              bt15(ndt::type((type_id_t)15)),
              bt16(ndt::type((type_id_t)16)),
              bt17(ndt::type((type_id_t)17)),
              bt18(ndt::type((type_id_t)18))
        {
            static_builtins_instance[1] = ndt::type(&bt1, true);
            static_builtins_instance[2] = ndt::type(&bt2, true);
            static_builtins_instance[3] = ndt::type(&bt3, true);
            static_builtins_instance[4] = ndt::type(&bt4, true);
            static_builtins_instance[5] = ndt::type(&bt5, true);
            static_builtins_instance[6] = ndt::type(&bt6, true);
            static_builtins_instance[7] = ndt::type(&bt7, true);
            static_builtins_instance[8] = ndt::type(&bt8, true);
            static_builtins_instance[9] = ndt::type(&bt9, true);
            static_builtins_instance[10] = ndt::type(&bt10, true);
            static_builtins_instance[11] = ndt::type(&bt11, true);
            static_builtins_instance[12] = ndt::type(&bt12, true);
            static_builtins_instance[13] = ndt::type(&bt13, true);
            static_builtins_instance[14] = ndt::type(&bt14, true);
            static_builtins_instance[15] = ndt::type(&bt15, true);
            static_builtins_instance[16] = ndt::type(&bt16, true);
            static_builtins_instance[17] = ndt::type(&bt17, true);
            static_builtins_instance[18] = ndt::type(&bt18, true);
        }
    };
} // anonymous namespace

ndt::type ndt::make_fixed_dimsym(const ndt::type &element_tp)
{
  // Static instances of the types, which have a reference
  // count > 0 for the lifetime of the program. This static
  // construction is inside a function to ensure correct creation
  // order during startup.
  static static_strided_dims ssd;

  if (element_tp.is_builtin()) {
    return ssd.static_builtins_instance[element_tp.get_type_id()];
  }
  else {
    return ndt::type(new fixed_dimsym_type(element_tp), false);
  }
}
