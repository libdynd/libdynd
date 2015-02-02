//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/arrfunc_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/func/make_callable.hpp>
#include <dynd/ensure_immutable_contig.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>

using namespace std;
using namespace dynd;

static bool is_simple_identifier_name(const char *begin, const char *end)
{
  if (begin == end) {
    return false;
  } else {
    char c = *begin++;
    if (!(('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '_')) {
      return false;
    }
    while (begin < end) {
      c = *begin++;
      if (!(('0' <= c && c <= '9') || ('a' <= c && c <= 'z') ||
            ('A' <= c && c <= 'Z') || c == '_')) {
        return false;
      }
    }
    return true;
  }
}

arrfunc_type::arrfunc_type(const ndt::type &pos_types,
                           const ndt::type &ret_type)
    : base_type(arrfunc_type_id, function_kind, sizeof(arrfunc_type_data),
                scalar_align_of<uint64_t>::value,
                type_flag_scalar | type_flag_zeroinit | type_flag_destructor, 0,
                0, 0),
      m_return_type(ret_type), m_pos_tuple(pos_types)
{
  if (m_pos_tuple.get_type_id() != tuple_type_id) {
    stringstream ss;
    ss << "dynd arrfunc positional arg types require a tuple type, got a "
          "type \"" << m_pos_tuple << "\"";
    throw invalid_argument(ss.str());
  }
  m_kwd_struct =
      ndt::make_empty_struct(m_pos_tuple.extended<tuple_type>()->is_variadic());

  // Note that we don't base the flags of this type on that of its arguments
  // and return types, because it is something the can be instantiated, even
  // for arguments that are symbolic.
}

arrfunc_type::arrfunc_type(const ndt::type &pos_types,
                           const ndt::type &kwd_types,
                           const ndt::type &ret_type)
    : base_type(arrfunc_type_id, function_kind, sizeof(arrfunc_type_data),
                scalar_align_of<uint64_t>::value,
                type_flag_scalar | type_flag_zeroinit | type_flag_destructor, 0,
                0, 0),
      m_return_type(ret_type), m_pos_tuple(pos_types), m_kwd_struct(kwd_types)
{
  if (m_pos_tuple.get_type_id() != tuple_type_id) {
    stringstream ss;
    ss << "dynd arrfunc positional arg types require a tuple type, got a "
          "type \"" << m_pos_tuple << "\"";
    throw invalid_argument(ss.str());
  }
  if (m_kwd_struct.get_type_id() != struct_type_id) {
    stringstream ss;
    ss << "dynd arrfunc keyword arg types require a struct type, got a "
          "type \"" << m_kwd_struct << "\"";
    throw invalid_argument(ss.str());
  }

  for (intptr_t i = 0, i_end = get_nkwd(); i < i_end; ++i) {
    if (m_kwd_struct.extended<tuple_type>()->get_field_type(i).get_type_id() ==
        option_type_id) {
      m_opt_kwd_indices.push_back(i);
    }
  }

  // TODO: Should check that all the kwd names are simple identifier names
  //       because struct_type does not check that.

  // Note that we don't base the flags of this type on that of its arguments
  // and return types, because it is something the can be instantiated, even
  // for arguments that are symbolic.
}

static void print_arrfunc(std::ostream &o,
                          const arrfunc_type *DYND_UNUSED(af_tp),
                          const arrfunc_type_data *af)
{
  o << "<arrfunc at " << (void *)af << ">";
}

void arrfunc_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta),
                              const char *data) const
{
  const arrfunc_type_data *af =
      reinterpret_cast<const arrfunc_type_data *>(data);
  print_arrfunc(o, this, af);
}

void arrfunc_type::print_type(std::ostream &o) const
{
  intptr_t npos = get_npos();
  intptr_t nkwd = get_nkwd();

  o << "(";

  for (intptr_t i = 0; i < npos; ++i) {
    if (i > 0) {
      o << ", ";
    }

    o << get_pos_type(i);
  }
  if (m_pos_tuple.extended<tuple_type>()->is_variadic()) {
    if (npos > 0) {
      o << ", ...";
    } else {
      o << "...";
    }
  }
  for (intptr_t i = 0; i < nkwd; ++i) {
    if (i > 0 || npos > 0) {
      o << ", ";
    }

    // TODO: names should be validated on input, not just
    //       printed specially like in struct_type.
    const string_type_data &an = get_kwd_name_raw(i);
    if (is_simple_identifier_name(an.begin, an.end)) {
      o.write(an.begin, an.end - an.begin);
    } else {
      print_escaped_utf8_string(o, an.begin, an.end, true);
    }
    o << ": " << get_kwd_type(i);
  }
  if (nkwd > 0 && m_kwd_struct.extended<struct_type>()->is_variadic()) {
    o << "...";
  }

  o << ") -> " << m_return_type;
}

void arrfunc_type::transform_child_types(type_transform_fn_t transform_fn,
                                         intptr_t arrmeta_offset, void *extra,
                                         ndt::type &out_transformed_tp,
                                         bool &out_was_transformed) const
{
  ndt::type tmp_return_type, tmp_pos_types, tmp_kwd_types;

  bool was_transformed = false;
  transform_fn(m_return_type, arrmeta_offset, extra, tmp_return_type,
               was_transformed);
  transform_fn(m_pos_tuple, arrmeta_offset, extra, tmp_pos_types,
               was_transformed);
  transform_fn(m_kwd_struct, arrmeta_offset, extra, tmp_kwd_types,
               was_transformed);
  if (was_transformed) {
    out_transformed_tp =
        ndt::make_arrfunc(tmp_pos_types, tmp_kwd_types, tmp_return_type);
    out_was_transformed = true;
  } else {
    out_transformed_tp = ndt::type(this, true);
  }
}

ndt::type arrfunc_type::get_canonical_type() const
{
  ndt::type tmp_return_type, tmp_pos_types, tmp_kwd_types;

  tmp_return_type = m_return_type.get_canonical_type();
  tmp_pos_types = m_pos_tuple.get_canonical_type();
  tmp_kwd_types = m_kwd_struct.get_canonical_type();
  return ndt::make_arrfunc(tmp_pos_types, tmp_kwd_types, tmp_return_type);
}

ndt::type arrfunc_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
    size_t DYND_UNUSED(current_i), const ndt::type &DYND_UNUSED(root_tp),
    bool DYND_UNUSED(leading_dimension)) const
{
  throw type_error("Cannot store data of funcproto type");
}

intptr_t arrfunc_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
    const char *DYND_UNUSED(arrmeta), const ndt::type &DYND_UNUSED(result_tp),
    char *DYND_UNUSED(out_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference),
    size_t DYND_UNUSED(current_i), const ndt::type &DYND_UNUSED(root_tp),
    bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
    memory_block_data **DYND_UNUSED(inout_dataref)) const
{
  throw type_error("Cannot store data of funcproto type");
}

bool arrfunc_type::is_lossless_assignment(const ndt::type &dst_tp,
                                          const ndt::type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    } else if (src_tp.get_type_id() == arrfunc_type_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

bool arrfunc_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != arrfunc_type_id) {
    return false;
  } else {
    const arrfunc_type *fpt = static_cast<const arrfunc_type *>(&rhs);
    return m_return_type == fpt->m_return_type &&
           m_pos_tuple == fpt->m_pos_tuple && m_kwd_struct == fpt->m_kwd_struct;
  }
}

void
arrfunc_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta),
                                        bool DYND_UNUSED(blockref_alloc)) const
{
}

void arrfunc_type::arrmeta_copy_construct(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
}

void arrfunc_type::arrmeta_reset_buffers(char *DYND_UNUSED(arrmeta)) const {}

void arrfunc_type::arrmeta_finalize_buffers(char *DYND_UNUSED(arrmeta)) const {}

void arrfunc_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {}

void arrfunc_type::data_destruct(const char *DYND_UNUSED(arrmeta),
                                 char *data) const
{
  const arrfunc_type_data *d = reinterpret_cast<arrfunc_type_data *>(data);
  d->~arrfunc_type_data();
}

void arrfunc_type::data_destruct_strided(const char *DYND_UNUSED(arrmeta),
                                         char *data, intptr_t stride,
                                         size_t count) const
{
  for (size_t i = 0; i != count; ++i, data += stride) {
    const arrfunc_type_data *d = reinterpret_cast<arrfunc_type_data *>(data);
    d->~arrfunc_type_data();
  }
}

/////////////////////////////////////////
// arrfunc to string assignment

namespace {
struct arrfunc_to_string_ck : public kernels::unary_ck<arrfunc_to_string_ck> {
  ndt::type m_src_tp, m_dst_string_dt;
  const char *m_dst_arrmeta;
  eval::eval_context m_ectx;

  inline void single(char *dst, const char *src)
  {
    const arrfunc_type_data *af =
        reinterpret_cast<const arrfunc_type_data *>(src);
    stringstream ss;
    print_arrfunc(ss, m_src_tp.extended<arrfunc_type>(), af);
    m_dst_string_dt.extended<base_string_type>()->set_from_utf8_string(
        m_dst_arrmeta, dst, ss.str(), &m_ectx);
  }
};
} // anonymous namespace

static intptr_t make_arrfunc_to_string_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_string_dt,
    const char *dst_arrmeta, const ndt::type &src_tp, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  typedef arrfunc_to_string_ck self_type;
  self_type *self = self_type::create_leaf(ckb, kernreq, ckb_offset);
  // The kernel data owns a reference to this type
  self->m_src_tp = src_tp;
  self->m_dst_string_dt = dst_string_dt;
  self->m_dst_arrmeta = dst_arrmeta;
  self->m_ectx = *ectx;
  return ckb_offset;
}

intptr_t arrfunc_type::make_assignment_kernel(
    const arrfunc_type_data *DYND_UNUSED(self),
    const arrfunc_type *DYND_UNUSED(af_tp), void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, const ndt::type &src_tp,
    const char *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &DYND_UNUSED(kwds)) const
{
  if (this == dst_tp.extended()) {
  } else {
    if (dst_tp.get_kind() == string_kind) {
      // Assignment to strings
      return make_arrfunc_to_string_assignment_kernel(
          ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, kernreq, ectx);
    }
  }

  // Nothing can be assigned to/from arrfunc
  stringstream ss;
  ss << "Cannot assign from " << src_tp << " to " << dst_tp;
  throw dynd::type_error(ss.str());
}

bool arrfunc_type::matches(const char *arrmeta, const ndt::type &other,
                           std::map<nd::string, ndt::type> &tp_vars) const
{
  if (other.get_type_id() != arrfunc_type_id) {
    return false;
  }

  // First match the return type
  if (!m_return_type.matches(
          arrmeta, other.extended<arrfunc_type>()->m_return_type, tp_vars)) {
    return false;
  }

  // Next match all the positional parameters
  if (!m_pos_tuple.matches(arrmeta, other.extended<arrfunc_type>()->m_pos_tuple,
                           tp_vars)) {
    return false;
  }

  // Finally match all the keyword parameters
  if (!m_kwd_struct.matches(
          arrmeta, other.extended<arrfunc_type>()->get_kwd_struct(), tp_vars)) {
    return false;
  }

  return true;
}

static nd::array property_get_pos(const ndt::type &tp)
{
  return tp.extended<arrfunc_type>()->get_pos_types();
}

static nd::array property_get_kwd(const ndt::type &tp)
{
  return tp.extended<arrfunc_type>()->get_kwd_types();
}

static nd::array property_get_pos_types(const ndt::type &tp)
{
  return tp.extended<arrfunc_type>()->get_pos_types();
}

static nd::array property_get_kwd_types(const ndt::type &tp)
{
  return tp.extended<arrfunc_type>()->get_kwd_types();
}

static nd::array property_get_kwd_names(const ndt::type &tp)
{
  return tp.extended<arrfunc_type>()->get_kwd_names();
}

static nd::array property_get_return_type(const ndt::type &dt)
{
  return dt.extended<arrfunc_type>()->get_return_type();
}

void arrfunc_type::get_dynamic_type_properties(
    const std::pair<std::string, gfunc::callable> **out_properties,
    size_t *out_count) const
{
  static pair<string, gfunc::callable> type_properties[] = {
      pair<string, gfunc::callable>(
          "pos", gfunc::make_callable(&property_get_pos, "self")),
      pair<string, gfunc::callable>(
          "kwd", gfunc::make_callable(&property_get_kwd, "self")),
      pair<string, gfunc::callable>(
          "pos_types", gfunc::make_callable(&property_get_pos_types, "self")),
      pair<string, gfunc::callable>(
          "kwd_types", gfunc::make_callable(&property_get_kwd_types, "self")),
      pair<string, gfunc::callable>(
          "kwd_names", gfunc::make_callable(&property_get_kwd_names, "self")),
      pair<string, gfunc::callable>(
          "return_type",
          gfunc::make_callable(&property_get_return_type, "self"))};

  *out_properties = type_properties;
  *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}

ndt::type ndt::make_generic_funcproto(intptr_t nargs)
{
  nd::array args = ndt::make_typevar_range("T", nargs);
  args.flag_as_immutable();
  ndt::type ret = ndt::make_typevar("R");
  return ndt::make_arrfunc(ndt::make_tuple(args), ret);
}

///////// functions on the nd::array

// Maximum number of args (including out) for now
// (need to add varargs capability to this calling convention)
static const int max_args = 6;

static array_preamble *function___call__(const array_preamble *params,
                                         void *DYND_UNUSED(self))
{
  // TODO: Remove the const_cast
  nd::array par(const_cast<array_preamble *>(params), true);
  const nd::array *par_arrs =
      reinterpret_cast<const nd::array *>(par.get_readonly_originptr());
  if (par_arrs[0].get_type().get_type_id() != arrfunc_type_id) {
    throw runtime_error("arrfunc method '__call__' only works on individual "
                        "arrfunc instances presently");
  }
  // Figure out how many args were provided
  int nargs;
  nd::array args[max_args];
  for (nargs = 1; nargs < max_args; ++nargs) {
    // Stop at the first NULL arg (means it was default)
    if (par_arrs[nargs].get_ndo() == NULL) {
      break;
    } else {
      args[nargs - 1] = par_arrs[nargs];
    }
  }
  const arrfunc_type_data *af = reinterpret_cast<const arrfunc_type_data *>(
      par_arrs[0].get_readonly_originptr());
  const arrfunc_type *af_tp = par_arrs[0].get_type().extended<arrfunc_type>();
  nargs -= 1;
  // Validate the number of arguments
  if (nargs != af_tp->get_narg() + 1) {
    stringstream ss;
    ss << "arrfunc expected " << (af_tp->get_narg() + 1) << " arguments, got "
       << nargs;
    throw runtime_error(ss.str());
  }
  // Instantiate the ckernel
  ndt::type src_tp[max_args];
  for (int i = 0; i < nargs - 1; ++i) {
    src_tp[i] = args[i + 1].get_type();
  }
  const char *dynd_arrmeta[max_args];
  for (int i = 0; i < nargs - 1; ++i) {
    dynd_arrmeta[i] = args[i + 1].get_arrmeta();
  }
  ckernel_builder<kernel_request_host> ckb;
  af->instantiate(af, af_tp, &ckb, 0, args[0].get_type(), args[0].get_arrmeta(),
                  src_tp, dynd_arrmeta, kernel_request_single,
                  &eval::default_eval_context, nd::array(), std::map<nd::string, ndt::type>());
  // Call the ckernel
  expr_single_t usngo = ckb.get()->get_function<expr_single_t>();
  char *in_ptrs[max_args];
  for (int i = 0; i < nargs - 1; ++i) {
    in_ptrs[i] = const_cast<char *>(args[i + 1].get_readonly_originptr());
  }
  usngo(args[0].get_readwrite_originptr(), in_ptrs, ckb.get());
  // Return void
  return nd::empty(ndt::make_type<void>()).release();
}

void arrfunc_type::get_dynamic_array_functions(
    const std::pair<std::string, gfunc::callable> **out_functions,
    size_t *out_count) const
{
  static pair<string, gfunc::callable> arrfunc_array_functions[] = {
      pair<string, gfunc::callable>(
          "execute",
          gfunc::callable(
              ndt::type("c{self:ndarrayarg,out:ndarrayarg,p0:ndarrayarg,"
                        "p1:ndarrayarg,p2:ndarrayarg,"
                        "p3:ndarrayarg,p4:ndarrayarg}"),
              &function___call__, NULL, 3,
              nd::empty("c{self:ndarrayarg,out:ndarrayarg,p0:ndarrayarg,"
                        "p1:ndarrayarg,p2:ndarrayarg,"
                        "p3:ndarrayarg,p4:ndarrayarg}")))};

  *out_functions = arrfunc_array_functions;
  *out_count =
      sizeof(arrfunc_array_functions) / sizeof(arrfunc_array_functions[0]);
}
