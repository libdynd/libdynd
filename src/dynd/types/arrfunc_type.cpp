//
// Copyright (C) 2011-14 DyND Developers
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
  }
  else {
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


arrfunc_type::arrfunc_type(const ndt::type &return_type, const nd::array &arg_types, const nd::array &arg_names)
    : base_type(arrfunc_type_id, function_kind, sizeof(arrfunc_type_data),
                scalar_align_of<uint64_t>::value,
                type_flag_scalar | type_flag_zeroinit | type_flag_destructor, 0,
                0, 0), m_return_type(return_type), m_arg_types(arg_types), m_arg_names(arg_names)
{
    if (!nd::ensure_immutable_contig<ndt::type>(m_arg_types)) {
        stringstream ss;
        ss << "dynd funcproto arg types requires an array of types, got an "
              "array with type " << m_arg_types.get_type();
        throw invalid_argument(ss.str());
    }

    if (!m_arg_names.is_null() && !nd::ensure_immutable_contig<nd::string>(m_arg_names)) {
        stringstream ss;
        ss << "dynd funcproto arg names requires an array of strings, got an "
              "array with type " << m_arg_names.get_type();
        throw invalid_argument(ss.str());
    }

  // Note that we don't base the flags of this type on that of its arguments
  // and return types, because it is something the can be instantiated, even
  // for arguments that are symbolic.
}

static void print_arrfunc(std::ostream &o, const arrfunc_type *af_tp,
                          const arrfunc_type_data *DYND_UNUSED(af))
{
  o << "arrfunc with signature '" << af_tp << "'";
}

void arrfunc_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta),
                              const char *data) const
{
  const arrfunc_type_data *af =
      reinterpret_cast<const arrfunc_type_data *>(data);
  print_arrfunc(o, this, af);
}

void arrfunc_type::print_type(std::ostream& o) const
{
    const ndt::type *arg_types = get_arg_types_raw();
    intptr_t npos = get_npos();
    intptr_t narg = get_narg();

    o << "(";

    for (intptr_t i = 0; i < npos; ++i) {
        if (i > 0) {
            o << ", ";
        }

        o << arg_types[i];
    }
    for (intptr_t i = npos; i < narg; ++i) {
        if (i > 0) {
            o << ", ";
        }

        const string_type_data& an = get_arg_name_raw(i - npos);
        if (is_simple_identifier_name(an.begin, an.end)) {
            o.write(an.begin, an.end - an.begin);
        } else {
            print_escaped_utf8_string(o, an.begin, an.end, true);
        }
        o << ": " << arg_types[i];
    }

    o << ") -> " << m_return_type;
}

void arrfunc_type::transform_child_types(type_transform_fn_t transform_fn,
                                         intptr_t arrmeta_offset, void *extra,
                                         ndt::type &out_transformed_tp,
                                         bool &out_was_transformed) const
{
  const ndt::type *arg_types = get_arg_types_raw();
  std::vector<ndt::type> tmp_arg_types(get_narg());
  ndt::type tmp_return_type;

  bool was_transformed = false;
  for (size_t i = 0, i_end = get_narg(); i != i_end; ++i) {
    transform_fn(arg_types[i], arrmeta_offset, extra, tmp_arg_types[i],
                 was_transformed);
  }
  transform_fn(m_return_type, arrmeta_offset, extra, tmp_return_type,
               was_transformed);
  if (was_transformed) {
    out_transformed_tp = ndt::make_funcproto(tmp_arg_types, tmp_return_type);
    out_was_transformed = true;
  }
  else {
    out_transformed_tp = ndt::type(this, true);
  }
}

ndt::type arrfunc_type::get_canonical_type() const
{
  const ndt::type *arg_types = get_arg_types_raw();
  std::vector<ndt::type> tmp_arg_types(get_narg());
  ndt::type return_type;

  for (size_t i = 0, i_end = get_narg(); i != i_end; ++i) {
    tmp_arg_types[i] = arg_types[i].get_canonical_type();
  }
  return_type = m_return_type.get_canonical_type();

  return ndt::make_funcproto(tmp_arg_types, return_type);
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
    }
    else if (src_tp.get_type_id() == arrfunc_type_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

intptr_t arrfunc_type::get_arg_index(const char *arg_name_begin,
                                       const char *arg_name_end) const
{
    size_t size = arg_name_end - arg_name_begin;
    if (size > 0) {
        char firstchar = *arg_name_begin;
        intptr_t narg = get_narg();
        const char *fn_ptr = m_arg_names.get_readonly_originptr();
        intptr_t fn_stride =
            reinterpret_cast<const fixed_dim_type_arrmeta *>(
                m_arg_names.get_arrmeta())->stride;
        for (intptr_t i = 0; i != narg; ++i, fn_ptr += fn_stride) {
            const string_type_data *fn = reinterpret_cast<const string_type_data *>(fn_ptr);
            const char *begin = fn->begin, *end = fn->end;
            if ((size_t)(end - begin) == size && *begin == firstchar) {
                if (memcmp(fn->begin, arg_name_begin, size) == 0) {
                    return i;
                }
            }
        }
    }

    return -1;
}

/*
size_t arrfunc_type::make_assignment_kernel(
                ckernel_builder *DYND_UNUSED(ckb), size_t DYND_UNUSED(ckb_offset),
                const ndt::type& dst_tp, const char *DYND_UNUSED(dst_arrmeta),
                const ndt::type& src_tp, const char *DYND_UNUSED(src_arrmeta),
                kernel_request_t DYND_UNUSED(kernreq), assign_error_mode DYND_UNUSED(errmode),
                const eval::eval_context *DYND_UNUSED(ectx)) const
{
    throw type_error("Cannot store data of funcproto type");
}

size_t arrfunc_type::make_comparison_kernel(
                ckernel_builder *DYND_UNUSED(ckb), intptr_t DYND_UNUSED(ckb_offset),
                const ndt::type& src0_tp, const char *DYND_UNUSED(src0_arrmeta),
                const ndt::type& src1_tp, const char *DYND_UNUSED(src1_arrmeta),
                comparison_type_t comptype,
                const eval::eval_context *DYND_UNUSED(ectx)) const
{
    throw type_error("Cannot store data of funcproto type");
}
*/

bool arrfunc_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_type_id() != arrfunc_type_id) {
    return false;
  }
  else {
    const arrfunc_type *fpt = static_cast<const arrfunc_type *>(&rhs);
    return m_return_type == fpt->m_return_type && m_arg_types.equals_exact(fpt->m_arg_types);
//            && m_arg_names.equals_exact(fpt->m_arg_names);
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
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_string_dt,
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

size_t arrfunc_type::make_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp,
    const char *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
    const eval::eval_context *ectx) const
{
  if (this == dst_tp.extended()) {
  }
  else {
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

static nd::array property_get_arg_types(const ndt::type &dt)
{
  return dt.extended<arrfunc_type>()->get_arg_types();
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
          "arg_types", gfunc::make_callable(&property_get_arg_types, "self")),
      pair<string, gfunc::callable>(
          "return_type",
          gfunc::make_callable(&property_get_return_type, "self"))};

  *out_properties = type_properties;
  *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}

ndt::type ndt::make_generic_funcproto(intptr_t nargs)
{
  vector<ndt::type> args;
  ndt::make_typevar_range("T", nargs, args);
  ndt::type ret = ndt::make_typevar("R");
  return ndt::make_funcproto(args, ret);
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
    }
    else {
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
  ckernel_builder ckb;
  af->instantiate(af, af_tp, &ckb, 0, args[0].get_type(), args[0].get_arrmeta(),
                  src_tp, dynd_arrmeta, kernel_request_single,
                  &eval::default_eval_context, nd::array(), nd::array());
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
