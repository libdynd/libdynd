//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/callable_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/func/apply.hpp>
#include <dynd/ensure_immutable_contig.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/kernels/base_property_kernel.hpp>

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
      if (!(('0' <= c && c <= '9') || ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '_')) {
        return false;
      }
    }
    return true;
  }
}

ndt::callable_type::callable_type(const type &ret_type, const type &pos_types, const type &kwd_types)
    : base_type(callable_type_id, function_kind, sizeof(callable_type_data), scalar_align_of<uint64_t>::value,
                type_flag_zeroinit | type_flag_destructor, 0, 0, 0),
      m_return_type(ret_type), m_pos_tuple(pos_types), m_kwd_struct(kwd_types)
{
  if (m_pos_tuple.get_type_id() != tuple_type_id) {
    stringstream ss;
    ss << "dynd callable positional arg types require a tuple type, got a "
          "type \"" << m_pos_tuple << "\"";
    throw invalid_argument(ss.str());
  }
  if (m_kwd_struct.get_type_id() != struct_type_id) {
    stringstream ss;
    ss << "dynd callable keyword arg types require a struct type, got a "
          "type \"" << m_kwd_struct << "\"";
    throw invalid_argument(ss.str());
  }

  for (intptr_t i = 0, i_end = get_nkwd(); i < i_end; ++i) {
    if (m_kwd_struct.extended<tuple_type>()->get_field_type(i).get_type_id() == option_type_id) {
      m_opt_kwd_indices.push_back(i);
    }
  }

  // TODO: Should check that all the kwd names are simple identifier names
  //       because struct_type does not check that.

  // Note that we don't base the flags of this type on that of its arguments
  // and return types, because it is something the can be instantiated, even
  // for arguments that are symbolic.
}

static void print_callable(std::ostream &o, const ndt::callable_type *DYND_UNUSED(af_tp), const callable_type_data *af)
{
  o << "<callable at " << (void *)af << ">";
}

void ndt::callable_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
  const callable_type_data *af = reinterpret_cast<const callable_type_data *>(data);
  print_callable(o, this, af);
}

void ndt::callable_type::print_type(std::ostream &o) const
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
    const string &an = get_kwd_name_raw(i);
    if (is_simple_identifier_name(an.begin(), an.end())) {
      o.write(an.begin(), an.end() - an.begin());
    } else {
      print_escaped_utf8_string(o, an.begin(), an.end(), true);
    }
    o << ": " << get_kwd_type(i);
  }
  if (nkwd > 0 && m_kwd_struct.extended<struct_type>()->is_variadic()) {
    o << "...";
  }

  o << ") -> " << m_return_type;
}

void ndt::callable_type::transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                                               type &out_transformed_tp, bool &out_was_transformed) const
{
  type tmp_return_type, tmp_pos_types, tmp_kwd_types;

  bool was_transformed = false;
  transform_fn(m_return_type, arrmeta_offset, extra, tmp_return_type, was_transformed);
  transform_fn(m_pos_tuple, arrmeta_offset, extra, tmp_pos_types, was_transformed);
  transform_fn(m_kwd_struct, arrmeta_offset, extra, tmp_kwd_types, was_transformed);
  if (was_transformed) {
    out_transformed_tp = make(tmp_return_type, tmp_pos_types, tmp_kwd_types);
    out_was_transformed = true;
  } else {
    out_transformed_tp = type(this, true);
  }
}

ndt::type ndt::callable_type::get_canonical_type() const
{
  type tmp_return_type, tmp_pos_types, tmp_kwd_types;

  tmp_return_type = m_return_type.get_canonical_type();
  tmp_pos_types = m_pos_tuple.get_canonical_type();
  tmp_kwd_types = m_kwd_struct.get_canonical_type();
  return make(tmp_return_type, tmp_pos_types, tmp_kwd_types);
}

void ndt::callable_type::get_vars(std::unordered_set<std::string> &vars) const
{
  m_return_type.get_vars(vars);
  m_pos_tuple.get_vars(vars);
  m_kwd_struct.get_vars(vars);
}

ndt::type ndt::callable_type::apply_linear_index(intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
                                                 size_t DYND_UNUSED(current_i), const type &DYND_UNUSED(root_tp),
                                                 bool DYND_UNUSED(leading_dimension)) const
{
  throw type_error("Cannot store data of funcproto type");
}

intptr_t ndt::callable_type::apply_linear_index(intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
                                                const char *DYND_UNUSED(arrmeta), const type &DYND_UNUSED(result_tp),
                                                char *DYND_UNUSED(out_arrmeta),
                                                memory_block_data *DYND_UNUSED(embedded_reference),
                                                size_t DYND_UNUSED(current_i), const type &DYND_UNUSED(root_tp),
                                                bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
                                                memory_block_data **DYND_UNUSED(inout_dataref)) const
{
  throw type_error("Cannot store data of funcproto type");
}

bool ndt::callable_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    } else if (src_tp.get_type_id() == callable_type_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

bool ndt::callable_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != callable_type_id) {
    return false;
  } else {
    const callable_type *fpt = static_cast<const callable_type *>(&rhs);
    return m_return_type == fpt->m_return_type && m_pos_tuple == fpt->m_pos_tuple && m_kwd_struct == fpt->m_kwd_struct;
  }
}

void ndt::callable_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta), bool DYND_UNUSED(blockref_alloc)) const
{
}

void ndt::callable_type::arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
                                                memory_block_data *DYND_UNUSED(embedded_reference)) const
{
}

void ndt::callable_type::arrmeta_reset_buffers(char *DYND_UNUSED(arrmeta)) const
{
}

void ndt::callable_type::arrmeta_finalize_buffers(char *DYND_UNUSED(arrmeta)) const
{
}

void ndt::callable_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const
{
}

void ndt::callable_type::data_destruct(const char *DYND_UNUSED(arrmeta), char *data) const
{
  const callable_type_data *d = reinterpret_cast<callable_type_data *>(data);
  d->~callable_type_data();
}

void ndt::callable_type::data_destruct_strided(const char *DYND_UNUSED(arrmeta), char *data, intptr_t stride,
                                               size_t count) const
{
  for (size_t i = 0; i != count; ++i, data += stride) {
    const callable_type_data *d = reinterpret_cast<callable_type_data *>(data);
    d->~callable_type_data();
  }
}

/////////////////////////////////////////
// callable to string assignment

namespace {
struct callable_to_string_ck : nd::base_kernel<callable_to_string_ck, 1> {
  ndt::type m_src_tp, m_dst_string_dt;
  const char *m_dst_arrmeta;
  eval::eval_context m_ectx;

  void single(char *dst, char *const *src)
  {
    const callable_type_data *af = reinterpret_cast<const callable_type_data *>(src[0]);
    stringstream ss;
    print_callable(ss, m_src_tp.extended<ndt::callable_type>(), af);
    m_dst_string_dt.extended<ndt::base_string_type>()->set_from_utf8_string(m_dst_arrmeta, dst, ss.str(), &m_ectx);
  }
};
} // anonymous namespace

static intptr_t make_callable_to_string_assignment_kernel(void *ckb, intptr_t ckb_offset,
                                                          const ndt::type &dst_string_dt, const char *dst_arrmeta,
                                                          const ndt::type &src_tp, kernel_request_t kernreq,
                                                          const eval::eval_context *ectx)
{
  typedef callable_to_string_ck self_type;
  self_type *self = self_type::make(ckb, kernreq, ckb_offset);
  // The kernel data owns a reference to this type
  self->m_src_tp = src_tp;
  self->m_dst_string_dt = dst_string_dt;
  self->m_dst_arrmeta = dst_arrmeta;
  self->m_ectx = *ectx;
  return ckb_offset;
}

intptr_t ndt::callable_type::make_assignment_kernel(void *ckb, intptr_t ckb_offset, const type &dst_tp,
                                                    const char *dst_arrmeta, const type &src_tp,
                                                    const char *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                                    const eval::eval_context *ectx) const
{
  if (this == dst_tp.extended()) {
  } else {
    if (dst_tp.get_kind() == string_kind) {
      // Assignment to strings
      return make_callable_to_string_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, kernreq, ectx);
    }
  }

  // Nothing can be assigned to/from callable
  stringstream ss;
  ss << "Cannot assign from " << src_tp << " to " << dst_tp;
  throw dynd::type_error(ss.str());
}

bool ndt::callable_type::match(const char *arrmeta, const type &candidate_tp, const char *candidate_arrmeta,
                               std::map<std::string, type> &tp_vars) const
{
  if (candidate_tp.get_type_id() != callable_type_id) {
    return false;
  }

  // First match the return type
  if (!m_return_type.match(arrmeta, candidate_tp.extended<callable_type>()->m_return_type, candidate_arrmeta,
                           tp_vars)) {
    return false;
  }

  // Next match all the positional parameters
  if (!m_pos_tuple.match(arrmeta, candidate_tp.extended<callable_type>()->m_pos_tuple, candidate_arrmeta, tp_vars)) {
    return false;
  }

  // Finally match all the keyword parameters
  if (!m_kwd_struct.match(arrmeta, candidate_tp.extended<callable_type>()->get_kwd_struct(), candidate_arrmeta,
                          tp_vars)) {
    return false;
  }

  return true;
}

/*
static nd::array property_get_pos(const ndt::type &tp)
{
  return tp.extended<ndt::callable_type>()->get_pos_types();
}

static nd::array property_get_kwd(const ndt::type &tp)
{
  return tp.extended<ndt::callable_type>()->get_kwd_types();
}

static nd::array property_get_pos_types(const ndt::type &tp)
{
  return tp.extended<ndt::callable_type>()->get_pos_types();
}

static nd::array property_get_kwd_types(const ndt::type &tp)
{
  return tp.extended<ndt::callable_type>()->get_kwd_types();
}

static nd::array property_get_kwd_names(const ndt::type &tp)
{
  return tp.extended<ndt::callable_type>()->get_kwd_names();
}

*/

static ndt::type property_get_return_type(ndt::type tp)
{
  return tp.extended<ndt::callable_type>()->get_return_type();
}

void ndt::callable_type::get_dynamic_type_properties(const std::pair<std::string, nd::callable> **out_properties,
                                                     size_t *out_count) const
{
  struct pos_types_kernel : nd::base_property_kernel<pos_types_kernel> {
    pos_types_kernel(const ndt::type &tp, const ndt::type &dst_tp, const char *dst_arrmeta)
        : base_property_kernel<pos_types_kernel>(tp, dst_tp, dst_arrmeta)
    {
    }

    void single(char *dst, char *const *DYND_UNUSED(src))
    {
      typed_data_copy(dst_tp, dst_arrmeta, dst, tp.extended<callable_type>()->get_pos_types().get_arrmeta(),
                      tp.extended<callable_type>()->get_pos_types().get_data());
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size), char *data,
                                 ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const dynd::nd::array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      const type &tp = *reinterpret_cast<const ndt::type *>(data);
      dst_tp = make_fixed_dim(tp.extended<callable_type>()->get_npos(), make_type());
    }
  };

  struct kwd_types_kernel : nd::base_property_kernel<kwd_types_kernel> {
    kwd_types_kernel(const ndt::type &tp, const ndt::type &dst_tp, const char *dst_arrmeta)
        : base_property_kernel<kwd_types_kernel>(tp, dst_tp, dst_arrmeta)
    {
    }

    void single(char *dst, char *const *DYND_UNUSED(src))
    {
      typed_data_copy(dst_tp, dst_arrmeta, dst, tp.extended<callable_type>()->get_kwd_types().get_arrmeta(),
                      tp.extended<callable_type>()->get_kwd_types().get_data());
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size), char *data,
                                 ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const dynd::nd::array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      const type &tp = *reinterpret_cast<const ndt::type *>(data);
      dst_tp = make_fixed_dim(tp.extended<callable_type>()->get_nkwd(), make_type());
    }
  };

  struct kwd_names_kernel : nd::base_property_kernel<kwd_names_kernel> {
    kwd_names_kernel(const ndt::type &tp, const ndt::type &dst_tp, const char *dst_arrmeta)
        : base_property_kernel<kwd_names_kernel>(tp, dst_tp, dst_arrmeta)
    {
    }

    void single(char *dst, char *const *DYND_UNUSED(src))
    {
      typed_data_copy(dst_tp, dst_arrmeta, dst, tp.extended<callable_type>()->get_kwd_names().get_arrmeta(),
                      tp.extended<callable_type>()->get_kwd_names().get_data());
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size), char *data,
                                 ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const dynd::nd::array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      const type &tp = *reinterpret_cast<const ndt::type *>(data);
      dst_tp = tp.extended<callable_type>()->get_kwd_names().get_type();
    }
  };

  static pair<std::string, nd::callable> type_properties[] = {
      pair<std::string, nd::callable>("pos_types",
                                      nd::callable::make<pos_types_kernel>(type("(self: type) -> Fixed * type"))),
      pair<std::string, nd::callable>("kwd_types",
                                      nd::callable::make<kwd_types_kernel>(type("(self: type) -> Fixed * type"))),
      pair<std::string, nd::callable>("kwd_names",
                                      nd::callable::make<kwd_names_kernel>(type("(self: type) -> Fixed * Any"))),
      /*
            pair<string, nd::callable>("kwd",
                                      nd::functional::apply(&property_get_kwd)),
            pair<string, nd::callable>("pos_types",
                                      nd::functional::apply(&property_get_pos_types)),
            pair<string, nd::callable>("kwd_types",
                                      nd::functional::apply(&property_get_kwd_types)),
            pair<string, nd::callable>("kwd_names",
                                      nd::functional::apply(&property_get_kwd_names)),
      */
      pair<std::string, nd::callable>("return_type", nd::functional::apply(&property_get_return_type, "self"))};

  *out_properties = type_properties;
  *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}

ndt::type ndt::make_generic_funcproto(intptr_t nargs)
{
  return callable_type::make(typevar_type::make("R"), make_typevar_range("T", nargs));
}

///////// functions on the nd::array

// Maximum number of args (including out) for now
// (need to add varargs capability to this calling convention)
static const int max_args = 6;

static array_preamble *function___call__(const array_preamble *params, void *DYND_UNUSED(self))
{
  // TODO: Remove the const_cast
  nd::array par(const_cast<array_preamble *>(params), true);
  const nd::array *par_arrs = reinterpret_cast<const nd::array *>(par.get_readonly_originptr());
  if (par_arrs[0].get_type().get_type_id() != callable_type_id) {
    throw runtime_error("callable method '__call__' only works on individual "
                        "callable instances presently");
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
  const callable_type_data *af = reinterpret_cast<const callable_type_data *>(par_arrs[0].get_readonly_originptr());
  const ndt::callable_type *af_tp = par_arrs[0].get_type().extended<ndt::callable_type>();
  nargs -= 1;
  // Validate the number of arguments
  if (nargs != af_tp->get_narg() + 1) {
    stringstream ss;
    ss << "callable expected " << (af_tp->get_narg() + 1) << " arguments, got " << nargs;
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
  af->instantiate(NULL, 0, NULL, &ckb, 0, args[0].get_type(), args[0].get_arrmeta(), nargs, src_tp, dynd_arrmeta,
                  kernel_request_single, &eval::default_eval_context, 0, NULL, std::map<std::string, ndt::type>());
  // Call the ckernel
  expr_single_t usngo = ckb.get()->get_function<expr_single_t>();
  char *in_ptrs[max_args];
  for (int i = 0; i < nargs - 1; ++i) {
    in_ptrs[i] = const_cast<char *>(args[i + 1].get_readonly_originptr());
  }
  usngo(ckb.get(), args[0].get_readwrite_originptr(), in_ptrs);
  // Return void
  return nd::empty(ndt::type::make<void>()).release();
}

void ndt::callable_type::get_dynamic_array_functions(const std::pair<std::string, gfunc::callable> **out_functions,
                                                     size_t *out_count) const
{
  static pair<std::string, gfunc::callable> callable_array_functions[] = {pair<std::string, gfunc::callable>(
      "execute", gfunc::callable(type("{self:ndarrayarg,out:ndarrayarg,p0:ndarrayarg,"
                                      "p1:ndarrayarg,p2:ndarrayarg,"
                                      "p3:ndarrayarg,p4:ndarrayarg}"),
                                 &function___call__, NULL, 3, nd::empty("{self:ndarrayarg,out:ndarrayarg,p0:ndarrayarg,"
                                                                        "p1:ndarrayarg,p2:ndarrayarg,"
                                                                        "p3:ndarrayarg,p4:ndarrayarg}")))};

  *out_functions = callable_array_functions;
  *out_count = sizeof(callable_array_functions) / sizeof(callable_array_functions[0]);
}

nd::array callable_type_data::operator()(ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                                         const char *const *src_arrmeta, char *const *src_data, intptr_t nkwd,
                                         const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
{
  // Allocate, then initialize, the data
  std::unique_ptr<char[]> data(new char[data_size]);
  if (data_size > 0) {
    data_init(static_data, data_size, data.get(), dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
  }

  // Resolve the destination type
  if (dst_tp.is_symbolic()) {
    if (resolve_dst_type == NULL) {
      throw std::runtime_error("dst_tp is symbolic, but resolve_dst_type is NULL");
    }

    resolve_dst_type(static_data, data_size, data.get(), dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
  }

  // Allocate the destination array
  nd::array dst = nd::empty(dst_tp);

  // Generate and evaluate the ckernel
  ckernel_builder<kernel_request_host> ckb;
  instantiate(static_data, data_size, data.get(), &ckb, 0, dst_tp, dst.get_arrmeta(), nsrc, src_tp, src_arrmeta,
              kernel_request_single, &eval::default_eval_context, nkwd, kwds, tp_vars);
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();
  fn(ckb.get(), dst.get_readwrite_originptr(), src_data);

  return dst;
}

nd::array callable_type_data::operator()(ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                                         const char *const *src_arrmeta, char **const *src_data, intptr_t nkwd,
                                         const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
{
  // Allocate, then initialize, the data
  std::unique_ptr<char[]> data(new char[data_size]);
  if (data_size > 0) {
    data_init(static_data, data_size, data.get(), dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
  }

  // Resolve the destination type
  if (dst_tp.is_symbolic()) {
    if (resolve_dst_type == NULL) {
      throw std::runtime_error("dst_tp is symbolic, but resolve_dst_type is NULL");
    }

    resolve_dst_type(static_data, data_size, data.get(), dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
  }

  // Allocate the destination array
  nd::array dst = nd::empty(dst_tp);

  // Generate and evaluate the ckernel
  ckernel_builder<kernel_request_host> ckb;
  instantiate(static_data, data_size, data.get(), &ckb, 0, dst_tp, dst.get_arrmeta(), nsrc, src_tp, src_arrmeta,
              kernreq, &eval::default_eval_context, nkwd, kwds, tp_vars);
  expr_metadata_single_t fn = ckb.get()->get_function<expr_metadata_single_t>();
  fn(ckb.get(), dst.get_arrmeta(), &dst.get_ndo()->data.ptr, const_cast<char *const *>(src_arrmeta), src_data);

  return dst;
}

void callable_type_data::operator()(const ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data, intptr_t nsrc,
                                    const ndt::type *src_tp, const char *const *src_arrmeta, char *const *src_data,
                                    intptr_t nkwd, const nd::array *kwds,
                                    const std::map<std::string, ndt::type> &tp_vars)
{
  std::unique_ptr<char[]> data(new char[data_size]);
  if (data_size > 0) {
    data_init(static_data, data_size, data.get(), dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
  }

  // Generate and evaluate the ckernel
  ckernel_builder<kernel_request_host> ckb;
  instantiate(static_data, data_size, data.get(), &ckb, 0, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta,
              kernel_request_single, &eval::default_eval_context, nkwd, kwds, tp_vars);
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();
  fn(ckb.get(), dst_data, src_data);
}

void callable_type_data::operator()(const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                    char *DYND_UNUSED(dst_data), intptr_t DYND_UNUSED(nsrc),
                                    const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                                    char **const *DYND_UNUSED(src_data), intptr_t DYND_UNUSED(nkwd),
                                    const nd::array *DYND_UNUSED(kwds),
                                    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
{
}
