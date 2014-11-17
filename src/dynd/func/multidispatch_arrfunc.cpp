//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <set>

#include <dynd/func/multidispatch_arrfunc.hpp>
#include <dynd/kernels/buffered_kernels.hpp>

using namespace std;
using namespace dynd;

/**
 * Placeholder hard-coded function for determining allowable
 * implicit conversions during dispatch. Allowing conversions based
 * on ``kind`` of the following forms:
 *
 * uint -> uint, where the size is nondecreasing
 * uint -> int, where the size is increasing
 * int -> int, where the size is nondecreasing
 * uint -> real, where the size is increasing
 * int -> real, where the size is increasing
 * real -> real, where the size is nondecreasing
 * real -> complex, where the size of the real component is nondecreasing
 *
 */
static bool can_implicitly_convert(const ndt::type &src, const ndt::type &dst,
                                   std::map<nd::string, ndt::type> &typevars)
{
  if (src == dst) {
    return true;
  }
  if (src.get_ndim() > 0 || dst.get_ndim() > 0) {
    ndt::type src_dtype, dst_dtype;
    if (ndt::pattern_match_dims(src, dst, typevars, src_dtype, dst_dtype)) {
      return can_implicitly_convert(src_dtype, dst_dtype, typevars);
    } else {
      return false;
    }
  }

  if (src.get_kind() == uint_kind &&
      (dst.get_kind() == uint_kind || dst.get_kind() == int_kind ||
       dst.get_kind() == real_kind)) {
    return src.get_data_size() < dst.get_data_size();
  }
  if (src.get_kind() == int_kind &&
      (dst.get_kind() == int_kind || dst.get_kind() == real_kind)) {
    return src.get_data_size() < dst.get_data_size();
  }
  if (src.get_kind() == real_kind) {
    if (dst.get_kind() == real_kind) {
      return src.get_data_size() < dst.get_data_size();
    } else if (dst.get_kind() == complex_kind) {
      return src.get_data_size() * 2 <= dst.get_data_size();
    }
  }
  return false;
}

/**
 * Returns true if every argument type in ``lhs`` is implicitly
 * convertible to the corresponding argument type in ``rhs``.
 *
 * e.g. "(int16, int16) -> int16)" and "(int32, int16) -> int32"
 */
static bool supercedes(const nd::arrfunc &lhs, const nd::arrfunc &rhs)
{
  intptr_t nargs = lhs.get_type()->get_narg();
  if (nargs == rhs.get_type()->get_narg()) {
    for (intptr_t i = 0; i < nargs; ++i) {
      const ndt::type &lpt = lhs.get_type()->get_arg_type(i);
      const ndt::type &rpt = rhs.get_type()->get_arg_type(i);
      std::map<nd::string, ndt::type> typevars;
      if (!can_implicitly_convert(lpt, rpt, typevars)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

/**
 * Returns true if every argument type in ``lhs`` is implicitly
 * convertible to the corresponding argument type in ``rhs``, or
 * has kind less than the kind of the type in ``rhs``.
 *
 * e.g. "(int16, int16) -> int16)" and "(int32, int16) -> int32"
 */
static bool toposort_edge(const nd::arrfunc &lhs, const nd::arrfunc &rhs)
{
  intptr_t nargs = lhs.get_type()->get_narg();
  if (nargs == rhs.get_type()->get_narg()) {
    for (intptr_t i = 0; i < nargs; ++i) {
      const ndt::type &lpt = lhs.get_type()->get_arg_type(i);
      const ndt::type &rpt = rhs.get_type()->get_arg_type(i);
      std::map<nd::string, ndt::type> typevars;
      if (lpt.get_kind() >= rpt.get_kind() &&
          !can_implicitly_convert(lpt, rpt, typevars)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

/**
 * Returns true if ``lhs`` and ``rhs`` are consistent, but neither
 * supercedes the other.
 *
 * e.g. "(int16, int32) -> int16)" and "(int32, int16) -> int32"
 */
static bool ambiguous(const nd::arrfunc &lhs, const nd::arrfunc &rhs)
{
  intptr_t nargs = lhs.get_type()->get_narg();
  if (nargs == rhs.get_type()->get_narg()) {
    intptr_t lsupercount = 0, rsupercount = 0;
    for (intptr_t i = 0; i < nargs; ++i) {
      const ndt::type &lpt = lhs.get_type()->get_arg_type(i);
      const ndt::type &rpt = rhs.get_type()->get_arg_type(i);
      bool either = false;
      std::map<nd::string, ndt::type> typevars;
      if (can_implicitly_convert(lpt, rpt, typevars)) {
        lsupercount++;
        either = true;
      }
      typevars.clear();
      if (can_implicitly_convert(rpt, lpt, typevars)) {
        rsupercount++;
        either = true;
      }
      if (!either) {
        return false;
      }
    }
    return (lsupercount != nargs && rsupercount != nargs) ||
           (lsupercount == rsupercount);
  }
  return false;
}

namespace {
struct toposort_marker {
  bool mark, temp_mark;
};
} // anonymous namespace

static void toposort_visit(intptr_t n, vector<toposort_marker> &marker,
                           vector<vector<intptr_t> > &adjlist, intptr_t naf,
                           const nd::arrfunc *af,
                           vector<nd::arrfunc> &sorted_af)
{
  if (marker[n].temp_mark) {
    throw invalid_argument("Detected a graph loop trying to topologically sort "
                           "arrfunc signatures for a multidispatch arrfunc");
  }
  if (!marker[n].mark) {
    marker[n].temp_mark = true;
    vector<intptr_t> &outedges = adjlist[n];
    for (vector<intptr_t>::iterator it = outedges.begin(); it != outedges.end();
         ++it) {
      toposort_visit(*it, marker, adjlist, naf, af, sorted_af);
    }
    marker[n].mark = true;
    marker[n].temp_mark = false;
    sorted_af.push_back(af[n]);
  }
}

/**
 * Does a DFS-based topological sort.
 */
static void toposort(vector<vector<intptr_t> > &adjlist, intptr_t naf,
                     const nd::arrfunc *af, vector<nd::arrfunc> &sorted_af)
{
  vector<toposort_marker> marker(naf);
  for (intptr_t n = 0; n < naf; ++n) {
    if (!marker[n].mark) {
      toposort_visit(n, marker, adjlist, naf, af, sorted_af);
    }
  }
  reverse(sorted_af.begin(), sorted_af.end());
}

/**
 * Returns all the edges with which to constrain multidispatch ordering.
 *
 * NOTE: Presently O(naf ** 2)
 */
static void get_graph(intptr_t naf, const nd::arrfunc *af,
                      vector<vector<intptr_t> > &adjlist)
{
  for (intptr_t i = 0; i < naf; ++i) {
    for (intptr_t j = 0; j < naf; ++j) {
      if (i != j && toposort_edge(af[i], af[j])) {
        if (!toposort_edge(af[j], af[i])) {
          adjlist[i].push_back(j);
        } else {
          stringstream ss;
          ss << "Multidispatch provided with two arrfunc signatures matching "
                "the same types: " << af[i].get_array_type() << " and "
             << af[j].get_array_type();
          throw invalid_argument(ss.str());
        }
      }
    }
  }
}

static void sort_arrfuncs(intptr_t naf, const nd::arrfunc *af,
                          vector<nd::arrfunc> &sorted_af)
{
  vector<vector<intptr_t> > adjlist(naf);
  get_graph(naf, af, adjlist);

  /*
  cout << "Graph is: " << endl;
  for (intptr_t i = 0; i < naf; ++i) {
    cout << i << ": ";
    for (size_t j = 0; j < adjlist[i].size(); ++j) {
      cout << adjlist[i][j] << " ";
    }
    cout << endl;
  }
  //*/
  toposort(adjlist, naf, af, sorted_af);
}

/**
 * Returns all the pairs of signatures which are ambiguous. This
 * assumes that the arrfunc array is already topologically sorted.
 *
 * NOTE: Presently O(naf ** 3)
 */
static void
get_ambiguous_pairs(intptr_t naf, const nd::arrfunc *af,
                    vector<pair<nd::arrfunc, nd::arrfunc> > &ambig_pairs)
{
  for (intptr_t i = 0; i < naf; ++i) {
    for (intptr_t j = i + 1; j < naf; ++j) {
      if (ambiguous(af[i], af[j])) {
        intptr_t k;
        for (k = 0; k < i; ++k) {
          if (supercedes(af[k], af[i]) && supercedes(af[k], af[j])) {
            break;
          }
        }
        if (k == i) {
          ambig_pairs.push_back(make_pair(af[i], af[j]));
        }
      }
    }
  }
}

static void free_multidispatch_af_data(arrfunc_type_data *self_af) {
  self_af->get_data_as<vector<nd::arrfunc> >()->~vector();
}

static intptr_t instantiate_multidispatch_af(
    const arrfunc_type_data *af_self, const arrfunc_type *DYND_UNUSED(af_tp),
    dynd::ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &args,
    const nd::array &kwds)
{
  const vector<nd::arrfunc> *icd = af_self->get_data_as<vector<nd::arrfunc> >();
  for (intptr_t i = 0; i < (intptr_t)icd->size(); ++i) {
    const nd::arrfunc &af = (*icd)[i];
    intptr_t isrc, nsrc = af.get_type()->get_npos();
    std::map<nd::string, ndt::type> typevars;
    for (isrc = 0; isrc < nsrc; ++isrc) {
      if (!can_implicitly_convert(src_tp[isrc], af.get_type()->get_arg_type(isrc),
                                  typevars)) {
        break;
      }
    }
    if (isrc == nsrc) {
      intptr_t j;
      for (j = 0; j < nsrc; ++j) {
        const ndt::type &arg_tp = af.get_type()->get_arg_type(j);
        if (!arg_tp.is_symbolic() && src_tp[j] != arg_tp) {
          break;
        }
      }
      if (j == nsrc) {
        return af.get()->instantiate(af.get(), af.get_type(), ckb, ckb_offset,
                                     dst_tp, dst_arrmeta, src_tp, src_arrmeta,
                                     kernreq, ectx, args, kwds);
      } else {
        return make_buffered_ckernel(af.get(), af.get_type(), ckb, ckb_offset,
                                     dst_tp, dst_arrmeta, nsrc, src_tp,
                                     af.get_type()->get_arg_types_raw(),
                                     src_arrmeta, kernreq, ectx);
      }
    }
  }
  // TODO: Good message here
  stringstream ss;
  ss << "No matching signature found in multidispatch arrfunc";
  throw invalid_argument(ss.str());
}

static int resolve_multidispatch_dst_type(
    const arrfunc_type_data *af_self, const arrfunc_type *DYND_UNUSED(af_tp),
    intptr_t nsrc, const ndt::type *src_tp, int throw_on_error,
    ndt::type &out_dst_tp, const nd::array &DYND_UNUSED(args),
    const nd::array &DYND_UNUSED(kwds))
{
  const vector<nd::arrfunc> *icd = af_self->get_data_as<vector<nd::arrfunc> >();
  for (intptr_t i = 0; i < (intptr_t)icd->size(); ++i) {
    const nd::arrfunc &af = (*icd)[i];
    if (nsrc == af.get_type()->get_npos()) {
      intptr_t isrc;
      std::map<nd::string, ndt::type> typevars;
      for (isrc = 0; isrc < nsrc; ++isrc) {
        if (!can_implicitly_convert(
                src_tp[isrc], af.get_type()->get_arg_type(isrc), typevars)) {
          break;
        }
      }
      if (isrc == nsrc) {
        out_dst_tp =
            ndt::substitute(af.get_type()->get_return_type(), typevars, true);
        return 1;
      }
    }
  }

  if (throw_on_error) {
    stringstream ss;
    ss << "Failed to find suitable signature in multidispatch resolution with "
          "input types (";
    for (intptr_t isrc = 0; isrc < nsrc; ++isrc) {
      ss << src_tp[isrc];
      if (isrc != nsrc - 1) {
        ss << ", ";
      }
    }
    ss << ")";
    throw type_error(ss.str());
  }
  else {
    return 0;
  }
}

nd::arrfunc dynd::make_multidispatch_arrfunc(intptr_t naf,
                                             const nd::arrfunc *child_af)
{
  if (naf <= 0) {
    throw invalid_argument(
        "Require one or more functions to create a multidispatch arrfunc");
  }
  // Number of parameters must be the same across all
  intptr_t nargs = child_af[0].get_type()->get_narg();
  for (intptr_t i = 1; i < naf; ++i) {
    if (nargs != child_af[i].get_type()->get_narg()) {
      stringstream ss;
      ss << "All child arrfuncs must have the same number of arguments to "
            "generate a multidispatch arrfunc, differing: "
         << child_af[0].get_array_type() << " and "
         << child_af[i].get_array_type();
      throw invalid_argument(ss.str());
    }
  }

  // Generate the topologically sorted array of arrfuncs
  vector<nd::arrfunc> sorted_af;
  sort_arrfuncs(naf, child_af, sorted_af);

  /*
  cout << "Before: \n";
  for (intptr_t i = 0; i < naf; ++i) {
    cout << i << ": " << af[i].get()->func_proto << endl;
  }
  cout << endl;
  cout << "After: \n";
  for (intptr_t i = 0; i < naf; ++i) {
    cout << sorted_af[i].get()->func_proto << endl;
  }
  cout << endl;
  //*/

  vector<pair<nd::arrfunc, nd::arrfunc> > ambig_pairs;
  get_ambiguous_pairs(naf, &sorted_af[0], ambig_pairs);
  if (!ambig_pairs.empty()) {
    stringstream ss;
    ss << "Arrfuncs provided to create multidispatch arrfunc have ambiguous "
          "case(s):\n";
    for (intptr_t i = 0; i < (intptr_t)ambig_pairs.size(); ++i) {
      ss << ambig_pairs[i].first.get_array_type() << " and "
         << ambig_pairs[i].second.get_array_type();
    }
    throw invalid_argument(ss.str());
  }

  // TODO: Component arrfuncs might be arrays, not just scalars
  nd::array af = nd::empty(ndt::make_generic_funcproto(nargs));
  arrfunc_type_data *out_af =
      reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
  vector<nd::arrfunc> *af_data =
      new (out_af->get_data_as<char>()) vector<nd::arrfunc>();
  out_af->free_func = &free_multidispatch_af_data;
  af_data->swap(sorted_af);
  out_af->instantiate = &instantiate_multidispatch_af;
  out_af->resolve_dst_type = &resolve_multidispatch_dst_type;
  af.flag_as_immutable();
  return af;
}
