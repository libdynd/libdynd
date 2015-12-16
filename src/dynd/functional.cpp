//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <memory>
#include <set>
#include <unordered_map>

#include <dynd/functional.hpp>

using namespace std;
using namespace dynd;

/**
 * Returns true if every argument type in ``lhs`` is implicitly
 * convertible to the corresponding argument type in ``rhs``.
 *
 * e.g. "(int16, int16) -> int16)" and "(int32, int16) -> int32"
 */
static bool supercedes(const nd::callable &lhs, const nd::callable &rhs)
{
  // TODO: Deal with keyword args
  if (lhs.get_type()->get_nkwd() > 0 || rhs.get_type()->get_nkwd() > 0) {
    return false;
  }

  intptr_t npos = lhs.get_type()->get_npos();
  if (npos == rhs.get_type()->get_npos()) {
    for (intptr_t i = 0; i < npos; ++i) {
      const ndt::type &lpt = lhs.get_type()->get_pos_type(i);
      const ndt::type &rpt = rhs.get_type()->get_pos_type(i);
      std::map<std::string, ndt::type> typevars;
      if (!nd::functional::can_implicitly_convert(lpt, rpt, typevars)) {
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
static bool toposort_edge(const nd::callable &lhs, const nd::callable &rhs)
{
  // TODO: Deal with keyword args
  if (lhs.get_type()->get_nkwd() > 0 || rhs.get_type()->get_nkwd() > 0) {
    return false;
  }

  intptr_t npos = lhs.get_type()->get_npos();
  if (npos == rhs.get_type()->get_npos()) {
    for (intptr_t i = 0; i < npos; ++i) {
      const ndt::type &lpt = lhs.get_type()->get_pos_type(i);
      const ndt::type &rpt = rhs.get_type()->get_pos_type(i);
      std::map<std::string, ndt::type> typevars;
      if (lpt.get_kind() >= rpt.get_kind() && !nd::functional::can_implicitly_convert(lpt, rpt, typevars)) {
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
static bool ambiguous(const nd::callable &lhs, const nd::callable &rhs)
{
  // TODO: Deal with keyword args
  if (lhs.get_type()->get_nkwd() > 0 || rhs.get_type()->get_nkwd() > 0) {
    return false;
  }

  intptr_t npos = lhs.get_type()->get_npos();
  if (npos == rhs.get_type()->get_npos()) {
    intptr_t lsupercount = 0, rsupercount = 0;
    for (intptr_t i = 0; i < npos; ++i) {
      const ndt::type &lpt = lhs.get_type()->get_pos_type(i);
      const ndt::type &rpt = rhs.get_type()->get_pos_type(i);
      bool either = false;
      std::map<std::string, ndt::type> typevars;
      if (nd::functional::can_implicitly_convert(lpt, rpt, typevars)) {
        lsupercount++;
        either = true;
      }
      typevars.clear();
      if (nd::functional::can_implicitly_convert(rpt, lpt, typevars)) {
        rsupercount++;
        either = true;
      }
      if (!either) {
        return false;
      }
    }
    return (lsupercount != npos && rsupercount != npos) || (lsupercount == rsupercount);
  }
  return false;
}

namespace {
struct toposort_marker {
  bool mark, temp_mark;
};
} // anonymous namespace

static void toposort_visit(intptr_t n, vector<toposort_marker> &marker, vector<vector<intptr_t>> &adjlist, intptr_t naf,
                           const nd::callable *af, vector<nd::callable> &sorted_af)
{
  if (marker[n].temp_mark) {
    throw invalid_argument("Detected a graph loop trying to topologically sort "
                           "callable signatures for a multidispatch callable");
  }
  if (!marker[n].mark) {
    marker[n].temp_mark = true;
    vector<intptr_t> &outedges = adjlist[n];
    for (vector<intptr_t>::iterator it = outedges.begin(); it != outedges.end(); ++it) {
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
static void toposort(vector<vector<intptr_t>> &adjlist, intptr_t naf, const nd::callable *af,
                     vector<nd::callable> &sorted_af)
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
static void get_graph(intptr_t naf, const nd::callable *af, vector<vector<intptr_t>> &adjlist)
{
  for (intptr_t i = 0; i < naf; ++i) {
    for (intptr_t j = 0; j < naf; ++j) {
      if (i != j && toposort_edge(af[i], af[j])) {
        if (!toposort_edge(af[j], af[i])) {
          adjlist[i].push_back(j);
        }
        else {
          stringstream ss;
          ss << "Multidispatch provided with two callable signatures matching "
                "the same types: "
             << af[i].get_array_type() << " and " << af[j].get_array_type();
          throw invalid_argument(ss.str());
        }
      }
    }
  }
}

static void sort_callables(intptr_t naf, const nd::callable *af, vector<nd::callable> &sorted_af)
{
  vector<vector<intptr_t>> adjlist(naf);
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
 * assumes that the callable array is already topologically sorted.
 *
 * NOTE: Presently O(naf ** 3)
 */
static void get_ambiguous_pairs(intptr_t naf, const nd::callable *af,
                                vector<pair<nd::callable, nd::callable>> &ambig_pairs)
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

nd::callable nd::functional::old_multidispatch(intptr_t naf, const callable *child_af)
{
  if (naf <= 0) {
    throw invalid_argument("Require one or more functions to create a multidispatch callable");
  }
  // Number of parameters must be the same across all
  intptr_t nargs = child_af[0].get_type()->get_narg();
  for (intptr_t i = 1; i < naf; ++i) {
    if (nargs != child_af[i].get_type()->get_narg()) {
      stringstream ss;
      ss << "All child callables must have the same number of arguments to "
            "generate a multidispatch callable, differing: "
         << child_af[0].get_array_type() << " and " << child_af[i].get_array_type();
      throw invalid_argument(ss.str());
    }
  }

  // Generate the topologically sorted array of callables
  vector<nd::callable> sorted_af;
  sort_callables(naf, child_af, sorted_af);

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

  vector<pair<nd::callable, nd::callable>> ambig_pairs;
  get_ambiguous_pairs(naf, &sorted_af[0], ambig_pairs);
  if (!ambig_pairs.empty()) {
    stringstream ss;
    ss << "Arrfuncs provided to create multidispatch callable have ambiguous "
          "case(s):\n";
    for (intptr_t i = 0; i < (intptr_t)ambig_pairs.size(); ++i) {
      ss << ambig_pairs[i].first.get_array_type() << " and " << ambig_pairs[i].second.get_array_type();
    }
    throw invalid_argument(ss.str());
  }

  // TODO: Component callables might be arrays, not just scalars
  return callable::make<old_multidispatch_ck>(ndt::make_generic_funcproto(nargs), std::move(sorted_af));
}
