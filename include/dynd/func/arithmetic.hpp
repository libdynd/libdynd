//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arrfunc.hpp>
#include <dynd/func/call.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/multidispatch.hpp>
#include <dynd/kernels/arithmetic.hpp>

namespace dynd {
namespace nd {

  template <typename F, template <type_id_t...> class K, int N>
  struct arithmetic_operator;

  template <typename F, template <type_id_t> class K>
  struct arithmetic_operator<F, K, 1> : declfunc<F> {
    static arrfunc children[DYND_TYPE_ID_MAX + 1];
    static arrfunc default_child;

    static arrfunc make()
    {
      const arrfunc self = functional::call<F>(ndt::type("(Any) -> Any"));

      for (const std::pair<const type_id_t, arrfunc> &pair :
           arrfunc::make_all<K, numeric_type_ids>()) {
        children[pair.first] = pair.second;
      }

      for (type_id_t i0 : dim_type_ids::vals()) {
        const ndt::type child_tp = ndt::arrfunc_type::make(
            {ndt::type(i0)}, self.get_type()->get_return_type());
        children[i0] = functional::elwise(child_tp, self);
      }

      return functional::multidispatch(self.get_array_type(), children,
                                       default_child);
    }
  };

  template <typename F, template <type_id_t> class K>
  arrfunc arithmetic_operator<F, K, 1>::children[DYND_TYPE_ID_MAX + 1];

  template <typename F, template <type_id_t> class K>
  arrfunc arithmetic_operator<F, K, 1>::default_child;

  extern struct plus : arithmetic_operator<plus, plus_kernel, 1> {
  } plus;

  extern struct minus : arithmetic_operator<minus, minus_kernel, 1> {
  } minus;

  template <typename F, template <type_id_t, type_id_t> class K>
  struct arithmetic_operator<F, K, 2> : declfunc<F> {
    static arrfunc children[DYND_TYPE_ID_MAX + 1][DYND_TYPE_ID_MAX + 1];
    static arrfunc default_child;

    static arrfunc make()
    {
      arrfunc self = functional::call<F>(ndt::type("(Any, Any) -> Any"));

      for (const std::pair<std::array<type_id_t, 2>, arrfunc> &pair :
           arrfunc::make_all<K, numeric_type_ids, numeric_type_ids>()) {
        children[pair.first[0]][pair.first[1]] = pair.second;
      }

      for (type_id_t i0 : numeric_type_ids::vals()) {
        for (type_id_t i1 : dim_type_ids::vals()) {
          const ndt::type child_tp =
              ndt::arrfunc_type::make({ndt::type(i0), ndt::type(i1)},
                                      self.get_type()->get_return_type());
          children[i0][i1] = functional::elwise(child_tp, self);
        }
      }

      for (type_id_t i0 : dim_type_ids::vals()) {
        typedef join<numeric_type_ids, dim_type_ids>::type type_ids;
        for (type_id_t i1 : type_ids::vals()) {
          const ndt::type child_tp =
              ndt::arrfunc_type::make({ndt::type(i0), ndt::type(i1)},
                                      self.get_type()->get_return_type());
          children[i0][i1] = functional::elwise(child_tp, self);
        }
      }

      return functional::multidispatch(self.get_array_type(), children,
                                       default_child);
    }
  };

  template <typename T, template <type_id_t, type_id_t> class K>
  arrfunc arithmetic_operator<T, K, 2>::children[DYND_TYPE_ID_MAX +
                                                 1][DYND_TYPE_ID_MAX + 1];

  template <typename T, template <type_id_t, type_id_t> class K>
  arrfunc arithmetic_operator<T, K, 2>::default_child;

  extern struct add : arithmetic_operator<add, add_kernel, 2> {
  } add;

  extern struct subtract : arithmetic_operator<subtract, subtract_kernel, 2> {
  } subtract;

  extern struct multiply : arithmetic_operator<multiply, multiply_kernel, 2> {
  } multiply;

  extern struct divide : arithmetic_operator<divide, divide_kernel, 2> {
  } divide;

} // namespace dynd::nd
} // namespace dynd