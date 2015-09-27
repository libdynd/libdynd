//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/callable.hpp>
#include <dynd/func/call.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/multidispatch.hpp>
#include <dynd/kernels/arithmetic.hpp>
#include <dynd/kernels/compound_add_kernel.hpp>
#include <dynd/kernels/compound_div_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename F, template <type_id_t...> class K, int N>
  struct arithmetic_operator;

  template <typename FuncType, template <type_id_t> class KernelType>
  struct arithmetic_operator<FuncType, KernelType, 1> : declfunc<FuncType> {
    static callable children[DYND_TYPE_ID_MAX + 1];

    static callable &overload(const ndt::type &src0_tp)
    {
      return children[src0_tp.get_type_id()];
    }

    static callable make()
    {
      typedef type_id_sequence<int8_type_id, int16_type_id, int32_type_id,
                               int64_type_id, float32_type_id, float64_type_id,
                               complex_float32_type_id,
                               complex_float64_type_id> numeric_type_ids;

      const callable self =
          functional::call<FuncType>(ndt::type("(Any) -> Any"));

      for (const std::pair<const type_id_t, callable> &pair :
           callable::make_all<KernelType, numeric_type_ids>(0)) {
        children[pair.first] = pair.second;
      }

      for (type_id_t i0 : dim_type_ids()) {
        const ndt::type child_tp = ndt::callable_type::make(
            self.get_type()->get_return_type(), ndt::type(i0));
        children[i0] = functional::elwise(child_tp, self);
      }

      return functional::multidispatch(
          self.get_array_type(),
          [](const ndt::type & DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
             const ndt::type * src_tp)->callable &
      {
            callable &child = overload(src_tp[0]);
            if (child.is_null()) {
              throw std::runtime_error(FuncType::what(src_tp[0]));
            }

            return child;
          },
          0);
    }
  };

  template <typename FuncType, template <type_id_t> class KernelType>
  callable arithmetic_operator<FuncType, KernelType, 1>::children
      [DYND_TYPE_ID_MAX + 1];

#define DYND_DeclUnaryOpCallable(NAME)                                                               \
  extern DYND_API struct NAME : arithmetic_operator<NAME, NAME ## _kernel, 1> {                      \
    static std::string what(const ndt::type &src0_type)                                              \
    {                                                                                                \
      std::stringstream ss;                                                                          \
      ss << "no viable overload for dynd::nd::" #NAME " with argument type \"" << src0_type << "\""; \
      return ss.str();                                                                               \
    }                                                                                                \
  } NAME;                                                                                            \

  DYND_DeclUnaryOpCallable(plus)
  DYND_DeclUnaryOpCallable(minus)

#undef DYND_DeclUnaryOpCallable

  template <typename FuncType, template <type_id_t, type_id_t> class KernelType>
  struct arithmetic_operator<FuncType, KernelType, 2> : declfunc<FuncType> {
    static callable children[DYND_TYPE_ID_MAX + 1][DYND_TYPE_ID_MAX + 1];

    static callable &overload(const ndt::type &src0_tp,
                              const ndt::type &src1_tp)
    {
      return children[src0_tp.get_type_id()][src1_tp.get_type_id()];
    }

    static void fill()
    {
      typedef type_id_sequence<int8_type_id, int16_type_id, int32_type_id,
                               int64_type_id, float32_type_id, float64_type_id,
                               complex_float32_type_id,
                               complex_float64_type_id> numeric_type_ids;

      for (const auto &pair :
           callable::make_all<KernelType, numeric_type_ids, numeric_type_ids>(
               0)) {
        children[pair.first[0]][pair.first[1]] = pair.second;
      }
      for (type_id_t i : numeric_type_ids()) {
        children[option_type_id][i] = callable::make<option_arithmetic_kernel<FuncType, true, false>>();
        children[i][option_type_id] = callable::make<option_arithmetic_kernel<FuncType, false, true>>();
      }
      children[option_type_id][option_type_id] = callable::make<option_arithmetic_kernel<FuncType, true, true>>();

      callable self =
          functional::call<FuncType>(ndt::type("(Any, Any) -> Any"));
      for (type_id_t i0 : numeric_type_ids()) {
        for (type_id_t i1 : dim_type_ids()) {
          children[i0][i1] = functional::elwise(self);
        }
      }

      for (type_id_t i0 : dim_type_ids()) {
        typedef join<numeric_type_ids, dim_type_ids>::type type_ids;
        for (type_id_t i1 : type_ids()) {
          children[i0][i1] = functional::elwise(self);
        }
      }
    }

    static callable make()
    {
      FuncType::fill();

      return functional::multidispatch(
          ndt::type("(Any, Any) -> Any"),
          [](const ndt::type & DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
             const ndt::type * src_tp)->callable &
      {
            callable &child = overload(src_tp[0], src_tp[1]);
            if (child.is_null()) {
              throw std::runtime_error(FuncType::what(src_tp[0], src_tp[1]));
            }

            return child;
          },
          0);
    }
  };

  template <typename FuncType, template <type_id_t, type_id_t> class KernelType>
  callable arithmetic_operator<FuncType, KernelType, 2>::children
      [DYND_TYPE_ID_MAX + 1][DYND_TYPE_ID_MAX + 1];

#define DYND_DeclBinaryOpCallable(NAME)                                         \
  extern DYND_API struct NAME : arithmetic_operator<NAME, NAME ## _kernel, 2> { \
    static std::string what(const ndt::type &src0_tp, const ndt::type &src1_tp) \
    {                                                                           \
      std::stringstream ss;                                                     \
      ss << "no viable overload for dynd::nd::" #NAME " with argument types \"" \
         << src0_tp << "\" and \"" << src1_tp << "\"";                          \
      return ss.str();                                                          \
    }                                                                           \
  } NAME;                                                                       \

  DYND_DeclBinaryOpCallable(add)
  DYND_DeclBinaryOpCallable(subtract)
  DYND_DeclBinaryOpCallable(multiply)
  DYND_DeclBinaryOpCallable(divide)

#undef DYND_DeclBinaryOpCallable

  template <typename FuncType, template <type_id_t, type_id_t> class KernelType>
  struct compound_arithmetic_operator : declfunc<FuncType> {
    static callable children[DYND_TYPE_ID_MAX + 1][DYND_TYPE_ID_MAX + 1];

    static void fill()
    {
        typedef type_id_sequence<int8_type_id, int16_type_id, int32_type_id,
                                 int64_type_id, float32_type_id, float64_type_id,
                                 complex_float32_type_id,
                                 complex_float64_type_id> numeric_type_ids;
      for (const auto &pair : callable::make_all<KernelType, numeric_type_ids,
                                                 numeric_type_ids>()) {
        children[pair.first[0]][pair.first[1]] = pair.second;
      }

      callable self =
          functional::call<FuncType>(ndt::type("(Any, Any) -> Any"));
      for (type_id_t i0 : numeric_type_ids()) {
        for (type_id_t i1 : dim_type_ids()) {
          children[i0][i1] = functional::elwise(self);
        }
      }

      for (type_id_t i0 : dim_type_ids()) {
        typedef join<numeric_type_ids, dim_type_ids>::type type_ids;
        for (type_id_t i1 : type_ids()) {
          children[i0][i1] = functional::elwise(self);
        }
      }
    }

    static callable make()
    {
      FuncType::fill();

      return functional::multidispatch(
          ndt::type("(Any) -> Any"),
          [](const ndt::type & dst_tp, intptr_t DYND_UNUSED(nsrc),
             const ndt::type * src_tp)->callable &
      {
            callable &child =
                children[dst_tp.get_type_id()][src_tp[0].get_type_id()];
            if (child.is_null()) {
              throw std::runtime_error("no child found");
            }

            return child;
          },
          0);
    }
  };

  template <typename FuncType, template <type_id_t, type_id_t> class KernelType>
  callable compound_arithmetic_operator<FuncType, KernelType>::children
      [DYND_TYPE_ID_MAX + 1][DYND_TYPE_ID_MAX + 1];

#define DYND_DeclCompoundOpCallable(NAME)                                                     \
  extern DYND_API struct NAME : compound_arithmetic_operator<NAME, NAME ##_kernel_t> {} NAME; \

  DYND_DeclCompoundOpCallable(compound_add)
  DYND_DeclCompoundOpCallable(compound_div)

#undef DYND_DeclCompoundOpCallable

} // namespace dynd::nd
} // namespace dynd
