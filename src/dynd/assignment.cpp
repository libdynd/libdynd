//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/assignment.hpp>
#include <dynd/callables/assign_callable.hpp>
#include <dynd/callables/copy_callable.hpp>
#include <dynd/callables/multidispatch_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/any_kind_type.hpp>

using namespace std;
using namespace dynd;

namespace {

static std::vector<ndt::type> func_ptr(const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp) {
  return {dst_tp, src_tp[0]};
}

template <typename VariadicType, template <typename, typename, VariadicType...> class T>
struct DYND_API _bind {
  template <typename Type0, typename Type1>
  using type = T<Type0, Type1>;
};

nd::callable make_assign() {
  typedef type_sequence<bool1, int8_t, int16_t, int32_t, int64_t, int128, uint8_t, uint16_t, uint32_t, uint64_t,
                        uint128, float, double, dynd::complex<float>, dynd::complex<double>>
      numeric_types;

  ndt::type self_tp = ndt::make_type<ndt::callable_type>(
      ndt::make_type<ndt::any_kind_type>(), {ndt::make_type<ndt::any_kind_type>()},
      {{ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>()), "error_mode"}});

  auto dispatcher =
      nd::callable::make_all<_bind<assign_error_mode, nd::assign_callable>::type, numeric_types, numeric_types>(
          func_ptr);
  dispatcher.insert(nd::make_callable<nd::assign_callable<dynd::string, dynd::string>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<dynd::bytes, dynd::bytes>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<ndt::fixed_bytes_type, ndt::fixed_bytes_type>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<ndt::fixed_string_type, ndt::fixed_string_type>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<ndt::fixed_string_type, ndt::fixed_string_type>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<ndt::char_type, dynd::string>>());

  //  dispatcher.insert(
  //    {{{adapt_id, ndt::make_type<ndt::any_kind_type>()}, nd::make_callable<nd::adapt_assign_to_callable>()},
  //   {{ndt::make_type<ndt::any_kind_type>(), adapt_id}, nd::make_callable<nd::adapt_assign_from_callable>()},
  // {{adapt_id, adapt_id}, nd::make_callable<nd::adapt_assign_from_callable>()}});
  dispatcher.insert(nd::make_callable<nd::assign_callable<ndt::fixed_string_type, ndt::fixed_string_type>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<dynd::string, ndt::char_type>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<ndt::type, ndt::type>>());
  dispatcher.insert(nd::make_callable<nd::string_to_int_assign_callable<int32_t>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<ndt::fixed_string_type, ndt::fixed_string_type>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<ndt::fixed_string_type, dynd::string>>());
  //  dispatcher.insert({{ndt::make_type<ndt::fixed_string_kind_type>(), ndt::make_type<uint8_t>()},
  //  callable::make<assignment_kernel<ndt::make_type<ndt::fixed_string_kind_type>(), ndt::make_type<uint8_t>()>>()});
  //  dispatcher.insert({{ndt::make_type<ndt::fixed_string_kind_type>(), ndt::make_type<uint16_t>()},
  //  callable::make<assignment_kernel<ndt::make_type<ndt::fixed_string_kind_type>(),
  //  ndt::make_type<uint16_t>()>>()});
  // dispatcher.insert({{ndt::make_type<ndt::fixed_string_kind_type>(), ndt::make_type<uint32_t>()},
  // callable::make<assignment_kernel<ndt::make_type<ndt::fixed_string_kind_type>(), ndt::make_type<uint32_t>()>>()});
  //  dispatcher.insert({{ndt::make_type<ndt::fixed_string_kind_type>(), ndt::make_type<uint64_t>()},
  //  callable::make<assignment_kernel<ndt::make_type<ndt::fixed_string_kind_type>(),
  //  ndt::make_type<uint64_t>()>>()});
  // dispatcher.insert({{ndt::make_type<ndt::fixed_string_kind_type>(), ndt::make_type<uint128>()},
  // callable::make<assignment_kernel<ndt::make_type<ndt::fixed_string_kind_type>(),
  // ndt::make_type<uint128>()>>()});
  dispatcher.insert(nd::make_callable<nd::int_to_string_assign_callable<int32_t>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<dynd::string, dynd::string>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<dynd::string, ndt::fixed_string_type>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<bool1, dynd::string>>());
  dispatcher.insert({nd::make_callable<nd::option_to_value_callable>(),
                     nd::make_callable<nd::assign_callable<ndt::option_type, ndt::option_type>>(),
                     nd::make_callable<nd::assignment_option_callable>()});
  dispatcher.insert(nd::make_callable<nd::assign_callable<ndt::option_type, dynd::string>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<ndt::option_type, ndt::float_kind_type>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<dynd::string, ndt::type>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<ndt::type, dynd::string>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<ndt::pointer_type, ndt::pointer_type>>());
  dispatcher.insert(nd::make_callable<nd::int_to_string_assign_callable<int8_t>>());
  dispatcher.insert(nd::make_callable<nd::int_to_string_assign_callable<int16_t>>());
  dispatcher.insert(nd::make_callable<nd::int_to_string_assign_callable<int32_t>>());
  dispatcher.insert(nd::make_callable<nd::int_to_string_assign_callable<int64_t>>());
  //  dispatcher.insert({{ndt::make_type<uint8_t>(), ndt::make_type<ndt::string_type>()},
  //  callable::make<assignment_kernel<ndt::make_type<uint8_t>(),
  //  ndt::make_type<ndt::string_type>()>>()});
  //  dispatcher.insert({{ndt::make_type<uint16_t>(), ndt::make_type<ndt::string_type>()},
  //  callable::make<assignment_kernel<ndt::make_type<uint16_t>(),
  //  ndt::make_type<ndt::string_type>()>>()});
  // dispatcher.insert({{ndt::make_type<uint32_t>(), ndt::make_type<ndt::string_type>()},
  // callable::make<assignment_kernel<ndt::make_type<uint32_t>(),
  // ndt::make_type<ndt::string_type>()>>()});
  // dispatcher.insert({{ndt::make_type<uint64_t>(), ndt::make_type<ndt::string_type>()},
  // callable::make<assignment_kernel<ndt::make_type<uint64_t>(),
  // ndt::make_type<ndt::string_type>()>>()});
  dispatcher.insert(nd::make_callable<nd::assign_callable<float, dynd::string>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<double, dynd::string>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<ndt::tuple_type, ndt::tuple_type>>());
  dispatcher.insert(nd::make_callable<nd::assign_callable<ndt::struct_type, ndt::struct_type>>());
  dispatcher.insert({nd::get_elwise(ndt::type("(Dim) -> Scalar")), nd::get_elwise(ndt::type("(Scalar) -> Dim")),
                     nd::get_elwise(ndt::type("(Dim) -> Dim"))});

  return nd::make_callable<nd::multidispatch_callable<2>>(self_tp, dispatcher);
}

} // anonymous namespace

DYND_API nd::callable nd::assign = make_assign();

DYND_API nd::callable nd::copy = nd::make_callable<nd::copy_callable>();
