//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type_registry.hpp>
#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/categorical_kind_type.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/fixed_bytes_kind_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/fixed_string_kind_type.hpp>
#include <dynd/types/scalar_kind_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

ndt::type_registry::type_registry()
{
  /*
      new_id(base_id_of<scalar_kind_id>::value);

      new_id(base_id_of<bool_kind_id>::value);
      new_id(base_id_of<bool_id>::value);

    new_id(base_id_of<int_kind_id>::value);
    new_id(base_id_of<int8_id>::value);
    new_id(base_id_of<int16_id>::value);
    new_id(base_id_of<int32_id>::value);
    new_id(base_id_of<int64_id>::value);
    new_id(base_id_of<int128_id>::value);

  new_id(base_id_of<uint_kind_id>::value);
  new_id(base_id_of<uint8_id>::value);
  new_id(base_id_of<uint16_id>::value);
  new_id(base_id_of<uint32_id>::value);
  new_id(base_id_of<uint64_id>::value);
  new_id(base_id_of<uint128_id>::value);

  new_id(base_id_of<float_kind_id>::value);
  new_id(base_id_of<float16_id>::value);
  new_id(base_id_of<float32_id>::value);
  new_id(base_id_of<float64_id>::value);
  new_id(base_id_of<float128_id>::value);

  new_id(base_id_of<complex_kind_id>::value);
  new_id(base_id_of<complex_float32_id>::value);
  new_id(base_id_of<complex_float64_id>::value);

  new_id(base_id_of<void_id>::value);
  */

  new_id(base_id_of<dim_kind_id>::value);

  new_id(base_id_of<bytes_kind_id>::value);
  new_id(base_id_of<fixed_bytes_id>::value);
  new_id(base_id_of<bytes_id>::value);

  new_id(base_id_of<string_kind_id>::value);
  new_id(base_id_of<fixed_string_id>::value);
  new_id(base_id_of<char_id>::value);
  new_id(base_id_of<string_id>::value);

  new_id(base_id_of<tuple_id>::value);
  new_id(base_id_of<struct_id>::value);

  new_id(base_id_of<fixed_dim_kind_id>::value);
  new_id(base_id_of<fixed_dim_id>::value);
  new_id(base_id_of<var_dim_id>::value);

  new_id(scalar_kind_id); // categorical_id
  new_id(any_kind_id);    // option_id
  new_id(any_kind_id);    // pointer_id
  new_id(any_kind_id);    // memory_id

  new_id(base_id_of<type_id>::value);
  new_id(base_id_of<array_id>::value);
  new_id(base_id_of<callable_id>::value);

  new_id(any_kind_id);  // expr_kind_id
  new_id(expr_kind_id); // adapt_id
  new_id(expr_kind_id); // expr_id

  new_id(any_kind_id); // cuda_host_id
  new_id(any_kind_id); // cuda_device_id

  new_id(any_kind_id); // kind_sym_id
  new_id(any_kind_id); // int_sym_id

  new_id(any_kind_id); // typevar_id
  new_id(any_kind_id); // typevar_dim_id
  new_id(any_kind_id); // typevar_constructed_id
  new_id(any_kind_id); // pow_dimsym_id
  new_id(any_kind_id); // ellipsis_dim_id
  new_id(any_kind_id); // dim_fragment_id
}

DYNDT_API class ndt::type_registry ndt::type_registry;

DYNDT_API vector<id_info> &detail::infos()
{
  static vector<id_info> infos{{},
                               {any_kind_id},
                               {scalar_kind_id, base_ids<scalar_kind_id>()},
                               {bool_kind_id, base_ids<bool_kind_id>()},
                               {bool_id, base_ids<bool_id>()},
                               {int_kind_id, base_ids<int_kind_id>()},
                               {int8_id, base_ids<int8_id>()},
                               {int16_id, base_ids<int16_id>()},
                               {int32_id, base_ids<int32_id>()},
                               {int64_id, base_ids<int64_id>()},
                               {int128_id, base_ids<int128_id>()},
                               {uint_kind_id, base_ids<uint_kind_id>()},
                               {uint8_id, base_ids<uint8_id>()},
                               {uint16_id, base_ids<uint16_id>()},
                               {uint32_id, base_ids<uint32_id>()},
                               {uint64_id, base_ids<uint64_id>()},
                               {uint128_id, base_ids<uint128_id>()},
                               {float_kind_id, base_ids<float_kind_id>()},
                               {float16_id, base_ids<float16_id>()},
                               {float32_id, base_ids<float32_id>()},
                               {float64_id, base_ids<float64_id>()},
                               {float128_id, base_ids<float128_id>()},
                               {complex_kind_id, base_ids<complex_kind_id>()},
                               {complex_float32_id, base_ids<complex_float32_id>()},
                               {complex_float64_id, base_ids<complex_float64_id>()},
                               {void_id, base_ids<void_id>()}};

  return infos;
}

DYNDT_API type_id_t dynd::new_id(type_id_t base_id)
{
  vector<id_info> &infos = detail::infos();

  type_id_t id = static_cast<type_id_t>(infos.size());

  vector<type_id_t> base_ids{base_id};
  for (type_id_t id : infos[base_id].base_ids) {
    base_ids.push_back(id);
  }

  infos.emplace_back(id, base_ids);

  return id;
}
