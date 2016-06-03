//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type_registry.hpp>
#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/categorical_kind_type.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/fixed_bytes_kind_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/fixed_string_kind_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/scalar_kind_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

DYNDT_API vector<id_info> &detail::infos() {
  static vector<id_info> infos{{},
                               {"Any"},
                               {"Scalar"},
                               {"Bool"},
                               {"bool"},
                               {"Int"},
                               {"int8"},
                               {"int16"},
                               {"int32"},
                               {"int64"},
                               {"int128"},
                               {"UInt"},
                               {"uint8"},
                               {"uint16"},
                               {"uint32"},
                               {"uint64"},
                               {"uint128"},
                               {"Float"},
                               {"float16"},
                               {"float32"},
                               {"float64"},
                               {"float128"},
                               {"Complex"},
                               {"complex32"},
                               {"complex64"},
                               {"void"},
                               {"Dim"},
                               {"Bytes"},
                               {"FixedBytes"},
                               {"fixed_bytes"},
                               {"bytes"},
                               {"String"},
                               {"FixedString"},
                               {"fixed_string"},
                               {"char"},
                               {"string"},
                               {"tuple"},
                               {"struct"},
                               {"Fixed"},
                               {"fixed"},
                               {"var"},
                               {"categorical"},
                               {"option"},
                               {"pointer"},
                               {"memory"},
                               {"type"},
                               {"array"},
                               {"callable"},
                               {"Expr"},
                               {"adapt"},
                               {"expr"},
                               {"cuda_host"},
                               {"cuda_device"},
                               {"State"},
                               {""},
                               {""},
                               {""},
                               {""},
                               {""},
                               {""}};

  return infos;
}

DYNDT_API type_id_t dynd::new_id(const char *name) {
  vector<id_info> &infos = detail::infos();

  type_id_t id = static_cast<type_id_t>(infos.size());

  infos.emplace_back(name);

  return id;
}
