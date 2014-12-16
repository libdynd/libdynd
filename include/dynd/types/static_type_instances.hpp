//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_type.hpp>

namespace dynd {

namespace init {
void static_types_init();
void static_types_cleanup();
} // namespace init

namespace types {
extern base_type *bytes_tp;
extern base_type *char_tp;
extern base_type *date_tp;
extern base_type *datetime_tp;
extern base_type *json_tp;
extern base_type *ndarrayarg_tp;
extern base_type *string_tp;
extern base_type *time_tp;
extern base_type *type_tp;
} // namespace ndt

} // namespace dynd
