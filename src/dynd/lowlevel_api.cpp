//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/lowlevel_api.hpp>

using namespace std;
using namespace dynd;

namespace {
// TODO: Return a static instance of base_type_members for the builtin types
const base_type_members *get_base_type_members(const dynd::base_type *bd)
{
  return &bd->get_base_type_members();
}

const lowlevel_api_t lowlevel_api = {
    0, // version, should increment this everytime the struct changes
    &memory_block_incref, &memory_block_decref, &detail::memory_block_free,
    &base_type_incref,    &base_type_decref,    &get_base_type_members};
} // anonymous namespace

extern "C" const void *dynd_get_lowlevel_api()
{
  return reinterpret_cast<const void *>(&lowlevel_api);
}
