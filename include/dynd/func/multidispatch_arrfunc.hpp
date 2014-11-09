//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/array.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {

/**
 * Creates a multiple dispatch arrfunc out of a set of arrfuncs. The
 * input arrfuncs must have concrete signatures.
 *
 * \param out_af  The output arrfunc which is filled.
 * \param naf  The number of arrfuncs provided.
 * \param af  The array of input arrfuncs, sized ``naf``.
 */
void make_multidispatch_arrfunc(arrfunc_old_type_data *out_af, intptr_t naf,
                                const nd::arrfunc *af);

inline nd::arrfunc make_multidispatch_arrfunc(intptr_t naf,
                                              const nd::arrfunc *af)
{
  nd::array out_af = nd::empty(ndt::make_arrfunc());
  make_multidispatch_arrfunc(
      reinterpret_cast<arrfunc_old_type_data *>(out_af.get_readwrite_originptr()),
      naf, af);
  out_af.flag_as_immutable();
  return out_af;
}

} // namespace dynd
