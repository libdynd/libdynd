//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arrfunc.hpp>

namespace dynd {
namespace nd {

  extern struct plus : declfunc<plus> {
    static arrfunc make();
  } plus;

  extern struct minus : declfunc<minus> {
    static arrfunc make();
  } minus;

  extern struct add : declfunc<add> {
    static arrfunc make();
  } add;

  extern struct subtract : declfunc<subtract> {
    static arrfunc make();
  } subtract;

  extern struct multiply : declfunc<multiply> {
    static arrfunc make();
  } multiply;

  extern struct divide : declfunc<divide> {
    static arrfunc make();
  } divide;

} // namespace nd
} // namespace dynd