//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {

  /*
    Concatenates `nop` strings in the `s` array, storing the result in
    `d`
  */
  void string_concat(size_t nop, dynd::string *d, const dynd::string *const *s);

  /*
    Returns the number of times needle appears in haystack.
  */
  intptr_t string_count(const dynd::string *const haystack,
                        const dynd::string *const needle);

  /*
    Returns byte index of the first occurrence of needle in haystack.
    Returns -1 if not found.
  */
  intptr_t string_find(const dynd::string *const haystack,
                       const dynd::string *const needle);

  /*
    In string `src`, replace all non-overlapping appearances of
    `old_str` with `new_str`, storing the result in `dst`.
  */
  void string_replace(dynd::string *const dst,
                      const dynd::string *const src,
                      const dynd::string *const old_str,
                      const dynd::string *const new_str);

  namespace nd {

    extern DYND_API struct DYND_API string_concatenation : declfunc<string_concatenation> {
      static callable make();
    } string_concatenation;

    extern DYND_API struct DYND_API string_count : declfunc<string_count> {
      static callable make();
    } string_count;

    extern DYND_API struct DYND_API string_find : declfunc<string_find> {
      static callable make();
    } string_find;

    extern DYND_API struct DYND_API string_replace : declfunc<string_replace> {
      static callable make();
    } string_replace;

  } // namespace dynd::nd
} // namespace dynd
