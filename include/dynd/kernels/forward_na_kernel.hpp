//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/option.hpp>
#include <dynd/types/option_type.hpp>

namespace dynd {
namespace nd {

  template <int... I>
  struct forward_na_kernel;

  template <int I>
  struct forward_na_kernel<I> : base_strided_kernel<forward_na_kernel<I>, 2> {
    constexpr size_t size() const { return sizeof(forward_na_kernel) + 2 * sizeof(size_t); }

    void single(char *res, char *const *args)
    {
      // Check if args[I] is not available
      bool1 is_na;
      this->get_child()->single(reinterpret_cast<char *>(&is_na), args + I);

      if (is_na) {
        // assign_na
        this->template get_child<2>()->single(res, nullptr);
      }
      else {
        // call the actual child
        this->template get_child<1>()->single(res, args);
      }
    }
  };

  template <bool Src0IsOption, bool Src1IsOption>
  struct option_comparison_kernel;

  template <>
  struct option_comparison_kernel<true, true> : base_strided_kernel<option_comparison_kernel<true, true>, 2> {
    intptr_t is_na_rhs_offset;
    intptr_t comp_offset;
    intptr_t assign_na_offset;

    void single(char *dst, char *const *src)
    {
      auto is_na_lhs = this->get_child();
      auto is_na_rhs = this->get_child(is_na_rhs_offset);
      bool child_dst_lhs;
      bool child_dst_rhs;
      is_na_lhs->single(reinterpret_cast<char *>(&child_dst_lhs), &src[0]);
      is_na_rhs->single(reinterpret_cast<char *>(&child_dst_rhs), &src[1]);
      if (!child_dst_lhs && !child_dst_rhs) {
        this->get_child(comp_offset)->single(dst, src);
      }
      else {
        this->get_child(assign_na_offset)->single(dst, nullptr);
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
