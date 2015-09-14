//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>

#ifdef DYND_USE_STD_ATOMIC
#include <atomic>
#endif

#include <dynd/config.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/date_util.hpp>

namespace dynd { namespace eval {

/**
 * Metafunction that returns true when the type is eval::eval_context
 */
template<typename T>
struct is_eval_context {
  static const bool value = false;
};

struct DYND_API eval_context {
    // If the compiler supports atomics, use them for access
    // to the evaluation context settings,
#ifdef DYND_USE_STD_ATOMIC
    // Default error mode for computations
    std::atomic<assign_error_mode> errmode;
    // Default error mode for CUDA device to device computations
    std::atomic<assign_error_mode> cuda_device_errmode;
    // Parse order of ambiguous date strings
    std::atomic<date_parse_order_t> date_parse_order;
    // Century selection for 2 digit years in date strings
    std::atomic<int> century_window;
#else
    // Default error mode for computations
    assign_error_mode errmode;
    // Default error mode for CUDA device to device computations
    assign_error_mode cuda_device_errmode;
    // Parse order of ambiguous date strings
    date_parse_order_t date_parse_order;
    // Century selection for 2 digit years in date strings
    int century_window;
#endif

    DYND_CONSTEXPR eval_context()
        : errmode(assign_error_fractional),
          cuda_device_errmode(assign_error_nocheck),
          date_parse_order(date_parse_no_ambig), century_window(70)
    {
    }

#ifdef DYND_USE_STD_ATOMIC
    // Note: the entire eval_context isn't atomic, just its pieces
    DYND_CONSTEXPR eval_context(const eval_context &rhs)
        : errmode(rhs.errmode.load()),
          cuda_device_errmode(rhs.cuda_device_errmode.load()),
          date_parse_order(rhs.date_parse_order.load()),
          century_window(rhs.century_window.load())
    {
    }

    eval_context &operator=(const eval_context &rhs) {
        errmode.store(rhs.errmode.load());
        cuda_device_errmode.store(rhs.cuda_device_errmode.load());
        date_parse_order.store(rhs.date_parse_order.load());
        century_window.store(rhs.century_window.load());
        return *this;
    }
#endif
};

template <>
struct is_eval_context<eval_context *> {
  static const bool value = true;
};

template <>
struct is_eval_context<const eval_context *> {
  static const bool value = true;
};

template <>
struct is_eval_context<eval_context *&> {
  static const bool value = true;
};

template <>
struct is_eval_context<const eval_context *&> {
  static const bool value = true;
};

extern DYND_API eval_context default_eval_context;

}} // namespace dynd::eval
