//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__EVAL_CONTEXT_HPP_
#define _DYND__EVAL_CONTEXT_HPP_

#include <dynd/config.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/date_util.hpp>

namespace dynd { namespace eval {

struct eval_context {
    assign_error_mode default_assign_error_mode;
    assign_error_mode default_cuda_device_to_device_assign_error_mode;
    // Controls assumed parse order of ambiguous date strings
    date_parse_order_t date_parse_order;
    // Controls century selection of 2 digit years in date strings
    int century_window;

    DYND_CONSTEXPR eval_context()
        : default_assign_error_mode(assign_error_fractional),
          default_cuda_device_to_device_assign_error_mode(assign_error_none),
          date_parse_order(date_parse_no_ambig), century_window(70)
    {
    }
};

extern const eval_context default_eval_context;

}} // namespace dynd::eval

#endif // _DYND__EVAL_CONTEXT_HPP_
