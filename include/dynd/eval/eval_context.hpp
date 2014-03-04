//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__EVAL_CONTEXT_HPP_
#define _DYND__EVAL_CONTEXT_HPP_

#include <dynd/config.hpp>
#include <dynd/typed_data_assign.hpp>

namespace dynd { namespace eval {

struct eval_context {
    assign_error_mode default_assign_error_mode;
    assign_error_mode default_cuda_device_to_device_assign_error_mode;

    DYND_CONSTEXPR eval_context()
        : default_assign_error_mode(assign_error_fractional),
            default_cuda_device_to_device_assign_error_mode(assign_error_none)
    {
    }
};

extern const eval_context default_eval_context;

}} // namespace dynd::eval

#endif // _DYND__EVAL_CONTEXT_HPP_
