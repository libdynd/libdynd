//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__EVAL_CONTEXT_HPP_
#define _DND__EVAL_CONTEXT_HPP_

#include <dnd/config.hpp>
#include <dnd/dtype_assign.hpp>

namespace dnd { namespace eval {

struct eval_context {
    assign_error_mode default_assign_error_mode;

    DND_CONSTEXPR eval_context()
        : default_assign_error_mode(assign_error_fractional)
    {
    }
};

extern const eval_context default_eval_context;

}} // namespace dnd::eval

#endif // _DND__EVAL_CONTEXT_HPP_
