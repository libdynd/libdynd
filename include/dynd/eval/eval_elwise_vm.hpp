//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/array.hpp>
#include <dynd/memblock/memory_block.hpp>
#include <dynd/vm/elwise_program.hpp>
#include <dynd/eval/eval_context.hpp>

namespace dynd { namespace eval {

nd::array evaluate_elwise_vm(const vm::elwise_program& ep, std::vector<nd::array> inputs,
                    const eval::eval_context *ectx = &eval::default_eval_context);

}} // namespace dynd::eval
