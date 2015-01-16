//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/eval/eval_elwise_vm.hpp>
#include <dynd/vm/register_allocation.hpp>
#include <dynd/shape_tools.hpp>

using namespace std;
using namespace dynd;

nd::array dynd::eval::evaluate_elwise_vm(const vm::elwise_program& ep, std::vector<nd::array> DYND_UNUSED(inputs),
                    const eval::eval_context *DYND_UNUSED(ectx))
{
    // Allocate contiguous registers for the VM
    vm::register_allocation reg(ep.get_register_types(), 0x8000, 0x8000*16);

    // Determine the result broadcast shape
    //int ndim;
    //dimvector shape;
    //broadcast_input_shapes(inputs, &ndim, &shape);


    
    return nd::array();
}
