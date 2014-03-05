//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/config.hpp>

bool dynd::built_with_cuda()
{
#ifdef DYND_CUDA
    return true;
#else
    return false;
#endif
}
