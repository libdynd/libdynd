//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/kernel_prefix.hpp>
#include <dynd/kernels/kernel_builder.hpp>

using namespace std;
using namespace dynd;

void nd::kernel_builder::destroy()
{
  if (m_data != NULL) {
    // Destroy whatever was created
    reinterpret_cast<kernel_prefix *>(m_data)->destroy();
    // Free the memory
    free(m_data);
  }
}
