//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/config.hpp>
#include <dynd/types/static_type_instances.hpp>
#include <dynd/fft.hpp>

#include <math.h>

// TODO: Move elsewhere
#include <dynd/func/apply_arrfunc.hpp>
#include <dynd/types/datashape_parser.hpp>
#include <dynd/func/arrfunc_registry.hpp>

using namespace std;

bool dynd::built_with_cuda()
{
#ifdef DYND_CUDA
  return true;
#else
  return false;
#endif
}

int dynd::libdynd_init()
{
  dynd::init::static_types_init();
  dynd::init::datashape_parser_init();
  dynd::init::arrfunc_registry_init();
#ifdef DYND_FFTW
  dynd::init::fft_init();
#endif

  return 0;
}

void dynd::libdynd_cleanup()
{
#ifdef DYND_FFTW
  dynd::init::fft_cleanup();
#endif
  dynd::init::arrfunc_registry_cleanup();
  dynd::init::datashape_parser_cleanup();
  dynd::init::static_types_cleanup();
}
