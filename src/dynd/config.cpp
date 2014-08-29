//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/config.hpp>
#include <dynd/types/static_type_instances.hpp>
#include <dynd/fft.hpp>

#include <math.h>

// TODO: Move elsewhere
#include <dynd/func/functor_arrfunc.hpp>
#include <dynd/func/math_arrfunc.hpp>
#include <dynd/types/datashape_parser.hpp>

using namespace std;

namespace dynd { namespace math {
  nd::pod_arrfunc sin;
}}

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
#ifdef DYND_FFTW
  dynd::init::fft_init();
#endif
  math::sin.init(
      nd::make_functor_arrfunc(static_cast<double (*)(double)>(&::sin)));

  return 0;
}

void dynd::libdynd_cleanup()
{
  math::sin.cleanup();
#ifdef DYND_FFTW
  dynd::init::fft_cleanup();
#endif
  dynd::init::datashape_parser_cleanup();
  dynd::init::static_types_cleanup();
}
