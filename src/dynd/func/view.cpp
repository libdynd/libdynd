//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/view.hpp>
#include <dynd/callables/view_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::view::make() { return make_callable<view_callable>(); }

DYND_DEFAULT_DECLFUNC_GET(nd::view)

DYND_API struct nd::view nd::view;
