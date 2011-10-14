//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__CONFIG_HPP_
#define _DND__CONFIG_HPP_

#ifdef __clang__

//#  define DND_RVALUE_REFS
//#  define DND_INIT_LIST

#define USE_BOOST_SHARED_PTR

#else

// TODO: Specific gcc versions with rvalue ref support
#  define DND_RVALUE_REFS
// TODO: Specific gcc versions with initializer list support
#  define DND_INIT_LIST

#endif

#ifdef DND_RVALUE_REFS
#  define DND_MOVE(x) (std::move(x))
#else
#  define DND_MOVE(x) (x)
#endif

#ifdef USE_BOOST_SHARED_PTR
#include <boost/shared_ptr.hpp>
namespace dnd {
    using ::boost::shared_ptr;
}
#else
#include <memory>
namespace dnd {
    using ::std::shared_ptr;
}
#endif

#ifdef DND_INIT_LIST
#include <initializer_list>
#endif

#endif // _DND__CONFIG_HPP_
