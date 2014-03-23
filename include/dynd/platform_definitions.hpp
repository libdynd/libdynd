//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef __PLATFORM_DEFINITIONS_H_
#define __PLATFORM_DEFINITIONS_H_

// platform macros
// based on info found in
// http://sourceforge.net/apps/mediawiki/predef/index.php?title=Main_Page


// Operating System
#if defined(_WIN32)
// platform is a Windows variant, either win32 or win64
////////////////////////////////////////////////////////////////////////////////
#   define DYND_OS_WINDOWS

#elif defined(__linux__)
// platform is a Linux
#   define DYND_OS_LINUX
#elif defined(__APPLE__) && defined(__MACH__)
// platform is mac os x (darwin)
#   define DYND_OS_DARWIN
#else
#   error Unknown platform
#endif


// Architecture
#if defined(DYND_OS_WINDOWS)
#   if defined(_M_X64)
#       define DYND_ISA_X64
#   elif defined(_M_IX86)
#       define DYND_ISA_X86
#   elif
#       error Unsupported ISA in windows
#   endif
#elif defined(DYND_OS_DARWIN) || defined (DYND_OS_LINUX)
#   if defined(__x86_64__)
#       define DYND_ISA_X64
#   elif defined(__i386__)
#       define DYND_ISA_X86
#   else
#       error Unsupported ISA in darwin/linux
#   endif
#endif


// convenience OS-ISA combinations
#if defined(DYND_OS_WINDOWS) && defined(DYND_ISA_X64)
#   define DYND_PLATFORM_WINDOWS_ON_X64
#elif defined(DYND_OS_WINDOWS) && defined(DYND_ISA_X86)
#   define DYND_PLATFORM_WINDOWS_ON_X32
#elif defined(DYND_OS_LINUX) && defined(DYND_ISA_X64)
#   define DYND_PLATFORM_LINUX_ON_X64
#elif defined(DYND_OS_LINUX) && defined(DYND_ISA_X86)
#   define DYND_PLATFORM_LINUX_ON_X86
#elif defined(DYND_OS_DARWIN) && defined(DYND_ISA_X64)
#   define DYND_PLATFORM_DARWIN_ON_X64
#elif defined(DYND_OS_DARWIN) && defined(DYND_ISA_X86)
#   define DYND_PLATFORM_DARWIN_ON_X86
#else
#   error Unsupported ISA-OS configuration
#endif

// calling convention identifiers
#if defined(DYND_PLATFORM_WINDOWS_ON_X64)
// http://msdn.microsoft.com/en-us/library/ms235286(v=vs.80).aspx
#   define DYND_CALL_MSFT_X64
#elif defined(DYND_PLATFORM_WINDOWS_ON_X32)
#   define DYND_CALL_MSFT_X32
#elif defined(DYND_PLATFORM_LINUX_ON_X64) || defined(DYND_PLATFORM_DARWIN_ON_X64)
// http://www.x86-64.org/documentation/abi.pdf
#   define DYND_CALL_SYSV_X64
#elif defined(DYND_PLATFORM_LINUX_ON_X86)
#define DYND_CALL_LINUX_X32
#else
#   error unknown calling convention
#endif


// POSIX...
#if defined(DYND_OS_LINUX) || defined(DYND_OS_DARWIN)
#   define DYND_POSIX
// this one contains posix version macros
#   include <unistd.h>
#endif


#endif // _PLATFORM_DEFINITIONS_H_
