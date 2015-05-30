//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

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
#   include <sys/param.h>
#   if defined(BSD)
#       define DYND_OS_BSD
#   else
#       error Unknown platform
#   endif
#endif


// Architecture
#if defined(DYND_OS_WINDOWS)
#   if defined(_M_X64) || defined(__x86_64__)
#       define DYND_ISA_X64
#   elif defined(_M_IX86) || defined(__i386__)
#       define DYND_ISA_X86
#   else
#       error Unsupported ISA in windows
#   endif
#else
#   if defined(__x86_64__)
#       define DYND_ISA_X64
#   elif defined(__i386__)
#       define DYND_ISA_X86
#   else
#       error Unsupported ISA configuration
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
#elif defined(DYND_OS_BSD) && defined(DYND_ISA_X64)
#   define DYND_PLATFORM_BSD_ON_X64
#elif defined(DYND_OS_BSD) && defined(DYND_ISA_X86)
#   define DYND_PLATFORM_BSD_ON_X86
#else
#   error Unsupported ISA-OS configuration
#endif

// calling convention identifiers
#if defined(DYND_PLATFORM_WINDOWS_ON_X64)
// http://msdn.microsoft.com/en-us/library/ms235286(v=vs.80).aspx
#   define DYND_CALL_MSFT_X64
#elif defined(DYND_PLATFORM_WINDOWS_ON_X32)
#   define DYND_CALL_MSFT_X32
#elif defined(DYND_PLATFORM_LINUX_ON_X64) || defined(DYND_PLATFORM_DARWIN_ON_X64) || defined(DYND_PLATFORM_BSD_ON_X64)
// http://www.x86-64.org/documentation/abi.pdf
#   define DYND_CALL_SYSV_X64
#elif defined(DYND_PLATFORM_LINUX_ON_X86) || defined(DYND_PLATFORM_BSD_ON_X86)
#   define DYND_CALL_LINUX_X32
#else
#   error unknown calling convention
#endif


// POSIX...
#if defined(DYND_OS_LINUX) || defined(DYND_OS_DARWIN)
#   define DYND_POSIX
// this one contains posix version macros
#   include <unistd.h>
#endif
