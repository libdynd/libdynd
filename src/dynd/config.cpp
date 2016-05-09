//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <regex>
#include <string>

#if __linux__ || __APPLE__
#include <dlfcn.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <dynd/config.hpp>

using namespace std;
using namespace dynd;

void dynd::load(const std::string &DYND_IGNORE_UNUSED(raw_path)) {
  static const char *shared_library_suffix = DYND_SHARED_LIBRARY_SUFFIX;

  std::string path = raw_path;

  size_t i = path.find(".");
  if (i == std::string::npos) {
    path += shared_library_suffix;
  }

  std::string name = path.substr(0, path.find("."));
  name.erase(0, 3);
  name += "_init";

#if __linux__ || __APPLE__
  void *lib = dlopen(path.c_str(), RTLD_LAZY);
  if (lib == nullptr) {
    throw runtime_error("could not load plugin");
  }

  void *sym = dlsym(lib, name.c_str());
  if (sym == nullptr) {
    dlclose(lib);
    throw runtime_error("could not load init function in plugin");
  }

  void (*init)() = reinterpret_cast<void (*)()>(sym);
  init();
#elif defined(_WIN32)
  HMODULE lib = LoadLibraryA(path.c_str());
  if (lib == nullptr) {
    throw runtime_error("Could not load DyND plugin \"" + path + "\"");
  }
  void *sym = reinterpret_cast<void *>(GetProcAddress(lib, name.c_str()));
  if (sym == nullptr) {
    FreeLibrary(lib);
    throw runtime_error("Could not load DyND plugin \"" + path + "\", it does not contain an initialization function");
  }

  void (*init)() = reinterpret_cast<void (*)()>(sym);
  init();
#endif
}
