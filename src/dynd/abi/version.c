#include "dynd/abi/version.h"

dynd_size_t dynd_abi_library_version_major(void) dynd_noexcept {
  return DYND_ABI_HEADER_VERSION_MAJOR;
}

dynd_size_t dynd_abi_library_version_minor(void) dynd_noexcept {
  return DYND_ABI_HEADER_VERSION_MINOR;
}

dynd_size_t dynd_abi_library_version_debug(void) dynd_noexcept {
  return DYND_ABI_HEADER_VERSION_DEBUG;
}
