#include "dynd/abi/version.h"

DYND_SIZE_T dynd_abi_library_version_major(void) DYND_NOEXCEPT {
  return DYND_ABI_HEADER_VERSION_MAJOR;
}

DYND_SIZE_T dynd_abi_library_version_minor(void) DYND_NOEXCEPT {
  return DYND_ABI_HEADER_VERSION_MINOR;
}

DYND_SIZE_T dynd_abi_library_version_debug(void) DYND_NOEXCEPT {
  return DYND_ABI_HEADER_VERSION_DEBUG;
}
