#include <assert.h>

#include "dynd/abi/resource.h"
extern "C" {

DYND_ABI_EXPORT void dynd_abi_resource_never_release(dynd_resource*) noexcept {
  assert(false);
}

}
