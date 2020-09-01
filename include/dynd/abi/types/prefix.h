#if !defined(DYND_ABI_TYPES_PREFIX_H)
#define DYND_ABI_TYPES_PREFIX_H

#include "dynd/abi/version.h"

// TODO: Expand the DYND_TYPE_IMPL macro into something more generally useful?
#define DYND_TYPE_IMPL(prefix, name) DYND_ABI(prefix##name)
#define DYND_TYPE(name) DYND_TYPE_IMPL(types_, name)

#endif // !defined(DYND_ABI_TYPES_PREFIX_H)
